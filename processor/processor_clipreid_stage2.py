import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def do_train_stage2(cfg,
                    model,
                    center_criterion,   # 该参数无作用
                    train_loader_stage2,
                    val_loader,
                    optimizer,
                    optimizer_center,
                    scheduler,
                    loss_fn,
                    num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD  # 日志输出周期
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD  # 训练检查点周期
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD  # 模型评估周期
    instance = cfg.DATALOADER.NUM_INSTANCE  # 数据加载器实例数

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS  # epoch数

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        # 将模型移动到指定设备
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes  # 指定行人总数
    # 损失和准确率的计量器
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    # 创建一个评估器对象，用于评估模型性能。输入测试集的样本数量，，yes
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()  # 创建一个用于混合精度训练的 GradScaler 对象
    xent = SupConLoss(device)  # 对比学习损失函数

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()  # 时间记录器

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH  # batch大小
    # 计算实际批次i_ter
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    # 文本特征
    text_features = []
    with torch.no_grad():
        # 该部分不计算梯度
        for i in range(i_ter):
            # 根据批次获取索引列表
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            # 采用混合进度上下文管理，减小内存
            with amp.autocast(enabled=True):
                # 提取每一批次的文本特征（经过阶段1优化的），并按顺序加入列表text_features
                text_feature = model(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        # 将所有文本特征连接起来，然后移动到cuda
        text_features = torch.cat(text_features, 0).cuda()
    # 进行每个epoch的循环训练
    for epoch in range(1, epochs + 1):
        start_time = time.time()  # 开始计时
        loss_meter.reset()  # 重置损失
        acc_meter.reset()  # 重置正确率
        evaluator.reset()  # 重置评估器
        scheduler.step()  # 调整优化器学习率

        model.train()  # 将模型设计为训练模式
        # 按批次从数据加载器取出数据，其中图像数据经过了非常规预处理，然后训练模型
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            # 清除优化器的梯度
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            # 将图像和图像标签移动到设备上
            img = img.to(device)
            target = vid.to(device)
            # igonre
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            # 使用自动混合精度（Automatic Mixed Precision，AMP）进行计算，加速训练。
            with amp.autocast(enabled=True):
                # 输入图像数据和标签得到一批图像数据预测真实标签的预测分数logits，
                # 组合特征（图像编码器最后一个transformer输出，全部transformer输出，投影降维输出）和 投影降维图像特征
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view)
                # 这里计算了投影降维图像特征与文本特征之间的点积得到图像到文本类别的预测logits。
                logits = image_features @ text_features.t()
                # 使用前面计算得到的图像到标签的预测分数logits、组合特征、真实标签、以及图像到文本类别的logits，计算阶段二的损失函数。
                # 其中用score和target计算图像到真实标签的交叉熵损失，用feat和target计算图像特征之间的三元组损失，用logits计算图像到文本的交叉熵损失
                loss = loss_fn(score, feat, target, target_cam, logits)
            # 反向传播
            scaler.scale(loss).backward()
            # 更新优化器,同时更新模型权重参数，这里优化的参数主要是clip的图像编码器
            scaler.step(optimizer)
            scaler.update()  # 在每个批次后更新缩放因子
            # no
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            # 计算图像预测文本标签平均正确率
            acc = (logits.max(1)[1] == target).float().mean()
            # 更新损失
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)  # 更新正确率
            # 同步cuda
            torch.cuda.synchronize()
            # 每到一定批次输出日志
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        # 计算耗时
        end_time = time.time()
        # 平均每批次耗时
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            # 输出该epoch的每批次耗时
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))
        # 每训练一定的epoch保存模型
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:

                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        # 每迭代eval_period个epoch
        if epoch % eval_period == 0:
            # no
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()  # 评估模式
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:  # yes
                model.eval()  # 评估模式
                # 遍历验证集每个批次的数据
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    # 不计算梯度
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else:  # yes
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:  # yes
                            target_view = None
                        # 获取图像经过所有transformer层后的特征与投影层后的特征（皆取第一个位置的向量）拼接作为feat
                        feat = model(img, cam_label=camids, view_label=target_view)
                        # 用该批次的feat特征，真实标签，相机id更新评估器状态
                        evaluator.update((feat, vid, camid))
                # 在所有验证集数据批次处理完成后，调用评估器的 compute() 方法计算性能指标。这些性能指标包括累积匹配曲线 (CMC) 和平均准确率 (mAP)。
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                # 对于排名为 1、5 和 10 的位置，记录累积匹配曲线 (CMC) 的结果。
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                # 清空 CUDA 设备上的缓存，释放内存。
                torch.cuda.empty_cache()
    # 计算总耗时，打印
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
