import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def do_train_stage1(cfg,
                    model,
                    train_loader_stage1,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD  # 50

    logger = logging.getLogger("transreid.train")
    logger.info('start training stage1')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)  # 在对应的显卡上训练
        # no
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()  # 损失计量器，内部记录了损失的当前值，平均值，总和与批次数量
    scaler = amp.GradScaler()  # amp.GradScaler() 是自动混合精度的训练组件，主要作用是自动调整梯度的大小，以便在反向传播时避免梯度下溢（underflow）或上溢（overflow）
    xent = SupConLoss(device)  # 对比损失函数

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()  # 启动秒表，记录时间
    # logger.info("model: {}".format(model))
    image_features = []  # 图像特征
    labels = []  # 标签
    with torch.no_grad():  # 不追踪梯度
        # 获取每个批次的数据，包括图像，行人id，相机id，视图
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            # print(vid)
            # 将当前batch的图像数据和行人id（即标签）移动到设备上
            img = img.to(device)
            target = vid.to(device)
            # 使用自动混合精度（AMP）上下文管理器，减少内存消耗，加快训练
            with amp.autocast(enabled=True):
                # 提取当前batch的图像特征
                image_feature = model(img, target, get_image=True)
                # print("debug point2")
                # 将标签和图像特征加入labels和image_features列表，其中图像特征保存在cpu
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
                # print("debug point2")
        # 将各个批次的图像标签堆叠成张量，然后移动到gpu，形状为[batch_num*batch_size,label_dim]
        labels_list = torch.stack(labels, dim=0).cuda()
        # 将各个批次的图像特征堆叠成张量，然后移动到gpu，形状为[batch_num*batch_size,feat_dim]
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH  # batch_size
        num_image = labels_list.shape[0]  # 图片总数
        i_ter = num_image // batch  # batch_num
    del labels, image_features  # 删除引用，释放内存
    # 对于每个epoch，从1到epoch
    for epoch in range(1, epochs + 1):
        # print("epoch: " + str(epoch))
        loss_meter.reset()  # 重置损失计量器
        scheduler.step(epoch)  # 调度器根据周期数调整优化器的学习率
        model.train()  # 将模型设置为训练模式，确保所有的层（如Dropout和BatchNorm）都以训练模式运行。
        # 将索引打乱后存放在列表中，并移动到指定设备上
        iter_list = torch.randperm(num_image).to(device)
        # 对于每个batch，从0到batch_num, execute batch_num times in total
        for i in range(i_ter):
            # print("batch: "+str(i)+"start")
            optimizer.zero_grad()
            # 按顺序取出batch_size个索引，对于最后一个batch，数量可能不足batch_size
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]
            # 根据索引可找到在labels_list中对应的图像标签和图像特征，即提取当前批次的图像标签和特征
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            # print(target)
            with amp.autocast(enabled=True):
                # 获取图像标签对应的相同数量的文本特征
                text_features = model(label=target, get_text=True)
            # print("debug point3")
            # 计算两个方向的损失（从图像到文本和从文本到图像）
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)
            # print("debug point4")
            loss = loss_i2t + loss_t2i
            # 使用自动混合精度的scaler对象
            scaler.scale(loss).backward()  # 按比例缩放loss，然后计算并反向传播损失的梯度
            scaler.step(optimizer)  # 根据梯度更新优化器，同时更新模型参数
            scaler.update()  # 更新缩放因子
            # 将损失和批次大小记录到损失计量器
            loss_meter.update(loss.item(), img.shape[0])
            # 确保所有CUDA操作都已完成。
            torch.cuda.synchronize()
            # 每50个batch输出一次日志，记录损失
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))
            # print("batch: " + str(i) + "end")
        # 每checkpoint_period（120）个周期，保存模型的当前状态字典。
        if epoch % checkpoint_period == 0:
            # no 分布式训练
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
    # 记录时间
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
