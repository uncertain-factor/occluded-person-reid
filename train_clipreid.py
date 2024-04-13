from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg


# 为随机数生成器设置种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # 为该脚本添加命令行参数 --config_file  opts  --local_rank  并将运行时的动态参数保存进args对象
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # 将配置文件和命令行参数到配置对象cfg中，并冻结配置
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    # 设置显卡
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    # 设置输出目录
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 创建名为transreid的日志记录器，并将日志输出在outputdir中
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    # 加载配置文件，将配置信息输出到日志
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # 是否进行分布式训练
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    # 获取训练集，训练normal集和val验证集（query+gallery）的数据加载器，测试集的样本数量，行人数，相机数，视图个数（1）
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    # 根据配置文件，行人，相机，视图个数建立模型，模型包含image部分（image encoder）和text部分（prompt learning和text encoder），可提取图像和文本特征
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    # 两个损失函数，一个是loss_func（由三元组损失，交叉熵损失，图文相似度损失组成），一个是中心损失center_criterion
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    # 创建训练阶段1的adam优化器（针对prompt learning的优化器）和基于多步预热的优化器学习率调度器
    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
                                        lr_min=cfg.SOLVER.STAGE1.LR_MIN,
                                        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
                                        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range=None)
    # 该阶段创建对比学习损失xent，然后在每个epoch内，按批次提取出图像标签，特征和文本特征，计算图像和文本的彼此对比学习损失作为该阶段的训练损失函数，训练prompt learning的参数，输出日志。
    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )
    # make_optimizer_2stage函数，创建第二阶段的adam优化器和中心损失的优化器
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
    # 针对adam优化器创建学习率调度器
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA,
                                         cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                         cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)
    # 训练阶段2
    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank
    )
