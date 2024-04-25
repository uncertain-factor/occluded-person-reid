import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, SequentialSampler

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .occ_reid import OCC_ReID
from .vehicleid import VehicleID
from .veri import VeRi

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'occ_reid': OCC_ReID
}


# 对每个批次的数据进行合并，合并每个维度的数据形成一维张量
def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    # 数据集item为（文件路径，行人id，相机id，1）
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


# 对图像进行预处理
def make_dataloader(cfg):
    # 对图像的非常规预处理函数，进行了图像翻转，填充，擦除，裁剪等操作
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),  # 设置图像尺寸，使用双三次插值（Bicubic interpolation）作为缩放算法。
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),  # 以一定概率翻转图像
        T.Pad(cfg.INPUT.PADDING),  # 对图像的上下左右边界之外进行像素填充
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),  # 从图像中随机裁剪出一定尺寸的新图像
        T.ToTensor(),  # 将图像的像素矩阵转变为张量，并将0-255映射到0-1（归一化）
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),  # 输入均值和标准差进行标准化
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),  # 以一定概率随机擦除随机数量的像素
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    # 对图像的常规预处理函数
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),  # 重设置尺寸为与输入一致
        T.ToTensor(),  # 将图像的像素矩阵转变为张量，并将0-255映射到0-1（归一化）
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)  # 输入均值和标准差进行标准化
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    # 进行了非常规处理的全身图训练集(用于阶段二)
    train_set_whole = ImageDataset(dataset.whole_train, train_transforms)
    # 进行了常规处理的遮挡图训练集（用于阶段二）
    train_set_occ = ImageDataset(dataset.occ_train, val_transforms)
    # 进行了常规处理的训练集（用于阶段一）
    train_set_whole_normal = ImageDataset(dataset.whole_train, val_transforms)
    num_classes = dataset.num_train_pids    # 训练集行人id数
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        # 为训练阶段2创建数据加载器，顺序取样，每批次样本为25
        # 全身图训练数据集，
        whole_train_loader_stage2 = DataLoader(
            train_set_whole,
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            sampler=SequentialSampler(dataset.whole_train),
            collate_fn=train_collate_fn
        )
        # 遮挡图训练数据集
        occ_train_loader_stage2 = DataLoader(
            train_set_occ,
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            sampler=SequentialSampler(dataset.occ_train),
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    # 将查询集（query）和图库集（gallery）连接起来，形成一个新的验证集（validation set）
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    # 为验证集创建数据加载器，图像数据经过常规预处理，每批次30个样本，不打乱，加载数据的子进程数量为8，
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    # 为训练阶段1创建数据加载器，图像数据经过常规预处理，每批次30个样本，打乱，加载数据的子进程数量为8
    train_loader_stage1 = DataLoader(
        train_set_whole_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    )
    return whole_train_loader_stage2, occ_train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num
