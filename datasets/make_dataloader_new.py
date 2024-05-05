import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, SequentialSampler

from .bases import ImageDataset, PairImageDataset
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


# 将一个批次里所有样本的各维度数据堆叠成列表或张量，在数据加载器加载一个批次的数据时被调用
# 训练阶段二训练数据集的批处理函数
def train2_collate_fn(batch):
    # 返回关于图片数据，行人id，相机id，视图id的张量
    imgs1, imags2, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs1, dim=0), torch.stack(imags2, dim=0), pids, camids, viewids,

# 训练阶段一训练数据集的批处理函数
def train1_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    # 返回关于图片数据，行人id，相机id，视图id的张量
    imgs, pids, camids, viewids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

# 验证集数据加载器的批处理函数
def val_collate_fn(batch):
    # 返回图片数据张量，行人id张量，相机id列表，相机id张量，图片名字张量
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
    # 阶段二的成对训练集，全身集dataset1非常规处理，遮挡集dataset2常规处理
    train_set = PairImageDataset(dataset1=dataset.whole_train,
                                 dataset2=dataset.occ_train,
                                 transform1=train_transforms,
                                 transform2=val_transforms)
    # 进行了常规处理的训练集（用于阶段一）
    train_set_whole_normal = ImageDataset(dataset.whole_train, val_transforms)

    num_classes = dataset.num_train_pids    # 训练集行人id数
    cam_num = dataset.num_train_cams    # 训练集相机数
    view_num = dataset.num_train_vids   # 训练集视图数

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        # 为训练阶段2创建数据加载器，数据集为PairImageDataSet,随机取样，每批次样本为50
        train_loader_stage2 = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.whole_train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                          cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers,  # 加载数据的子进程数量为5
            collate_fn=train2_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    # 将查询集（query）和图库集（gallery）连接起来，形成一个新的验证集（validation set）
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    # 为验证集创建数据加载器，图像数据经过常规预处理，每批次50个样本，不打乱，加载数据的子进程数量为5，
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    # 为训练阶段1创建数据加载器，图像数据经过常规预处理，每批50个样本，打乱，加载数据的子进程数量为5
    train_loader_stage1 = DataLoader(
        train_set_whole_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train1_collate_fn
    )
    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num
