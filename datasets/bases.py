from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """
    #item为【图像绝对路径，行人id，相机id，1】，获取数据集的行人种类数，图像样本数，相机种类数，1
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError

# 总数据集，可统计训练集，测试集和图库集信息
class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

# 从数据集路径读取图片数据并应用转化函数
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    # 根据索引获取样本对象的详细信息（图片数据，行人id，相机id，图片名）
    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]


# 成对的图片数据集，输入样本索引，返回成对的图片
class PairImageDataset(Dataset):
    def __init__(self, dataset1, dataset2, transform1=None, transform2=None):
        self.dataset1 = dataset1
        self.transform1 = transform1
        self.dataset2 = dataset2
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataset1)

    # 根据索引获取样本对象的详细信息（图片数据1，图片数据2，行人id，相机id，图片名）
    def __getitem__(self, index):
        img_path1, pid1, camid1, trackid1 = self.dataset1[index]
        img_path2, _, _, _ = self.dataset2[index]
        img1 = read_image(img_path1)
        img2 = read_image(img_path2)
        if self.transform1 is not None:
            img1 = self.transform(img1)
        if self.transform2 is not None:
            img2 = self.transform(img2)
        return img1, img2, pid1, camid1, trackid1, img_path1.split('/')[-1]
