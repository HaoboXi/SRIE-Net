from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
import cv2
import os



ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
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
    def get_imagedata_info(self, data):
        pids, cams, tracks, clos = [], [], [], []

        for _, pid, camid, trackid , mask_path in data:
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


class BaseImageDataset(BaseDataset):
    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # clos")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:6d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:6d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:6d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None , mask = False):
        self.dataset = dataset
        self.transform = transform
        self.mask = mask
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid , msk_path = self.dataset[index]
        img = read_image(img_path)
        msk = np.zeros(1)
        if self.transform is not None:
            img = self.transform(img)
        if self.mask == True:
            if msk_path.find('.npy') != -1:
                C, H, W = img.shape
                msk = np.load(msk_path) 
                msk = torch.from_numpy(msk).permute(2, 0, 1).unsqueeze(dim=0)
                msk = torch.nn.functional.interpolate(msk, size=(384,128), mode='bilinear', align_corners=True) 
               
            else:
                msk = Image.open(msk_path).convert('L')
                msk = np.array(msk, dtype=np.float)  
                msk = torch.from_numpy(msk).unsqueeze(dim=0).unsqueeze(dim=0)  
                msk = torch.nn.functional.interpolate(msk, size=(384,128), mode='bilinear', align_corners=True)  