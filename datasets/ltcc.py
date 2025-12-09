import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import numpy as np
import os
class Ltcc(BaseImageDataset):
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Ltcc, self).__init__()
        self.dataset_dir = "../data/ltcc/cloth_change/" 
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self.mask_dir = "../data/ltcc/mask_18"
        self.msk_train_dir = osp.join(self.mask_dir, 'train')
        self.msk_query_dir = osp.join(self.mask_dir, 'query')
        self.msk_gallery_dir = osp.join(self.mask_dir, 'test')
        self._check_before_run()

        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir, relabel=True , msk_dir=self.msk_train_dir)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)


        if verbose:
            print("LTCC_Reid loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery


        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False , msk_dir = ''):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')
        pid_container = set()
        clothes_container = set()

        for img_path in sorted(img_paths):
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            if pid == -1: continue  
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}
        pid2clothes = np.zeros((num_pids, num_clothes))

        for img_path in sorted(img_paths):
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            if pid == -1: continue  
            clothes = pattern2.search(img_path).group(1)
            clothes_id = clothes2label[clothes]
            if msk_dir != '':
                name = img_path.split('.')[0] + '.npy'
                name = name.split('/')[-1]
                mask_path = osp.join(msk_dir, name)
            else:
                pass
            camid -= 1 
            if relabel:
                pid = pid2label[pid]
            if msk_dir == '':   
                dataset.append((img_path, self.pid_begin + pid, camid, 1 ,msk_dir ))
            else:
                dataset.append((img_path, self.pid_begin + pid, camid, 1, mask_path ))
        return dataset
