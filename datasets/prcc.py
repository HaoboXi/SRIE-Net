# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import os
import glob
import os.path as osp
import numpy as np
from .bases import BaseImageDataset
import os

class PRCC(BaseImageDataset):

    dataset_dir = 'prcc/rgb/'
    def __init__(self, root='../data/prcc/rgb/', verbose=True, pid_begin=0, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)      
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir= osp.join(self.dataset_dir, 'test')
        self.query_dir = osp.join(self.dataset_dir, 'test')
        self.mask_dir = "../data/prcc/mask_18/"
        self.msk_train_dir = osp.join(self.mask_dir, 'train')
        self._check_before_run()

        self.pid_begin = pid_begin

        train = self._process_dir(self.train_dir,self.msk_train_dir)


        gallery = self._process_dir_test_gallery(self.query_dir) 
        query = self._process_dir_test_query(self.query_dir)     
        query_B = self._process_dir_test_query_B(self.query_dir) 

        if verbose:
            print("prcc loaded")
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


    def _process_dir(self, dir_path = '', msk_dir = ''):
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs.sort() 
        pid_container = set() 
        clothes_container = set()
        for pdir in pdirs:
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            pid = int(osp.basename(pdir)) 
            pid_container.add(pid)
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0]  
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir) + osp.basename(img_dir)[0])
        pid_container = sorted(pid_container)  
        clothes_container = sorted(clothes_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id:label for label, clothes_id in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir)) 
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0] 
                label = pid2label[pid]   
                camid = cam2label[cam]   
                if cam in ['A', 'B']:
                    clothes_id = clothes2label[osp.basename(pdir)]
                else:
                    clothes_id = clothes2label[osp.basename(pdir)+osp.basename(img_dir)[0]]
                cam = osp.basename(img_dir)[0] 
                label = pid2label[pid] 
                name = img_dir.split('.')[0] + '.npy'
                name = name.split('/')[-1]
                name_01 = pdir.split('/')[-1]
                mask_path = osp.join(msk_dir,name_01,name)
                dataset.append((img_dir, label, camid, 1,mask_path))
        return dataset


    def _process_dir_test_gallery(self, test_path):
        pdirs = glob.glob(osp.join(test_path, '*'))
        pdirs.sort()
        mask_path = ''

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        gallery_dataset = []

        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:
                    camid = cam2label[cam]
                    if cam == 'A':
                        clothes_id = pid2label[pid] * 2
                        gallery_dataset.append((img_dir, pid, camid, 1,mask_path))

                    else:
                        pass
        return gallery_dataset


    def _process_dir_test_query(self, test_path):
        pdirs = glob.glob(osp.join(test_path, '*'))
        pdirs.sort()
        pid_container = set()

        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}
        mask_path = ''
        query_dataset = []
        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:
                    camid = cam2label[cam]
                    if cam == 'C':
                        clothes_id = pid2label[pid] * 2 + 1
                        query_dataset.append((img_dir, pid, camid, 1,mask_path))
                    else:
                        pass
        return query_dataset

    def _process_dir_test_query_B(self, test_path):
        pdirs = glob.glob(osp.join(test_path, '*'))
        pdirs.sort()

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        cam2label = {'A': 0, 'B': 1, 'C': 2}
        mask_path = ''
        query_dataset = []
        for cam in ['A', 'B', 'C']:
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            for pdir in pdirs:
                pid = int(osp.basename(pdir))
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in img_dirs:

                    camid = cam2label[cam]
                    if cam == 'B':
                        query_dataset.append((img_dir, pid, camid, 1,mask_path))
                    else:
                        pass
        return query_dataset