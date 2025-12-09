# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import os.path as osp
import numpy as np
from .bases import BaseImageDataset

class NKUP(BaseImageDataset):
    dataset_dir = 'nkup'
    msk_dir = 'nkup_mask18'
    def __init__(self, root='../data/', verbose=True, pid_begin=0, **kwargs):
        super(NKUP, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.mask_dir = osp.join(root, self.msk_dir)
        self.msk_train_dir = osp.join(self.mask_dir,'bounding_box_train')
        self._check_before_run()

        self.pid_begin = pid_begin
        self.pid2label = self.get_pid2label(self.train_dir)

        train = self._process_dir(self.train_dir, relabel=True, pid2label=self.pid2label, msk_dir=self.msk_train_dir)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print('=> NKUP loaded')
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
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def get_pid2label(self, dir_path):
        images = os.listdir(dir_path)
        persons = [int(img.split('_')[0]) for img in images]
        pid_container = np.sort(list(set(persons)))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def _process_dir(self, dir_path, pid2label=None, relabel=False, msk_dir=''):
        images = os.listdir(dir_path)
        dataset = []
        for img in images:
            pid_s = int(img.split('_')[0])
            cid = int(img.split('_')[2][1:3])
            if relabel and pid2label is not None:
                pid = pid2label[pid_s]
            else:
                pid = int(pid_s)
            img_path = os.path.join(dir_path, img)
            if pid == -1: continue  
            if msk_dir != '':
                name = img_path.split('.')[0] + '.npy'
                name = name.split('/')[-1]
                mask_path = osp.join(msk_dir, name)
            else:
                pass
            cid -= 1
            if msk_dir == '': 
                if not os.path.exists(img_path):
                    continue
                else:
                    dataset.append((img_path, self.pid_begin + pid, cid, 1, msk_dir))
            else:
                dataset.append((img_path, self.pid_begin + pid, cid, 1, mask_path))
        return dataset
