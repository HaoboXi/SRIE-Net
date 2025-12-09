import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import random

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .ltcc import Ltcc
from .prcc import PRCC
from .nkup import NKUP
from .celeb import Celeb
from .celeb_light import Celeb_light
from .vcclothes import VCClothes
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'ltcc': Ltcc,
    'prcc': PRCC,
    'nkup':NKUP,
    'celeb':Celeb,
    'celeb_light':Celeb_light,
    'vcclothes':VCClothes
}

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)
    random.seed(torch.initial_seed() % 2**32 + worker_id)
    
def train_collate_fn(batch):
    imgs, pids, camids, viewids , _ , mask_path = zip(*batch) # 现在imgs已经是Tensor了，不是一个地址啊
    # print(imgs)
    # print("imgs.shape = ",imgs.shape)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    # mask_path = torch.tensor(mask_path,torch.int64)
    # print(torch.cat(mask_path).shape)
    return torch.stack(imgs, dim=0), pids, camids, viewids, torch.cat(mask_path)

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths ,clos= zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids,img_paths

# 现在要实现的是我们要先将图片进行打码并且转换之后，在进行normalize
def make_dataloader(cfg):
    # 使用与全局一致的随机种子，保证每次运行可复现
    seed = cfg.SOLVER.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 为 DataLoader 创建确定性的随机数生成器
    g = torch.Generator()
    g.manual_seed(seed)

    print(f"[make_dataloader] seed={seed}，随机增强开启但在相同 seed 下可复现")

    # 保留随机数据增强；在固定 seed 下其行为是确定的
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    # return img, pid, camid, trackid, img_path.split('/')[-1] , msk
    train_set = ImageDataset(dataset.train, train_transforms,mask = True)
    train_set_normal = ImageDataset(dataset.train, val_transforms , mask= True)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    # clos_num = dataset.

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
                generator=g,
                worker_init_fn=worker_init_fn if num_workers > 1 else None,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
                generator=g,
                worker_init_fn=worker_init_fn if num_workers > 1 else None,
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler (deterministic with fixed seed)')
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            generator=g,
            worker_init_fn=worker_init_fn if num_workers > 1 else None,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
