import logging
import os
import time
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    last_acc_val = best_epoch = best_mAP = best_cmc1 = 0.0

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    use_cuda = torch.cuda.is_available()
    device = "cuda"
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view,mask_path) in enumerate(train_loader):
            b,c,_,_ = img.shape
            mask_6D = mask_path.cuda() if use_cuda else mask_6D
            mask_1D = mask_6D.argmax(dim=1).unsqueeze(dim=1) 
            mask_1D = mask_1D.expand_as(img)  
            img_confused = copy.deepcopy(img)
            #

            img_fore = copy.deepcopy(img)

            index = np.random.permutation(b)
            msk_single = mask_1D[index]  
            img_single = img[index]
            img_fore[mask_1D == 4] =0
            img_confused[mask_1D == 0] = 0
            img_confused[mask_1D == 4] = img_single[msk_single == 4]
            img_cat = torch.cat([img,img_fore], dim=0)

            target_cat = torch.cat([vid, vid], dim=0) 

            target_view = torch.cat([target_view,target_view],dim=0)

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img_confused = img_confused.to(device)
            img_cat = img_cat.to(device)
            target_cat = target_cat.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            img = img.to(device)
            target = vid.to(device)
            img_fore = img_fore.to(device)
            with amp.autocast(enabled=True):
                feat, score , cloth_feat ,score_fore,fore_feat,score_transfg ,score_pose= model(img_cat, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(feat, score, cloth_feat,score_fore,fore_feat,score_transfg,score_pose,target_cat)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score.max(1)[1] == target_cat).float().mean()
            else:
                acc = (score.max(1)[1] == target_cat).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                    acc_best = 0.5 * (cmc[0] + mAP)
                    is_best = acc_best >= last_acc_val
                    if is_best:
                        last_acc_val = acc_best
                        best_epoch = epoch
                        best_mAP = mAP
                        best_cmc1 = cmc[0]
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'.format(best_epoch)))
                    logger.info("*-*-* Current Best In Epoch{} *-*-*:  mAP: {:.1%} , Rank-1: {:.1%} !!!\n"
                                .format(best_epoch, best_mAP, best_cmc1))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                acc_best = 0.5 * (cmc[0] + mAP)
                is_best = acc_best >= last_acc_val
                if is_best:
                    last_acc_val = acc_best
                    best_epoch = epoch
                    best_mAP = mAP
                    best_cmc1 = cmc[0]
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'.format(best_epoch)))
                logger.info("*-*-* Current Best :  mAP: {:.1%} , Rank-1: {:.1%} , In Epoch{} *-*-*\n"
                            .format(best_mAP, best_cmc1, best_epoch))
                torch.cuda.empty_cache()

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


