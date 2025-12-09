import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch.nn as nn
import torch


class FeatLoss(nn.Module):
    def __init__(self, ):
        super(FeatLoss, self).__init__()

    def forward(self, feat1, feat2): 
        B, C = feat1.shape

        dist = torch.pow(torch.abs(feat1 - feat2), 2).sum(dim=-1)

        loss = (1. / (1. + torch.exp(-dist))).mean()

        return loss


def orthogonal_loss(clothing_feat, identity_feat):
    clothing_feat_norm = F.normalize(clothing_feat, dim=1)
    identity_feat_norm = F.normalize(identity_feat, dim=1)

    similarity_matrix = torch.matmul(clothing_feat_norm, identity_feat_norm.t())

    diagonal_similarity = torch.diagonal(similarity_matrix)

    orth_loss = -torch.mean(diagonal_similarity)

    return orth_loss


def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss


def make_loss(cfg, num_classes):  
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    ft_loss = FeatLoss()
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(feat, score, cloth_feat, score_fore, fore_feat, score_transfg,score_pose, target):#
            ID_LOSS = F.cross_entropy(score, target[: 128])
            TRI_LOSS = triplet(feat, target[: 128])[0]
            ID_LOSS2 = F.cross_entropy(score_fore, target[: 64])
            TRI_LOSS3 = triplet(fore_feat, target[: 64])[0]
            clothing_reg = torch.mean(torch.square(cloth_feat))
            ID_LOSS3 = F.cross_entropy(score_transfg, target[: 64])
            ID_LOSS4 = F.cross_entropy(score_pose, target[: 64])
            return ID_LOSS + TRI_LOSS + clothing_reg + ID_LOSS2 + TRI_LOSS3 + ID_LOSS3+ ID_LOSS4
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


