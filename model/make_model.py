import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, \
    deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import torchvision.transforms as transforms
from .backbones.hrnet import HRNet
from .pose_net import SimpleHRNet
import cv2
from .backbones.model_gcn import GraphConvNet, generate_adj
import torch.nn.functional as F
from .backbones.SSE import CAMC
from .backbones.TransFG import VisionTransformer,CONFIGS
import numpy as np
from .backbones.rm import RelationModel
def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def orthogonal_complement_projection(global_feature, cloth_feature):
    P = torch.eye(global_feature.size(1), dtype=global_feature.dtype, device=global_feature.device) - torch.mm(
        cloth_feature.t(), cloth_feature) / torch.dot(cloth_feature.flatten(), cloth_feature.flatten())
    orthogonal_complement = torch.mm(global_feature, P.t())
    return orthogonal_complement 


class SalientGuidedAttentionModule(nn.Module):
    def __init__(self, input_channels, threshold):
        super(SalientGuidedAttentionModule, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold
        self.gmp = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        saliency_map = self.softmax(x)
        saliency_mask = torch.where(saliency_map > self.threshold, saliency_map, torch.zeros_like(saliency_map))

        attention_weights = self.sigmoid(saliency_mask)


        return attention_weights


class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.local_fc = nn.Linear(input_dim, input_dim)
        self.global_fc = nn.Linear(input_dim, input_dim)
        self.att_fc = nn.Linear(input_dim, 1)

    def forward(self,head, right_arm,left_arm,right_leg,left_leg, global_feat):
        head_weight = F.softmax(self.local_fc(head), dim=1)
        right_arm_weight = F.softmax(self.local_fc(right_arm), dim=1)
        left_arm_weight = F.softmax(self.local_fc(left_arm), dim=1)
        right_leg_weight = F.softmax(self.local_fc(right_leg), dim=1)
        left_leg_weight = F.softmax(self.local_fc(left_leg), dim=1)
        global_weight = F.softmax(self.global_fc(global_feat), dim=1)

        fusion_feat =head_weight*head + right_arm_weight * right_arm + left_arm_weight * left_arm + right_leg_weight * right_leg + left_leg_weight * left_leg + global_weight * global_feat  # 融合局部特征和全局特征

        att_weight = F.softmax(self.att_fc(fusion_feat), dim=1)  
        att_feat = att_weight * fusion_feat  
        return att_feat


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim):
        super(Attention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
       
    def forward(self, query, keys):

        
        query1=0.4*query
        keys1=0.6*keys
        attn_weights = torch.matmul(query1, keys1.transpose(-2, -1))  
        attn_weights = torch.softmax(attn_weights, dim=-1)  
        fused_feature = torch.matmul(attn_weights, keys1)+keys 
        return fused_feature


class AttentionAggregationModule(nn.Module):
    def __init__(self, input_size, num_parts):
        super(AttentionAggregationModule, self).__init__()
        self.input_size = input_size
        self.num_parts = num_parts
        self.attention_layer = nn.Linear(input_size, 1)  

    def forward(self, local_features):
        batch_size = local_features.size(0)

      
        attention_scores = self.attention_layer(local_features).squeeze(-1)  
        attention_weights = torch.softmax(attention_scores, dim=1)  
        weighted_features = local_features * attention_weights.unsqueeze(-1) 
        return weighted_features
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.in_planes1 = 128
        self.in_planes2 = 77#150
        self.org = orthogonal_complement_projection
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)

        self.base_01 = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                           camera=camera_num, view=view_num,
                                                           stride_size=cfg.MODEL.STRIDE_SIZE,
                                                           drop_path_rate=cfg.MODEL.DROP_PATH,
                                                           drop_rate=cfg.MODEL.DROP_OUT,
                                                           attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        config = CONFIGS['ViT-B_16']
        config.split = 'non-overlap'
        config.slide_step = 12
        self.transfg = VisionTransformer(config, img_size=cfg.INPUT.SIZE_TRAIN, zero_head=True, num_classes=num_classes,
                                  smoothing_value=0.0)

    
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            self.base_01.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.conv2 = nn.Conv2d(17, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(2048, 17, kernel_size=1, bias=False)
        self.pm_fc = nn.Linear(768 * 6, 128).cuda()
        self.agg = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(768, 8, 512, 0.1, 'relu'),
            num_layers=3
        )
        self.agg.apply(weights_init_kaiming)
        self.groups = [[0, 1, 2, 3, 4], [5, 7, 9], [6, 8, 10], [5,6,11,12], [11, 13, 15], [12, 14, 16]]  # [0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16][0,1,2,3,4],[5,7,9],[11,13,15],[6,8,10],[12,14,16],[5,6,7,8,9,10,11,12],[11,12,13,14,15,16]
        self.pose = SimpleHRNet(32,
                                17,
                                '../../pose_hrnet_w32_256x192.pth',
                                model_name='HRNet',
                                resolution=(384, 128),
                                interpolation=cv2.INTER_CUBIC,
                                multiperson=False,
                                return_heatmaps=True,
                                return_bounding_boxes=False,
                                max_batch_size=8,
                                device=torch.device("cuda")
                                )
        self.fc = nn.Linear(1536, 150, bias=False)
        self.attention=AttentionFusion(768)
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_01 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_01.apply(weights_init_classifier)
            self.classifier_02 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_02.apply(weights_init_classifier)
            self.classifier_03 = nn.Linear(self.in_planes1, self.num_classes, bias=False)
            self.classifier_03.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_01 = nn.BatchNorm1d(self.in_planes)#
        self.bottleneck_01.bias.requires_grad_(False)
        self.bottleneck_01.apply(weights_init_kaiming)

        self.bottleneck_02 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_02.bias.requires_grad_(False)
        self.bottleneck_02.apply(weights_init_kaiming)

        self.bottleneck_03 = nn.BatchNorm1d(self.in_planes1)
        self.bottleneck_03.bias.requires_grad_(False)
        self.bottleneck_03.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)  
        feat = self.bottleneck(global_feat) 

        if self.training:
            B = global_feat.shape[0]
            bs = B // 2

            heatmaps, joints = self.pose.predict(x[bs:])  
            heatmaps = torch.from_numpy(heatmaps).cuda()  
            features =  global_feat[bs:].unsqueeze(1)
            heatmaps = heatmaps.view(bs, heatmaps.shape[1], -1) 
            heatmap = torch.sum(heatmaps[:, self.groups[0]], dim=1, keepdim=True)
            self.pose_avg = nn.AdaptiveAvgPool2d((1, 768))
            heatmap = self.pose_avg(heatmap)  
            local_feat = heatmap * features
           
            for i in range(1, len(self.groups)):
                heatmapi = torch.sum(heatmaps[:, self.groups[i]], dim=1, keepdim=True)
                heatmapi = self.pose_avg(heatmapi)
                heatmap = torch.cat((heatmap, heatmapi), dim=1)  
                local_feat_i = heatmapi * features 
                local_feat = torch.cat((local_feat, local_feat_i), dim=1) 
            aggregation_module = AttentionAggregationModule(input_size=768, num_parts=6)
            aggregation_module.cuda()
            aggregated_feature = aggregation_module(local_feat)
            head = aggregated_feature[:, :1, :].squeeze(1) 
            left_arm = aggregated_feature[:, 1:2, :].squeeze(1) 
            right_arm = aggregated_feature[:, 2:3, :].squeeze(1)
            body = aggregated_feature[:, 3:4, :].squeeze(1)
            left_leg = aggregated_feature[:, 4:5, :].squeeze(1)  
            right_leg = aggregated_feature[:, 5:6, :].squeeze(1)
            feat_list = []
            feat_list.append(head)
            feat_list.append(left_arm)
            feat_list.append(right_arm)
            feat_list.append(body)
            feat_list.append(left_leg)
            feat_list.append(right_leg)
            feats = torch.stack(feat_list, dim=0)
            four = self.agg(feats)
            part_feat = four.permute(1,0,2)
            four = part_feat.reshape(part_feat.shape[0], -1)
       
            pose_feat = self.pm_fc(four)
   
            head_bn = self.bottleneck_03(pose_feat )
            cls_head = self.classifier_03(head_bn )

            transfg_feat=self.transfg(x[64:])
            attention = Attention(query_dim=768, key_dim=768)
            fused_features = attention(transfg_feat ,global_feat[:64])
            fused_bn = self.bottleneck_01(fused_features)
            cloth_feat = self.org(global_feat[:64], global_feat[64:])

            cls_score = self.classifier(feat)
            cls_transfg=self.classifier_02(transfg_feat)
            cls_fused = self.classifier_01(fused_bn)
            return global_feat, cls_score, cloth_feat, cls_fused, transfg_feat,cls_transfg,cls_head
        else:

            if self.neck_feat == 'after':
      
                return feat
            else:
                transfg_feat = self.transfg(x)
     
                attention = Attention(query_dim=768, key_dim=768)
                fused_features = attention(transfg_feat, global_feat)
                return fused_features

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE,
                                                     cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                    cls_score_4
                    ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                        local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type,
                                            rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
