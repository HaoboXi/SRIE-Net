import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class RelationModel(nn.Module):
    def __init__(
            self,
            last_conv_stride=1,
            last_conv_dilation=1,
            num_stripes=13,
            local_conv_out_channels=768,
            num_classes=0):
        super(RelationModel, self).__init__()
        self.num_stripes = num_stripes
        self.num_classes = num_classes

        self.local_6_conv_list = nn.ModuleList()
        self.rest_6_conv_list = nn.ModuleList()
        self.relation_6_conv_list = nn.ModuleList()
        self.global_6_max_conv_list = nn.ModuleList()
        self.global_6_rest_conv_list = nn.ModuleList()
        self.global_6_pooling_conv_list = nn.ModuleList()

        for i in range(num_stripes):
            self.local_6_conv_list.append(nn.Sequential(
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))
        for i in range(num_stripes):
            self.rest_6_conv_list.append(nn.Sequential(
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))


        self.global_6_max_conv_list.append(nn.Sequential(
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))
        self.global_6_rest_conv_list.append(nn.Sequential(
        
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        for i in range(num_stripes):
            self.relation_6_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))


        self.global_6_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

    def forward(self, local_feat):
        local_6_feat_list = []
        final_feat_list = []
        rest_6_feat_list = []
        for i in range(self.num_stripes):
            local_6_feat = local_feat[:,i:i+1].squeeze(1).unsqueeze(-1).unsqueeze(-1)

            local_6_feat_list.append(local_6_feat)

        for i in range(self.num_stripes):
            rest_6_feat_list.append((local_6_feat_list[(i + 1) % self.num_stripes]
                                     + local_6_feat_list[(i + 2) % self.num_stripes]
                                     + local_6_feat_list[(i + 3) % self.num_stripes]
                                     + local_6_feat_list[(i + 4) % self.num_stripes]
                                     + local_6_feat_list[(i + 5) % self.num_stripes]
                                     + local_6_feat_list[(i + 6) % self.num_stripes]
                                     + local_6_feat_list[(i + 7) % self.num_stripes]
                                     + local_6_feat_list[(i + 8) % self.num_stripes]
                                     + local_6_feat_list[(i + 9) % self.num_stripes]
                                     + local_6_feat_list[(i + 10) % self.num_stripes]
                                     + local_6_feat_list[(i + 11) % self.num_stripes]
                                     + local_6_feat_list[(i + 12) % self.num_stripes]
                                     ) / 12)

        for i in range(self.num_stripes):
            local_6_feat = self.local_6_conv_list[i](local_6_feat_list[i]).squeeze(3).squeeze(2)
            input_rest_6_feat = self.rest_6_conv_list[i](rest_6_feat_list[i]).squeeze(3).squeeze(2)

            input_local_rest_6_feat = torch.cat((local_6_feat, input_rest_6_feat), 1).unsqueeze(2).unsqueeze(3)

            local_rest_6_feat = self.relation_6_conv_list[i](input_local_rest_6_feat)

            local_rest_6_feat = (local_rest_6_feat
                                 + local_6_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

            final_feat_list.append(local_rest_6_feat)
        return final_feat_list