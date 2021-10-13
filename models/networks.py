# encoding: utf-8
import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, resnet50
from torchvision.transforms import functional
import torchvision
from torch.nn import init

from models.resnet import ResNet

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
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x

class BatchCrop(nn.Module):
    def __init__(self, ratio):
        super(BatchCrop, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rw = int(self.ratio * w)
            start = random.randint(0, h-1)
            if start + rw > h:
                select = list(range(0, start+rw-h)) + list(range(start, h))
            else:
                select = list(range(start, start+rw))
            mask = x.new_zeros(x.size())
            mask[:, :, select, :] = 1
            x = x * mask
        return x

class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=None, last_stride=1, pretrained=False):
        super().__init__()
        self.base = ResNet(last_stride)
        if pretrained:
            model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            self.base.load_param(model_zoo.load_url(model_url))

        self.num_classes = num_classes
        if num_classes is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.in_planes, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5)
            )
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(512, self.num_classes)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.base(x)
        global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.training and self.num_classes is not None:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return [global_feat], [cls_score]
        else:
            return global_feat

    def get_optim_policy(self):
        base_param_group = self.base.parameters()
        if self.num_classes is not None:
            add_param_group = itertools.chain(self.bottleneck.parameters(), self.classifier.parameters())
            return [
                {'params': base_param_group},
                {'params': add_param_group}
            ]
        else:
            return [
                {'params': base_param_group}
            ]

class BFE(nn.Module):
    def __init__(self, num_classes, width_ratio=0.5, height_ratio=0.5):
        super(BFE, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        reduction = nn.Sequential(
            nn.Conv2d(2048, 512, 1), 
            nn.BatchNorm2d(512), 
            nn.ReLU()
        )
         # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_softmax = nn.Linear(512, num_classes) 
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)

        # part branch
        self.res_part2 = Bottleneck(2048, 512)
     
        self.part_maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.batch_crop = BatchDrop(height_ratio, width_ratio)
        self.reduction = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)

        predict = []
        triplet_features = []
        softmax_features = []

        #global branch
        glob = self.global_avgpool(x)
        global_triplet_feature = self.global_reduction(glob).squeeze()
        global_softmax_class = self.global_softmax(global_triplet_feature)
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)
       
        #part branch
        x = self.res_part2(x)

        x = self.batch_crop(x)
        triplet_feature = self.part_maxpool(x).squeeze()
        feature = self.reduction(triplet_feature)
        softmax_feature = self.softmax(feature)
        triplet_features.append(feature)
        softmax_features.append(softmax_feature)
        predict.append(feature)

        if self.training:
            return triplet_features, softmax_features
        else:
            return predict#torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.global_softmax.parameters()},
            {'params': self.res_part2.parameters()},
            {'params': self.reduction.parameters()},
            {'params': self.softmax.parameters()},
        ]
        return params

class Resnet(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(Resnet, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            #resnet.layer4
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)

        x = self.global_avgpool(x).squeeze()
        feature = self.softmax(x)
        if self.training:
            return x,feature
        else:
            return x

    def get_optim_policy(self):
        return self.parameters()

class IDE(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(IDE, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AvgPool2d(kernel_size=(12, 4))

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        feature = self.global_avgpool(x).squeeze()
        if self.training:
            return [feature], []
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()

class RBDLF_BFE(nn.Module):
    def __init__(self, num_classes, width_ratio=0.5, height_ratio=0.5):
        super(RBDLF_BFE, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        reduction = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_softmax = nn.Linear(512, num_classes)
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)
        # local branches
        self.divide_stripes = nn.AdaptiveAvgPool2d((3, 1))
        # stripe1
        self.local1_softmax = nn.Linear(2048, num_classes)
        self.local1_softmax.apply(weights_init_kaiming)
        # stripe2
        self.local2_softmax = nn.Linear(2048, num_classes)
        self.local2_softmax.apply(weights_init_kaiming)
        # stripe3
        self.local3_softmax = nn.Linear(2048, num_classes)
        self.local3_softmax.apply(weights_init_kaiming)


        # part branch
        self.res_part2 = Bottleneck(2048, 512)

        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_crop = BatchDrop(height_ratio, width_ratio)
        self.reduction = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)


    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)
        # Feature maps: F
        spatial_f = x.detach()
        spatial_f = self.divide_stripes(spatial_f)  # [bs,2048,3,1]
        spatial_f = spatial_f.squeeze(-1)  # [bs,2048,3]
        # local logits
        logits1 = self.local1_softmax(spatial_f[:, :, 0])
        logits2 = self.local2_softmax(spatial_f[:, :, 1])
        logits3 = self.local3_softmax(spatial_f[:, :, 2])
        # local weights
        w1 = nn.Sigmoid()(logits1)
        w2 = nn.Sigmoid()(logits2)
        w3 = nn.Sigmoid()(logits3)

        predict = []
        triplet_features = []
        softmax_features = []

        # global branch
        glob = self.global_avgpool(x)
        global_triplet_feature = self.global_reduction(glob).squeeze()
        global_softmax_class = self.global_softmax(global_triplet_feature)
        global_softmax_class= global_softmax_class * w1 * w2 * w3
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)

        # part branch
        x = self.res_part2(x)

        x = self.batch_crop(x)
        triplet_feature = self.part_maxpool(x).squeeze()
        feature = self.reduction(triplet_feature)
        softmax_feature = self.softmax(feature)
        triplet_features.append(feature)
        softmax_features.append(softmax_feature)
        predict.append(feature)

        if self.training:
            return triplet_features, softmax_features, logits1, logits2, logits3
        else:
            return predict#torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.global_softmax.parameters()},
            {'params': self.res_part2.parameters()},
            {'params': self.reduction.parameters()},
            {'params': self.softmax.parameters()},
        ]
        return params

class RBDLF_BFE_wo_Embedding(nn.Module):
    def __init__(self, num_classes, width_ratio=0.5, height_ratio=0.5):
        super(RBDLF_BFE_wo_Embedding, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        # reduction = nn.Sequential(
        #     nn.Conv2d(2048, 512, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )
        # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_softmax = nn.Linear(2048, num_classes)
        self.global_softmax.apply(weights_init_kaiming)
        #self.global_reduction = copy.deepcopy(reduction)
        #self.global_reduction.apply(weights_init_kaiming)
        # local branches
        self.divide_stripes = nn.AdaptiveAvgPool2d((3, 1))
        # stripe1
        self.local1_softmax = nn.Linear(2048, num_classes)
        self.local1_softmax.apply(weights_init_kaiming)
        # stripe2
        self.local2_softmax = nn.Linear(2048, num_classes)
        self.local2_softmax.apply(weights_init_kaiming)
        # stripe3
        self.local3_softmax = nn.Linear(2048, num_classes)
        self.local3_softmax.apply(weights_init_kaiming)


        # part branch
        self.res_part2 = Bottleneck(2048, 512)

        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_crop = BatchDrop(height_ratio, width_ratio)
        self.reduction = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)


    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)
        # Feature maps: F
        spatial_f = x.detach()
        spatial_f = self.divide_stripes(spatial_f)  # [bs,2048,3,1]
        spatial_f = spatial_f.squeeze(-1)  # [bs,2048,3]
        # local logits
        logits1 = self.local1_softmax(spatial_f[:, :, 0])
        logits2 = self.local2_softmax(spatial_f[:, :, 1])
        logits3 = self.local3_softmax(spatial_f[:, :, 2])
        # local weights
        w1 = nn.Sigmoid()(logits1)
        w2 = nn.Sigmoid()(logits2)
        w3 = nn.Sigmoid()(logits3)

        predict = []
        triplet_features = []
        softmax_features = []

        # global branch
        glob = self.global_avgpool(x)
        #global_triplet_feature = self.global_reduction(glob).squeeze()
        global_triplet_feature=glob.squeeze()
        global_softmax_class = self.global_softmax(global_triplet_feature)
        global_softmax_class= global_softmax_class * w1 * w2 * w3
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)

        # part branch
        x = self.res_part2(x)

        x = self.batch_crop(x)
        triplet_feature = self.part_maxpool(x).squeeze()
        feature = self.reduction(triplet_feature)
        softmax_feature = self.softmax(feature)
        triplet_features.append(feature)
        softmax_features.append(softmax_feature)
        predict.append(feature)

        if self.training:
            return triplet_features, softmax_features, logits1, logits2, logits3
        else:
            return predict#torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            #{'params': self.global_reduction.parameters()},
            {'params': self.global_softmax.parameters()},
            {'params': self.res_part2.parameters()},
            {'params': self.reduction.parameters()},
            {'params': self.softmax.parameters()},
        ]
        return params

class ResNet_RBDLF_detach_wo_BNNeck(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(ResNet_RBDLF_detach_wo_BNNeck, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            #resnet.layer4
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(2048, num_classes)

        # local branches
        self.divide_stripes = nn.AdaptiveAvgPool2d((3, 1))
        # stripe1
        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier1.apply(weights_init_kaiming)
        # stripe2
        self.classifier2 = nn.Linear(2048, num_classes)
        self.classifier2.apply(weights_init_kaiming)
        # stripe3
        self.classifier3 = nn.Linear(2048, num_classes)
        self.classifier3.apply(weights_init_kaiming)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)
        # Feature maps: F
        spatial_f = x.detach()
        spatial_f = self.divide_stripes(spatial_f)  # [bs,2048,3,1]
        spatial_f = spatial_f.squeeze(-1)  # [bs,2048,3]
        # local logits
        logits1 = self.classifier1(spatial_f[:, :, 0])
        logits2 = self.classifier2(spatial_f[:, :, 1])
        logits3 = self.classifier3(spatial_f[:, :, 2])
        # local weights
        w1 = nn.Sigmoid()(logits1)
        w2 = nn.Sigmoid()(logits2)
        w3 = nn.Sigmoid()(logits3)
        x = self.global_avgpool(x).squeeze()
        feature = self.softmax(x)
        feature=feature* w1 * w2 * w3
        if self.training:
            return x,feature,logits1, logits2, logits3
        else:
            return x

    def get_optim_policy(self):
        return self.parameters()

class ResNet_RBDLF_detach_wo_BNNeck_shared(nn.Module):
    def __init__(self, num_classes,shared_model, resnet=None):
        super(ResNet_RBDLF_detach_wo_BNNeck_shared, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            shared_model.backbone[0],
            shared_model.backbone[1],
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            #resnet.layer4
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(2048, num_classes)

        # local branches
        self.divide_stripes = nn.AdaptiveAvgPool2d((3, 1))
        # stripe1
        self.classifier1 = nn.Linear(2048, num_classes)
        self.classifier1.apply(weights_init_kaiming)
        # stripe2
        self.classifier2 = nn.Linear(2048, num_classes)
        self.classifier2.apply(weights_init_kaiming)
        # stripe3
        self.classifier3 = nn.Linear(2048, num_classes)
        self.classifier3.apply(weights_init_kaiming)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)
        # Feature maps: F
        spatial_f = x.detach()
        spatial_f = self.divide_stripes(spatial_f)  # [bs,2048,3,1]
        spatial_f = spatial_f.squeeze(-1)  # [bs,2048,3]
        # local logits
        logits1 = self.classifier1(spatial_f[:, :, 0])
        logits2 = self.classifier2(spatial_f[:, :, 1])
        logits3 = self.classifier3(spatial_f[:, :, 2])
        # local weights
        w1 = nn.Sigmoid()(logits1)
        w2 = nn.Sigmoid()(logits2)
        w3 = nn.Sigmoid()(logits3)
        x = self.global_avgpool(x).squeeze()
        feature = self.softmax(x)
        feature=feature* w1 * w2 * w3
        if self.training:
            return x,feature,logits1, logits2, logits3
        else:
            return x

    def get_optim_policy(self):
        return self.parameters()

class ResNet_RBDLF_detach(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet_RBDLF_detach, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling # False
        # Construct base (pretrained) resnet
        if depth not in ResNet_RBDLF_detach.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet_RBDLF_detach.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.divide_stripes = nn.AdaptiveAvgPool2d((3, 1))

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0 # False
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
                # stripe1
                self.classifier1 = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier1.weight, std=0.001)
                # stripe2
                self.classifier2 = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier2.weight, std=0.001)
                # stripe3
                self.classifier3 = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier3.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, current_epoch=0,feature_withbn=False,start_epochs=0):
        x = self.base(x)
        # Feature maps: F
        spatial_f = x.detach()
        spatial_f = self.divide_stripes(spatial_f) # [bs,2048,3,1]
        spatial_f = spatial_f.squeeze(-1) # [bs,2048,3]
        # local logits
        logits1 = self.classifier1(spatial_f[:, :, 0])
        logits2 = self.classifier2(spatial_f[:, :, 1])
        logits3 = self.classifier3(spatial_f[:, :, 2])
        # local weights
        w1 = nn.Sigmoid()(logits1)
        w2 = nn.Sigmoid()(logits2)
        w3 = nn.Sigmoid()(logits3)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling: # False
            return x

        if self.has_embedding: # False
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            spatial_f1 = F.normalize(spatial_f[:, :, 0])
            spatial_f2 = F.normalize(spatial_f[:, :, 1])
            spatial_f3 = F.normalize(spatial_f[:, :, 2])
            return bn_x#, spatial_f1, spatial_f2, spatial_f3

        if self.norm: # False
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0: # False
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn: # False
            return bn_x, prob

        # HPG
        if start_epochs > 0:
            if current_epoch >= start_epochs:
                prob = prob * w1 * w2 * w3
        else:
            prob = prob * w1 * w2 * w3

        return x, prob, logits1, logits2, logits3

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())

# class Resnet(nn.Module):
#     __factory = {
#         18: torchvision.models.resnet18,
#         34: torchvision.models.resnet34,
#         50: torchvision.models.resnet50,
#         101: torchvision.models.resnet101,
#         152: torchvision.models.resnet152,
#     }
#
#     def __init__(self, depth, pretrained=True, cut_at_pooling=False,
#                  num_features=0, norm=False, dropout=0, num_classes=0):
#         super(Resnet, self).__init__()
#         self.pretrained = pretrained
#         self.depth = depth
#         self.cut_at_pooling = cut_at_pooling # False
#         # Construct base (pretrained) resnet
#         if depth not in Resnet.__factory:
#             raise KeyError("Unsupported depth:", depth)
#         resnet = Resnet.__factory[depth](pretrained=pretrained)
#         resnet.layer4[0].conv2.stride = (1,1)
#         resnet.layer4[0].downsample[0].stride = (1,1)
#         self.base = nn.Sequential(
#             resnet.conv1, resnet.bn1, resnet.maxpool, # no relu
#             resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#
#         if not self.cut_at_pooling:
#             self.num_features = num_features
#             self.norm = norm
#             self.dropout = dropout
#             self.has_embedding = num_features > 0 # False
#             self.num_classes = num_classes
#
#             out_planes = resnet.fc.in_features
#
#             # Append new layers
#             if self.has_embedding:
#                 self.feat = nn.Linear(out_planes, self.num_features)
#                 self.feat_bn = nn.BatchNorm1d(self.num_features)
#                 init.kaiming_normal_(self.feat.weight, mode='fan_out')
#                 init.constant_(self.feat.bias, 0)
#             else:
#                 # Change the num_features to CNN output channels
#                 self.num_features = out_planes
#                 self.feat_bn = nn.BatchNorm1d(self.num_features)
#             self.feat_bn.bias.requires_grad_(False)
#             if self.dropout > 0:
#                 self.drop = nn.Dropout(self.dropout)
#             if self.num_classes > 0:
#                 self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
#                 init.normal_(self.classifier.weight, std=0.001)
#         init.constant_(self.feat_bn.weight, 1)
#         init.constant_(self.feat_bn.bias, 0)
#
#         if not pretrained:
#             self.reset_params()
#
#     def forward(self, x, feature_withbn=False):
#         x = self.base(x)
#
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#
#         if self.cut_at_pooling: # False
#             return x
#
#         if self.has_embedding: # False
#             bn_x = self.feat_bn(self.feat(x))
#         else:
#             bn_x = self.feat_bn(x)
#
#         if self.training is False:
#             bn_x = F.normalize(bn_x)
#             return bn_x
#
#         if self.norm: # False
#             bn_x = F.normalize(bn_x)
#         elif self.has_embedding:
#             bn_x = F.relu(bn_x)
#
#         if self.dropout > 0: # False
#             bn_x = self.drop(bn_x)
#
#         if self.num_classes > 0:
#             prob = self.classifier(bn_x)
#         else:
#             return x, bn_x
#
#         if feature_withbn: # False
#             return bn_x, prob
#         return x, prob
#
#     def reset_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#         resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
#         self.base[0].load_state_dict(resnet.conv1.state_dict())
#         self.base[1].load_state_dict(resnet.bn1.state_dict())
#         self.base[2].load_state_dict(resnet.maxpool.state_dict())
#         self.base[3].load_state_dict(resnet.layer1.state_dict())
#         self.base[4].load_state_dict(resnet.layer2.state_dict())
#         self.base[5].load_state_dict(resnet.layer3.state_dict())
#         self.base[6].load_state_dict(resnet.layer4.state_dict())

if __name__ == "__main__":
    net = BFE(num_classes=751)
    print(net)
    print('net size: {:.5f}M'.format(sum(p.numel() for p in net.parameters()) / 1e6)) # 25.04683M
    for k,v in net.named_parameters():
        print(k)
    # import torch
    #
    # x = net(torch.zeros(1, 3, 256, 128))
    # print(x.shape)
