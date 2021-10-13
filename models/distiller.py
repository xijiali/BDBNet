import torch
from torch import nn

class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()
        self.t_net = t_net # BDBNet
        self.s_net = s_net # ResNet50-RBDLF

    def forward(self, x):
        if self.training:
            tri_feat1, ce_feat1 = self.t_net(x)
            tri_feat2, ce_feat2, logits1, logits2, logits3 = self.s_net(x)
            return tri_feat1, ce_feat1, tri_feat2, ce_feat2, logits1, logits2, logits3
        else:
            tri_feat1=self.t_net(x)
            tri_feat2=self.s_net(x)
            return tri_feat1,tri_feat2 # dim=13584

class DistillerOriginal(nn.Module):
    def __init__(self, t_net, s_net):
        super(DistillerOriginal, self).__init__()
        self.t_net = t_net # BDBNet
        self.s_net = s_net # ResNet50-RBDLF

    def forward(self, x):
        if self.training:
            tri_feat1, ce_feat1 = self.t_net(x)
            tri_feat2, ce_feat2 = self.s_net(x)
            return tri_feat1, ce_feat1, tri_feat2, ce_feat2
        else:
            tri_feat1=self.t_net(x)
            tri_feat2=self.s_net(x)
            return tri_feat1,tri_feat2 # dim=13584
