import torch
import os

from models.networks import BFE,ResNet_RBDLF_detach,ResNet_RBDLF_detach_wo_BNNeck,Resnet,ResNet_RBDLF_detach_wo_BNNeck_shared

def load_paper_settings(paper_setting,num_classes=100, width_ratio=0.5, height_ratio=0.5):
    if paper_setting == 'a':
        print('setting is:{}'.format(paper_setting))
        model1=BFE(num_classes,width_ratio,height_ratio)
        model2=ResNet_RBDLF_detach(depth=50,num_classes=num_classes)
    if paper_setting == 'b':
        print('setting is:{}'.format(paper_setting))
        model1=BFE(num_classes,width_ratio,height_ratio)
        model2=ResNet_RBDLF_detach_wo_BNNeck(num_classes=num_classes)
    if paper_setting == 'c':
        print('setting is:{}'.format(paper_setting))
        model1=BFE(num_classes,width_ratio,height_ratio)
        model2=Resnet(num_classes=num_classes)
    if paper_setting == 'd':
        print('setting is:{}'.format(paper_setting))
        model1 = BFE(num_classes, width_ratio, height_ratio)
        model2 = ResNet_RBDLF_detach_wo_BNNeck_shared(num_classes=num_classes,shared_model=model1)
    else:
        print('Undefined setting name !!!')
    return model1,model2
