# encoding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config_original_milestone_RBDLF import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models.networks import ResNetBuilder, IDE, Resnet, BFE, RBDLF_BFE
from trainers.evaluator import ResNetEvaluatorDistiller
from trainers.trainer import cls_tripletTrainerDistiller
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin
from utils.LiftedStructure import LiftedStructureLoss
from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
from utils.lr_scheduler import WarmupMultiStepLR
from models.load_settings import load_paper_settings
from models.distiller import Distiller

def train(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset, mode=opt.mode) #'market1501','retrieval'

    pin_memory = True if use_gpu else False

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(opt.datatype)),
        sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )
    queryFliploader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryFliploader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')
    t_net, s_net = load_paper_settings(opt.paper_setting,dataset.num_train_pids,1.0, 0.33)
    # if use_gpu:
    #     t_net = nn.DataParallel(t_net).cuda()
    #     s_net = nn.DataParallel(s_net).cuda()
    d_net = Distiller(t_net, s_net)
    if use_gpu:
        d_net = nn.DataParallel(d_net).cuda()

    print('t_net model size: {:.5f}M'.format(sum(p.numel() for p in t_net.parameters()) / 1e6))
    print('s_net model size: {:.5f}M'.format(sum(p.numel() for p in s_net.parameters()) / 1e6))
    print('d_net model size: {:.5f}M'.format(sum(p.numel() for p in d_net.parameters()) / 1e6))

    reid_evaluator = ResNetEvaluatorDistiller(d_net)

    if opt.evaluate:
        reid_evaluator.evaluate(queryloader, galleryloader,
                                queryFliploader, galleryFliploader, re_ranking=opt.re_ranking, savefig=opt.savefig)
        return

    # get optimizer
    params = []
    for key, value in t_net.named_parameters():
        if not value.requires_grad:
            continue
        else:
            params += [{"params": [value], "lr": opt.lr, "weight_decay": opt.weight_decay}]

    for key, value in s_net.named_parameters():
        if not value.requires_grad:
            continue
        if 'classifier1' in key or 'classifier2' in key or 'classifier3' in key:
            print('Yes.')
            print('key is:{}'.format(key))
            params += [{"params": [value], "lr": opt.lr * 1000, "weight_decay": opt.weight_decay}]
        else:
            params += [{"params": [value], "lr": opt.lr, "weight_decay": opt.weight_decay}]
    optimizer = torch.optim.Adam(params)

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    reid_trainer = cls_tripletTrainerDistiller(opt, d_net, optimizer, dataset.num_train_pids, summary_writer)

    lr_scheduler = WarmupMultiStepLR(optimizer, [40,70], gamma=0.1, warmup_factor=0.01, warmup_iters=10)

    # start training
    best_rank1 = opt.best_rank
    best_epoch = 0
    for epoch in range(start_epoch, opt.max_epoch):
        if opt.adjust_lr:
            lr_scheduler.step()
        reid_trainer.train(epoch, trainloader)

        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            if opt.mode == 'class':
                rank1 = test(d_net, queryloader)
            else:
                rank1 = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = d_net.module.state_dict()
            else:
                state_dict = d_net.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
                            is_best=is_best, save_dir=opt.save_dir,
                            filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


def test(model, queryloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, _ in queryloader:
            output = model(data).cpu()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    rank1 = 100. * correct / len(queryloader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(queryloader.dataset), rank1))
    return rank1


if __name__ == '__main__':
    train()
