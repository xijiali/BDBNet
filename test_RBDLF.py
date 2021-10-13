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

from config_RBDLF_test import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models.networks import ResNetBuilder, IDE, Resnet, BFE, RBDLF_BFE
from trainers.evaluator import ResNetEvaluatorRBDLF
from trainers.trainer import cls_tripletTrainer_RBDLF
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin
from utils.LiftedStructure import LiftedStructureLoss
from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
from utils.lr_scheduler import WarmupMultiStepLR


def train(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_test.txt'))

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
    # ResNet50 (CE+triplet)
    if opt.model_name == 'softmax' or opt.model_name == 'softmax_triplet':
        model = ResNetBuilder(dataset.num_train_pids, 1, True)
    # ResNet50 (triplet)
    elif opt.model_name == 'triplet':
        model = ResNetBuilder(None, 1, True)
    # BDBNet
    elif opt.model_name == 'bfe':
        if opt.datatype == "person":
            model = BFE(dataset.num_train_pids, 1.0, 0.33) # Here
        else:
            model = BFE(dataset.num_train_pids, 0.5, 0.5)
    # IDE
    elif opt.model_name == 'ide':
        model = IDE(dataset.num_train_pids)
    # ResNet50(CE)
    elif opt.model_name == 'resnet':
        model = Resnet(dataset.num_train_pids)
    # RBDLF-BDB
    elif opt.model_name == 'RBDLF':
        model = RBDLF_BFE(dataset.num_train_pids, 1.0, 0.33)

    optim_policy = model.get_optim_policy()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = ResNetEvaluatorRBDLF(model)

    if opt.evaluate:
        reid_evaluator.evaluate(queryloader, galleryloader,
                                queryFliploader, galleryFliploader, re_ranking=opt.re_ranking, savefig=opt.savefig)
        return

    # xent_criterion = nn.CrossEntropyLoss()
    xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_pids)

    if opt.loss == 'triplet':
        embedding_criterion = TripletLoss(opt.margin)
    elif opt.loss == 'lifted':
        embedding_criterion = LiftedStructureLoss(hard_mining=True)
    elif opt.loss == 'weight':
        embedding_criterion = Margin()

    def criterion(triplet_y, softmax_y, labels):
        losses = [embedding_criterion(output, labels)[0] for output in triplet_y] + \
                 [xent_criterion(output, labels) for output in softmax_y]
        loss = sum(losses)
        return loss

    # get optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            if 'local1_softmax' in key or 'local2_softmax' in key or 'local3_softmax' in key:
                print('Yes.')
                params += [{"params": [value], "lr": opt.lr * 1000, "weight_decay": opt.weight_decay}]
            else:
                params += [{"params": [value], "lr": opt.lr, "weight_decay": opt.weight_decay}]
        optimizer = torch.optim.Adam(params)
        #optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=opt.weight_decay)

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    reid_trainer = cls_tripletTrainer_RBDLF(opt, model, optimizer, dataset.num_train_pids, summary_writer)

    def adjust_lr(optimizer, ep):
        # [0,50):lr from 1e-4 to 1e-3 linearly
        # [50,200):lr=1e-3
        # [200,300):lr=1e-4
        # [300,500):lr=1e-5
        if ep < 50:
            lr = 1e-4 * (ep // 5 + 1)
        elif ep < 200:
            lr = 1e-3
        elif ep < 300:
            lr = 1e-4
        else:
            lr = 1e-5
        for p in optimizer.param_groups:
            p['lr'] = lr

    lr_scheduler = WarmupMultiStepLR(optimizer, [200,300], gamma=0.1, warmup_factor=0.1, warmup_iters=50)

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
                rank1 = test(model, queryloader)
            else:
                rank1 = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
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
