# encoding: utf-8
import math
import time
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.loss import euclidean_dist, hard_example_mining
from utils.meters import AverageMeter
from utils.loss import CrossEntropyLabelSmooth, SoftTripletLoss

class cls_tripletTrainer:
    def __init__(self, opt, model, optimzier, criterion, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer= optimzier
        self.criterion = criterion
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            self._parse_data(inputs)
            self._forward()
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('loss', self.loss.item(), global_step) # iterations
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))
        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        if self.opt.random_crop and random.random() > 0.3:
            h, w = imgs.size()[-2:]
            start = int((h-2*w)*random.random())
            mask = imgs.new_zeros(imgs.size())
            mask[:, :, start:start+2*w, :] = 1
            imgs = imgs * mask
        '''
        if random.random() > 0.5:
            h, w = imgs.size()[-2:]
            for attempt in range(100):
                area = h * w
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 3.33)
                ch = int(round(math.sqrt(target_area * aspect_ratio)))
                cw = int(round(math.sqrt(target_area / aspect_ratio)))
                if cw <  w and ch < h:
                    x1 = random.randint(0, h - ch)
                    y1 = random.randint(0, w - cw)
                    imgs[:, :, x1:x1+h, y1:y1+w] = 0
                    break
        '''
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, feat = self.model(self.data)
        self.loss = self.criterion(score, feat, self.target)

    def _backward(self):
        self.loss.backward()

class cls_tripletTrainer_wo_BFE:
    def __init__(self, opt, model, optimzier, criterion, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer= optimzier
        self.criterion = criterion
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        start = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            self._parse_data(inputs)
            self._forward()
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('loss', self.loss.item(), global_step) # iterations
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean))
        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        if self.opt.random_crop and random.random() > 0.3:
            h, w = imgs.size()[-2:]
            start = int((h-2*w)*random.random())
            mask = imgs.new_zeros(imgs.size())
            mask[:, :, start:start+2*w, :] = 1
            imgs = imgs * mask
        '''
        if random.random() > 0.5:
            h, w = imgs.size()[-2:]
            for attempt in range(100):
                area = h * w
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 3.33)
                ch = int(round(math.sqrt(target_area * aspect_ratio)))
                cw = int(round(math.sqrt(target_area / aspect_ratio)))
                if cw <  w and ch < h:
                    x1 = random.randint(0, h - ch)
                    y1 = random.randint(0, w - cw)
                    imgs[:, :, x1:x1+h, y1:y1+w] = 0
                    break
        '''
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, feat = self.model(self.data)
        #print('score[0] size is:{}'.format(score[0].size()))#[128, 512]
        #print('feat[0] size is :{}'.format(feat[0].size()))#[128, 751]
        self.loss = self.criterion([score[0]], [feat[0]], self.target)

    def _backward(self):
        self.loss.backward()

class cls_tripletTrainer_RBDLF:
    def __init__(self, opt, model, optimzier, num_classes, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer= optimzier
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss().cuda()
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr= AverageMeter()
        losses_ce_bfe= AverageMeter()
        losses_tr_bfe= AverageMeter()
        losses_ce_logits1= AverageMeter()
        losses_ce_logits2= AverageMeter()
        losses_ce_logits3= AverageMeter()


        start = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            self._parse_data(inputs)
            loss_ce,loss_tr,loss_ce_bfe,loss_tr_bfe,loss_ce_logits1,loss_ce_logits2,loss_ce_logits3=self._forward()
            self.loss = loss_ce + loss_tr + loss_ce_bfe + loss_tr_bfe + loss_ce_logits1 + loss_ce_logits2 + loss_ce_logits3
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_ce_bfe.update(loss_ce_bfe.item())
            losses_tr_bfe.update(loss_tr_bfe.item())
            losses_ce_logits1.update(loss_ce_logits1.item())
            losses_ce_logits2.update(loss_ce_logits2.item())
            losses_ce_logits3.update(loss_ce_logits3.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('all_losses', self.loss.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce', loss_ce.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_tr', loss_tr.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_bfe', loss_ce_bfe.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_tr_bfe', loss_tr_bfe.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits1', loss_ce_logits1.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits2', loss_ce_logits2.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits3', loss_ce_logits3.item(), global_step) # iterations
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'All_Loss {:.3f} ({:.3f})\t'
                      'loss_ce {:.3f} ({:.3f})\t'
                      'loss_tr {:.3f} ({:.3f})\t'
                      'loss_ce_bfe {:.3f} ({:.3f})\t'
                      'loss_tr_bfe {:.3f} ({:.3f})\t'
                      'loss_ce_logits1 {:.3f} ({:.3f})\t'
                      'loss_ce_logits2 {:.3f} ({:.3f})\t'
                      'loss_ce_logits3 {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean,
                              losses_ce.val, losses_ce.mean,
                              losses_tr.val, losses_tr.mean,
                              losses_ce_bfe.val, losses_ce_bfe.mean,
                              losses_tr_bfe.val, losses_tr_bfe.mean,
                              losses_ce_logits1.val, losses_ce_logits1.mean,
                              losses_ce_logits2.val, losses_ce_logits2.mean,
                              losses_ce_logits3.val, losses_ce_logits3.mean,
                              ))
        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        if self.opt.random_crop and random.random() > 0.3:
            h, w = imgs.size()[-2:]
            start = int((h-2*w)*random.random())
            mask = imgs.new_zeros(imgs.size())
            mask[:, :, start:start+2*w, :] = 1
            imgs = imgs * mask
        '''
        if random.random() > 0.5:
            h, w = imgs.size()[-2:]
            for attempt in range(100):
                area = h * w
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 3.33)
                ch = int(round(math.sqrt(target_area * aspect_ratio)))
                cw = int(round(math.sqrt(target_area / aspect_ratio)))
                if cw <  w and ch < h:
                    x1 = random.randint(0, h - ch)
                    y1 = random.randint(0, w - cw)
                    imgs[:, :, x1:x1+h, y1:y1+w] = 0
                    break
        '''
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, feat, logits1, logits2, logits3 = self.model(self.data)
        # score: triplet feature
        # feat: softmax feature
        # score[0]: global branch
        # score[1]: local branch
        #self.loss = self.criterion(score, feat, self.target)
        # global ce loss
        loss_ce = self.criterion_ce(feat[0], self.target)
        # global triplet loss
        loss_tr = self.criterion_triple(score[0], score[0],self.target)
        # bfe ce loss
        loss_ce_bfe=self.criterion_ce(feat[1], self.target)
        # bfe triplet loss
        loss_tr_bfe=self.criterion_triple(score[1], score[1],self.target)
        # Local losses
        loss_ce_logits1 = self.criterion_ce(logits1, self.target)
        loss_ce_logits2 = self.criterion_ce(logits2, self.target)
        loss_ce_logits3 = self.criterion_ce(logits3, self.target)
        return loss_ce,loss_tr,loss_ce_bfe,loss_tr_bfe,loss_ce_logits1,loss_ce_logits2,loss_ce_logits3


    def _backward(self):
        self.loss.backward()

class cls_tripletTrainer_RBDLF_weighted:
    def __init__(self, opt, model, optimzier, num_classes, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer= optimzier
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss().cuda()
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr= AverageMeter()
        losses_ce_bfe= AverageMeter()
        losses_tr_bfe= AverageMeter()
        losses_ce_logits1= AverageMeter()
        losses_ce_logits2= AverageMeter()
        losses_ce_logits3= AverageMeter()


        start = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            self._parse_data(inputs)
            loss_ce,loss_tr,loss_ce_bfe,loss_tr_bfe,loss_ce_logits1,loss_ce_logits2,loss_ce_logits3=self._forward()
            self.loss = loss_ce + loss_tr + (loss_ce_bfe + loss_tr_bfe)*0 + (loss_ce_logits1 + loss_ce_logits2 + loss_ce_logits3) * 1
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_ce_bfe.update(loss_ce_bfe.item())
            losses_tr_bfe.update(loss_tr_bfe.item())
            losses_ce_logits1.update(loss_ce_logits1.item())
            losses_ce_logits2.update(loss_ce_logits2.item())
            losses_ce_logits3.update(loss_ce_logits3.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('all_losses', self.loss.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce', loss_ce.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_tr', loss_tr.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_bfe', loss_ce_bfe.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_tr_bfe', loss_tr_bfe.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits1', loss_ce_logits1.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits2', loss_ce_logits2.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits3', loss_ce_logits3.item(), global_step) # iterations
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'All_Loss {:.3f} ({:.3f})\t'
                      'loss_ce {:.3f} ({:.3f})\t'
                      'loss_tr {:.3f} ({:.3f})\t'
                      'loss_ce_bfe {:.3f} ({:.3f})\t'
                      'loss_tr_bfe {:.3f} ({:.3f})\t'
                      'loss_ce_logits1 {:.3f} ({:.3f})\t'
                      'loss_ce_logits2 {:.3f} ({:.3f})\t'
                      'loss_ce_logits3 {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean,
                              losses_ce.val, losses_ce.mean,
                              losses_tr.val, losses_tr.mean,
                              losses_ce_bfe.val, losses_ce_bfe.mean,
                              losses_tr_bfe.val, losses_tr_bfe.mean,
                              losses_ce_logits1.val, losses_ce_logits1.mean,
                              losses_ce_logits2.val, losses_ce_logits2.mean,
                              losses_ce_logits3.val, losses_ce_logits3.mean,
                              ))
        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        if self.opt.random_crop and random.random() > 0.3:
            h, w = imgs.size()[-2:]
            start = int((h-2*w)*random.random())
            mask = imgs.new_zeros(imgs.size())
            mask[:, :, start:start+2*w, :] = 1
            imgs = imgs * mask
        '''
        if random.random() > 0.5:
            h, w = imgs.size()[-2:]
            for attempt in range(100):
                area = h * w
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 3.33)
                ch = int(round(math.sqrt(target_area * aspect_ratio)))
                cw = int(round(math.sqrt(target_area / aspect_ratio)))
                if cw <  w and ch < h:
                    x1 = random.randint(0, h - ch)
                    y1 = random.randint(0, w - cw)
                    imgs[:, :, x1:x1+h, y1:y1+w] = 0
                    break
        '''
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        score, feat, logits1, logits2, logits3 = self.model(self.data)
        # score: triplet feature
        # feat: softmax feature
        # score[0]: global branch
        # score[1]: local branch
        #self.loss = self.criterion(score, feat, self.target)
        # global ce loss
        loss_ce = self.criterion_ce(feat[0], self.target)
        # global triplet loss
        loss_tr = self.criterion_triple(score[0], score[0],self.target)
        # bfe ce loss
        loss_ce_bfe=self.criterion_ce(feat[1], self.target)
        # bfe triplet loss
        loss_tr_bfe=self.criterion_triple(score[1], score[1],self.target)
        # Local losses
        loss_ce_logits1 = self.criterion_ce(logits1, self.target)
        loss_ce_logits2 = self.criterion_ce(logits2, self.target)
        loss_ce_logits3 = self.criterion_ce(logits3, self.target)
        return loss_ce,loss_tr,loss_ce_bfe,loss_tr_bfe,loss_ce_logits1,loss_ce_logits2,loss_ce_logits3


    def _backward(self):
        self.loss.backward()


class cls_tripletTrainerDistiller:
    def __init__(self, opt, model, optimzier, num_classes, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer= optimzier
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=0.0).cuda()
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr= AverageMeter()
        losses_ce_bfe= AverageMeter()
        losses_tr_bfe= AverageMeter()
        losses_ce_logits1= AverageMeter()
        losses_ce_logits2= AverageMeter()
        losses_ce_logits3= AverageMeter()
        losses_ce2 = AverageMeter()
        losses_tr2 = AverageMeter()


        start = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            self._parse_data(inputs)
            loss_ce,loss_tr,loss_ce2,loss_tr2,loss_ce_bfe,loss_tr_bfe,loss_ce_logits1,loss_ce_logits2,loss_ce_logits3=self._forward()
            self.loss = loss_ce + loss_tr + loss_ce2 + loss_tr2 + loss_ce_bfe + loss_tr_bfe + loss_ce_logits1 + loss_ce_logits2 + loss_ce_logits3
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_ce_bfe.update(loss_ce_bfe.item())
            losses_tr_bfe.update(loss_tr_bfe.item())
            losses_ce_logits1.update(loss_ce_logits1.item())
            losses_ce_logits2.update(loss_ce_logits2.item())
            losses_ce_logits3.update(loss_ce_logits3.item())
            losses_ce2.update(loss_ce2.item())
            losses_tr2.update(loss_tr2.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('all_losses', self.loss.item(), global_step) # iterations
            self.summary_writer.add_scalar('BDBNet loss_ce', loss_ce.item(), global_step) # iterations
            self.summary_writer.add_scalar('BDBNet loss_tr', loss_tr.item(), global_step) # iterations
            self.summary_writer.add_scalar('RBDLF loss_ce', loss_ce2.item(), global_step)  # iterations
            self.summary_writer.add_scalar('RBDLF loss_tr', loss_tr2.item(), global_step)  # iterations
            self.summary_writer.add_scalar('loss_ce_bfe', loss_ce_bfe.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_tr_bfe', loss_tr_bfe.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits1', loss_ce_logits1.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits2', loss_ce_logits2.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_ce_logits3', loss_ce_logits3.item(), global_step) # iterations
            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'All_Loss {:.3f} ({:.3f})\t'
                      'loss_ce {:.3f} ({:.3f})\t'
                      'loss_tr {:.3f} ({:.3f})\t'
                      'loss_ce2 {:.3f} ({:.3f})\t'
                      'loss_tr2 {:.3f} ({:.3f})\t'
                      'loss_ce_bfe {:.3f} ({:.3f})\t'
                      'loss_tr_bfe {:.3f} ({:.3f})\t'
                      'loss_ce_logits1 {:.3f} ({:.3f})\t'
                      'loss_ce_logits2 {:.3f} ({:.3f})\t'
                      'loss_ce_logits3 {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean,
                              losses_ce.val, losses_ce.mean,
                              losses_tr.val, losses_tr.mean,
                              losses_ce2.val, losses_ce2.mean,
                              losses_tr2.val, losses_tr2.mean,
                              losses_ce_bfe.val, losses_ce_bfe.mean,
                              losses_tr_bfe.val, losses_tr_bfe.mean,
                              losses_ce_logits1.val, losses_ce_logits1.mean,
                              losses_ce_logits2.val, losses_ce_logits2.mean,
                              losses_ce_logits3.val, losses_ce_logits3.mean,
                              ))
        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        if self.opt.random_crop and random.random() > 0.3:
            h, w = imgs.size()[-2:]
            start = int((h-2*w)*random.random())
            mask = imgs.new_zeros(imgs.size())
            mask[:, :, start:start+2*w, :] = 1
            imgs = imgs * mask
        '''
        if random.random() > 0.5:
            h, w = imgs.size()[-2:]
            for attempt in range(100):
                area = h * w
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 3.33)
                ch = int(round(math.sqrt(target_area * aspect_ratio)))
                cw = int(round(math.sqrt(target_area / aspect_ratio)))
                if cw <  w and ch < h:
                    x1 = random.randint(0, h - ch)
                    y1 = random.randint(0, w - cw)
                    imgs[:, :, x1:x1+h, y1:y1+w] = 0
                    break
        '''
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        tri_feat1, ce_feat1, tri_feat2, ce_feat2, logits1, logits2, logits3 = self.model(self.data)
        # score: triplet feature
        # feat: softmax feature
        # score[0]: global branch
        # score[1]: local branch
        #self.loss = self.criterion(score, feat, self.target)
        # global ce loss
        loss_ce1 = self.criterion_ce(ce_feat1[0], self.target)
        # global triplet loss
        loss_tr1 = self.criterion_triple(tri_feat1[0], tri_feat1[0],self.target)
        # bfe ce loss
        loss_ce_bfe=self.criterion_ce(ce_feat1[1], self.target)
        # bfe triplet loss
        loss_tr_bfe=self.criterion_triple(tri_feat1[1], tri_feat1[1],self.target)
        # Local losses
        loss_ce_logits1 = self.criterion_ce(logits1, self.target)
        loss_ce_logits2 = self.criterion_ce(logits2, self.target)
        loss_ce_logits3 = self.criterion_ce(logits3, self.target)
        # global ce loss
        loss_ce2 = self.criterion_ce(ce_feat2, self.target)
        # global triplet loss
        loss_tr2 = self.criterion_triple(tri_feat2, tri_feat2, self.target)
        return loss_ce1,loss_tr1,loss_ce2,loss_tr2,loss_ce_bfe,loss_tr_bfe,loss_ce_logits1,loss_ce_logits2,loss_ce_logits3

    def _backward(self):
        self.loss.backward()


class cls_tripletTrainerDistillerOriginal:
    def __init__(self, opt, model, optimzier, num_classes, summary_writer):
        self.opt = opt
        self.model = model
        self.optimizer= optimzier
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=0.0).cuda()
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr= AverageMeter()
        losses_ce_bfe= AverageMeter()
        losses_tr_bfe= AverageMeter()

        losses_ce2 = AverageMeter()
        losses_tr2 = AverageMeter()


        start = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - start)

            # model optimizer
            self._parse_data(inputs)
            loss_ce,loss_tr,loss_ce2,loss_tr2,loss_ce_bfe,loss_tr_bfe=self._forward()
            self.loss = loss_ce + loss_tr + loss_ce2 + loss_tr2 + loss_ce_bfe + loss_tr_bfe
            self.optimizer.zero_grad()
            self._backward()
            self.optimizer.step()

            batch_time.update(time.time() - start)
            losses.update(self.loss.item())
            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_ce_bfe.update(loss_ce_bfe.item())
            losses_tr_bfe.update(loss_tr_bfe.item())

            losses_ce2.update(loss_ce2.item())
            losses_tr2.update(loss_tr2.item())

            # tensorboard
            global_step = epoch * len(data_loader) + i
            self.summary_writer.add_scalar('all_losses', self.loss.item(), global_step) # iterations
            self.summary_writer.add_scalar('BDBNet loss_ce', loss_ce.item(), global_step) # iterations
            self.summary_writer.add_scalar('BDBNet loss_tr', loss_tr.item(), global_step) # iterations
            self.summary_writer.add_scalar('RBDLF loss_ce', loss_ce2.item(), global_step)  # iterations
            self.summary_writer.add_scalar('RBDLF loss_tr', loss_tr2.item(), global_step)  # iterations
            self.summary_writer.add_scalar('loss_ce_bfe', loss_ce_bfe.item(), global_step) # iterations
            self.summary_writer.add_scalar('loss_tr_bfe', loss_tr_bfe.item(), global_step) # iterations

            self.summary_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

            start = time.time()

            if (i + 1) % self.opt.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Batch Time {:.3f} ({:.3f})\t'
                      'Data Time {:.3f} ({:.3f})\t'
                      'All_Loss {:.3f} ({:.3f})\t'
                      'loss_ce {:.3f} ({:.3f})\t'
                      'loss_tr {:.3f} ({:.3f})\t'
                      'loss_ce2 {:.3f} ({:.3f})\t'
                      'loss_tr2 {:.3f} ({:.3f})\t'
                      'loss_ce_bfe {:.3f} ({:.3f})\t'
                      'loss_tr_bfe {:.3f} ({:.3f})\t'

                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.mean,
                              data_time.val, data_time.mean,
                              losses.val, losses.mean,
                              losses_ce.val, losses_ce.mean,
                              losses_tr.val, losses_tr.mean,
                              losses_ce2.val, losses_ce2.mean,
                              losses_tr2.val, losses_tr2.mean,
                              losses_ce_bfe.val, losses_ce_bfe.mean,
                              losses_tr_bfe.val, losses_tr_bfe.mean,

                              ))
        param_group = self.optimizer.param_groups
        print('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
              'Lr {:.2e}'
              .format(epoch, batch_time.sum, losses.mean, param_group[0]['lr']))
        print()

    def _parse_data(self, inputs):
        imgs, pids, _ = inputs
        if self.opt.random_crop and random.random() > 0.3:
            h, w = imgs.size()[-2:]
            start = int((h-2*w)*random.random())
            mask = imgs.new_zeros(imgs.size())
            mask[:, :, start:start+2*w, :] = 1
            imgs = imgs * mask
        '''
        if random.random() > 0.5:
            h, w = imgs.size()[-2:]
            for attempt in range(100):
                area = h * w
                target_area = random.uniform(0.02, 0.4) * area
                aspect_ratio = random.uniform(0.3, 3.33)
                ch = int(round(math.sqrt(target_area * aspect_ratio)))
                cw = int(round(math.sqrt(target_area / aspect_ratio)))
                if cw <  w and ch < h:
                    x1 = random.randint(0, h - ch)
                    y1 = random.randint(0, w - cw)
                    imgs[:, :, x1:x1+h, y1:y1+w] = 0
                    break
        '''
        self.data = imgs.cuda()
        self.target = pids.cuda()

    def _forward(self):
        tri_feat1, ce_feat1, tri_feat2, ce_feat2 = self.model(self.data)
        # score: triplet feature
        # feat: softmax feature
        # score[0]: global branch
        # score[1]: local branch
        #self.loss = self.criterion(score, feat, self.target)
        # global ce loss
        loss_ce1 = self.criterion_ce(ce_feat1[0], self.target)
        # global triplet loss
        loss_tr1 = self.criterion_triple(tri_feat1[0], tri_feat1[0],self.target)
        # bfe ce loss
        loss_ce_bfe=self.criterion_ce(ce_feat1[1], self.target)
        # bfe triplet loss
        loss_tr_bfe=self.criterion_triple(tri_feat1[1], tri_feat1[1],self.target)
        # global ce loss
        loss_ce2 = self.criterion_ce(ce_feat2, self.target)
        # global triplet loss
        loss_tr2 = self.criterion_triple(tri_feat2, tri_feat2, self.target)
        return loss_ce1,loss_tr1,loss_ce2,loss_tr2,loss_ce_bfe,loss_tr_bfe

    def _backward(self):
        self.loss.backward()
