# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    # Seed
    seed = 0

    # dataset options
    dataset = 'market1501'
    datatype = 'person'
    mode = 'retrieval'
    # optimization options
    loss = 'triplet'
    optim = 'adam'
    max_epoch = 400
    train_batch = 128
    test_batch = 128
    adjust_lr = True
    lr =  1e-3
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    random_crop = False # Why False?
    margin = None
    num_instances = 4
    num_gpu = 2
    evaluate = False
    savefig = None
    re_ranking = False

    # model options
    model_name = 'RBDLF'  # triplet, softmax_triplet, bfe, ide
    last_stride = 1
    pretrained_model = None

    # miscs
    print_freq = 10
    eval_step = 50
    save_dir = './pytorch_ckpt_milestone_optimizer_distiller/market'
    workers = 10
    start_epoch = 0
    best_rank = -np.inf
    paper_setting='c'

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
            if 'cls' in self.dataset:
                self.mode = 'class'
            if 'market' in self.dataset or 'cuhk' in self.dataset or 'duke' in self.dataset:
                self.datatype = 'person'
            elif 'cub' in self.dataset:
                self.datatype = 'cub'
            elif 'car' in self.dataset:
                self.datatype = 'car'
            elif 'clothes' in self.dataset:
                self.datatype = 'clothes'
            elif 'product' in self.dataset:
                self.datatype = 'product'

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
