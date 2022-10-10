import os
import sys
import time
from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import torch
import logging
import datetime
import argparse

class Config:
    def __str__(self):
        string = self.__class__.__name__ + ":\n"
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                string += attr + " = " + str(getattr(self, attr)) + "\n"

        return string

data_set_path = {
    "ubisoft": {"path":"/data/dance/lafan1/", "clip_len":50, "pos_alpha": 0.1},
    "mstar": {"path":"/data/dance/mstarv2/", "clip_len":64, "pos_alpha": 1},
    "tianyu": {"path":"/data/dance/tianyu/", "clip_len":64, "pos_alpha": 1},
}
class Hyperparameters(Config):
    """
    usage: set hyper-parameters at first
    """
    # project name
    name = 'blender'
    premode = None
    mode = 'inbetween'  # insert, blend, inbetween
    # dataset parameters
    dataset = 'ubisoft'
    pos_ebd = "learnable" # learnable or optimal or sinusoidal or None
    cls_ebd = "add" # add or cat
    inout_net = "conv" # linear or conv
    is_mask_only = True
    is_global = True
    is_pos_only = False # for tsinghua
    is_normalize = True
    poss_scale = 0.1 # valid only for is_normalize = False
    is_conditional = False
    is_FK_loss = False # valid only for is_global = False
    is_T_loss = False # valid only for is_global = True
    weight_decay = 0
    # training param
    # clip_len = 50
    points_num = 22
    feature_size = 256
    # training param
    batch_size = 128
    l_rate = 1e-3
    n_epoch = 1000
    step_size = 200
    gamma = 0.75
    # display interval
    train_interval = 10
    val_interval = 10
    vis_interval = 10
    # snapshot interval
    snapshot = 100
    # output folders for logs and weights
    log_path = "./log/"
    weight_path = './weight/'
    vis_path = "./visualization/"
    prefix = '20210121'
    # use visdom or not
    trained_model = '' 
    gpu_id = 0
    base_id = 12138
    port = 12138
    # Flags
    visdom = True
    m_train = True
    m_eval = True

    def __init__(self, port=22):
        # parser = argparse.ArgumentParser(description="blender argparse")
        # parser.add_argument('--port', default=None)
        # self.args = parser.parse_args()
        self.port = port


        if self.args.port is not None:
            self.port = self.args.port

        if os.getenv("LOAD2RAM", "1") == "0":
            self.is2ram = False

        self.data_root = data_set_path[self.dataset]["path"]
        self.clip_len = data_set_path[self.dataset]["clip_len"]
        self.pos_alpha = data_set_path[self.dataset]["pos_alpha"]

        if self.dataset == "tianyu":
            self.batch_size = 32
            self.l_rate = 1e-4
            self.n_epoch = 1000
            self.step_size = 200
            self.gamma = 0.75

        if sys.platform == "win32":
            # self.data_root = "H:/0blender/newtianyu/data/"
            self.data_root = self.data_root.replace('/data/dance/', 'H:/0blender/0unify/dataset/')
        self.prefix = "{}_".format(time.strftime("%Y%m%d", time.localtime()))
        if self.premode is not None:
            self.trained_model = 'weight/{}_{}_1000.pkl'.format(self.name, self.premode)
        # print("using {} GPUs".format(torch.cuda.device_count()))
        # self.batch_size *= torch.cuda.device_count()

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.weight_path, exist_ok=True)
        os.makedirs(self.vis_path, exist_ok=True)
        if self.visdom:
            self.init_vis()

    def get_logger(self):
        # set logging
        time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_name = self.log_path + 'pytorch-' + self.name + '-' + time_str + '.log'
        # create a log
        """
        init logging
        """
        formatter = logging.Formatter(
            '[%(asctime)s %(levelname)s] %(name)s: %(message)s')
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        fl = logging.FileHandler(log_name)
        fl.setFormatter(formatter)
        # add a handler to logger
        """
        if sys.platform == "win32":
            cmd = logging.StreamHandler()
            cmd.setFormatter(formatter)
            logger.addHandler(cmd)
        """
        logger.addHandler(fl)
        logger.info('setting logging')
        logger.info(self)
        return logger

    def init_vis(self):
        self.vis = visdom.Visdom(env='main', port=self.port)

        self.visloss = \
            self.vis.line(
                X=torch.ones([1, 7]).cpu() * (0),
                Y=torch.zeros([1, 7]).cpu(),
                win=self.base_id + 0,
                opts=dict(xlabel='minibatches',
                        ylabel='Loss',
                        title='Training Loss',
                        legend=['loss_pos', 'loss_quat', 'loss_overall',
                            'val_root', 'val_quat', 
                            'baseline_root', 'baseline_quat']))

        self.visscore = \
            self.vis.line(
                X=torch.ones([1, 9]).cpu() * (0),
                Y=torch.zeros([1, 9]).cpu(),
                win=self.base_id + 1,
                opts=dict(xlabel='minibatches',
                        ylabel='score',
                        title='val score',
                        legend=[
                        'L2Q_zerov', 'L2Q_interp', 'L2Q_blender',
                        'L2P_zerov', 'L2P_interp', 'L2P_blender',
                        'NPSS_zerov', 'NPSS_interp', 'NPSS_blender']))


    def update_loss(self, epoch, loss):
        if self.visdom:
            self.vis.line(X=torch.ones([1, len(loss)]).cpu() * (epoch),
                          Y=torch.from_numpy(np.array(loss)).cpu().view(1, len(loss)),
                          win=self.visloss,
                          update='append')

    def update_score(self, epoch, loss):
        if self.visdom:
            self.vis.line(X=torch.ones([1, len(loss)]).cpu() * (epoch),
                          Y=torch.from_numpy(np.array(loss)).cpu().view(1, len(loss)),
                          win=self.visscore,
                          update='append')

    def update_trace(self, cvs):
        if self.visdom:
            self.vis.image(cvs, win=self.base_id + 10)
