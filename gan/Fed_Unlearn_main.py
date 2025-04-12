# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:35:11 2020

@author: user
"""
import pickle

# %%

import torch
import random
import numpy as np

from Fed_Unlearn_server import Guide
import os
import datetime
"""Step 0. Initialize Federated Unlearning parameters"""


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# ids = [0, 1, 2, 3, 4, 5, 6, 7]


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 128
        self.N_client = 10
        self.num_clients = 128
        # self.data_name = 'ffhq'  # purchase, cifar10, mnist, adult
        # self.global_epoch = 20
        # self.local_epoch = 10

        # Model Training Settings
        # self.local_batch_size = 64
        # self.local_lr = 0.005
        # self.test_batch_size = 64
        # self.seed = 1
        # self.save_all_model = True
        # self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        # # self.train_with_test = False
        self.device = "cuda:0"

        # Federated Unlearning Settings
        self.pretrained_ckpt = "./files/ffhqrebalanced512-128.pkl"
        with open(self.pretrained_ckpt, 'rb') as f:
            snapshot_data = pickle.load(f)
            # 假设snapshot_data包含了G_ema作为模型的参数
            init_model = snapshot_data["G_ema"]
        self.model = init_model

        self.lr = 1e-4
        self.seed = 0
        self.fov_deg = 18.837
        self.truncation_psi = 1.0
        self.truncation_cutoff = 14
        self.iter = 600  # 1000

        # self.inversion_image_path = None
        self.angle_p = -0.2
        self.angle_y_abs = np.pi / 12
        self.sample_views = 11

        self.local = True
        self.loss_local_mse_lambda = 1e-2  #1e-2
        self.loss_local_lpips_lambda = 1.0
        self.loss_local_id_lambda = 0.1

        self.adj = True
        self.loss_adj_mse_lambda = 1e-2
        self.loss_adj_lpips_lambda = 1.0
        self.loss_adj_id_lambda = 0.1
        self.loss_adj_batch = 2
        self.loss_adj_lambda = 1.0
        self.loss_adj_alpha_range_min = 0
        self.loss_adj_alpha_range_max = 15

        self.glob = True
        self.loss_global_lambda = 1.0
        self.loss_global_batch = 2

        self.target_idx = 3
        self.target_cid = 0  #casia shi 066
        self.exp = ("dp8_casia")
        self.inversion = "goae"
        self.target = "custom"
        self.target_d = -10
        self.encoder_ckpt = "./files/encoder_FFHQ.pt"


def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    # 获取当前时间
    current_time = datetime.datetime.now()
    # 打印当前时间
    print("当前时间是:", current_time)
    FL_params = Arguments()
    seed = FL_params.seed
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    print(FL_params.exp)
    # torch.manual_seed(FL_params.seed)

    device = torch.device("cuda:0" if FL_params.use_gpu and torch.cuda.is_available() else "cpu")
    # 实例化 Guide 类
    server = Guide(FL_params)

    # print(60 * '=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    server.guide() # python Fed_Unlearn_main.py
    # print("Step3. Fedearated neggrade Unlearning Training...")
    # server.neggrade()
    # #
    # print("Step4. Finetune Training...")
    # server.finetune_models()
    # # #
    # print("Step4. NegGrad Training...")
    # server.baseline_neggrade()
    #
    current_time2 = datetime.datetime.now()
    cost = current_time2 - current_time
    # 打印当前时间
    print("耗时:", cost)


if __name__ == '__main__':
    Federated_Unlearning()
