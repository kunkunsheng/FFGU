# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:15:13 2020

@author: jojo
"""

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
# from sklearn.metrics import accuracy_score
import numpy as np
import time
import os
import random
# ourself libs
from model_initiation import model_init



class Server(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_clients = args.num_clients
        # self.server_round = args.global_epoch
        # self.global_model = copy.deepcopy(args.model)
        # self.ratio = args.ratio
        self.clients = []
        self.selected_clients = []
        # self.eval_every = args.global_epoch
        self.best_acc = 0.0
        # self.dataset_name = args.dataset
        # self.un_client = []
        # self.data_dir = args.data_dir

    def select_random_clients(self):
        """
        随机选择指定数量的客户端目录
        :param data_dir: 包含所有客户端数据的根目录路径
        :param num_clients: 需要选择的客户端数量
        :return: 随机选择的客户端目录列表
        """
        all_clients = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        # 如果需要排除某些客户端
        all_clients = [client for client in all_clients if client not in self.un_client]
        selected_clients = random.sample(all_clients, self.num_clients)
        self.selected_clients = selected_clients
        return selected_clients
#
# selected_client_dirs = select_random_clients(data_dir, num_clients)
#
# # 打印或保存选中的客户端目录信息
# for client_dir in selected_client_dirs:
#     print(client_dir)
