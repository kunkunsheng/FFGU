# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
import math
from thop import profile, clever_format
import torch.functional as F
import time
from FL_server import Server
from FL_client import Client
import os
import legacy
import dnnlib
import torch
import random
import pickle
import copy
import lpips
import datetime
import torch.nn.functional as F
import numpy as np

from torch import optim
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils.misc import copy_params_and_buffers
from PIL import Image
from arcface import IDLoss



class Guide(Server):
    def __init__(self, args):
        super(Guide, self).__init__(args)
        self.tensor_to_image = self.tensor_to_image
        self.image_to_tensor = self.image_to_tensor
        self.clients = []  # 用于存储所有客户端实例

        # 假设 total_clients 是在 args 中定义的
        total_clients = args.num_clients  # 从父类中获取客户端数量
        # 初始化时创建所有客户端实例，并将它们添加到父类的 self.clients 中
        for cid in range(total_clients):
            # 创建客户端实例
            client = Client(args, cid)
            self.clients.append(client)



    def finetune_models(self):
        device = torch.device("cuda")
        exp = self.args.exp

        from goae import GOAEncoder
        from goae.swin_config import get_config
        # 配置并加载 GOAE 编码器：一个用于将图像转换为潜在空间的模型
        swin_config = get_config()  # 获取 Swin Transformer 的配置
        stage_list = [10000, 20000,
                      30000]  # 指定训练的 Swin Transformer 阶段 定义一个阶段列表，可能用于编码器的多阶段训练或推理过程。每个数字代表一个训练阶段的迭代次数
        encoder_ckpt = "./files/encoder_FFHQ.pt"  # 定义编码器的预训练权重文件路径，这个文件可能包含在 FFHQ（人脸高质量数据集）上训练的权重
        # 使用配置参数实例化 GOAEncoder 编码器，并将其移动到指定的设备（如GPU）
        encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
            device)  # 使用配置参数实例化 GOAEncoder 编码器
        # 加载预训练的权重到编码器中，使其可以执行图像反演任务。
        encoder.load_state_dict(
            torch.load(encoder_ckpt, map_location=device))

        # 获取客户端数量
        num_clients = len(self.clients)
        # 初始化一个字典来存储每个客户端的参数
        client_snapshots = []

        # for cid in range(num_clients):
        for cid in range(1,num_clients):
            # client = self.clients[cid]
            # client.upload_encoder(encoder)
            # client_snapshot = client.train()
            # client_snapshots.append(client_snapshot["G_ema"])
            # self.clients[cid]

            # 动态创建新客户端实例（旧实例自动回收）
            client = Client(self.args, cid)
            client.upload_encoder(encoder)  # 上传最新全局模型参数
            client_snapshot = client.train()
            client_snapshots.append(client_snapshot["G_ema"])
            # 显式删除引用 + 清理显存
            del client, client_snapshot
            torch.cuda.empty_cache()

        # 计算所有客户端的参数平均值
        averaged_state_dict = self.average_params(client_snapshots)

        # 假设我们有一个全局模型，可以用来更新
        pretrained_ckpt = "XXXXXXXX/FU-GUIDE/files/ffhqrebalanced512-128.pkl"

        with dnnlib.util.open_url(pretrained_ckpt) as f:
             global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
             # global_generator.state_dict()
        # 打印 global_generator 的所有参数名称和对应的参数值
        for name, param in global_generator.named_parameters():
            if name in averaged_state_dict:
                # 打印正在更新的参数名称

                # 使用 averaged_state_dict 中的值替换 global_generator 中的参数值
                param.data.copy_(averaged_state_dict[name].data)


        ckpt_dir = f"experiments/{exp}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        snapshot_data = dict()  # 初始化字典 snapshot_data，用于保存模型和数据
        snapshot_data["G_ema"] = copy.deepcopy(global_generator).eval().requires_grad_(False).cpu()
        with open(os.path.join(ckpt_dir, "finetune_final.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)



    def baseline_neggrade(self):
        device = torch.device("cuda")
        from goae import GOAEncoder
        from goae.swin_config import get_config
        # 配置并加载 GOAE 编码器：一个用于将图像转换为潜在空间的模型
        swin_config = get_config()  # 获取 Swin Transformer 的配置
        stage_list = [10000, 20000,
                      30000]  # 指定训练的 Swin Transformer 阶段 定义一个阶段列表，可能用于编码器的多阶段训练或推理过程。每个数字代表一个训练阶段的迭代次数
        encoder_ckpt = "./files/encoder_FFHQ.pt"  # 定义编码器的预训练权重文件路径，这个文件可能包含在 FFHQ（人脸高质量数据集）上训练的权重
        # 使用配置参数实例化 GOAEncoder 编码器，并将其移动到指定的设备（如GPU）
        encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
            device)  # 使用配置参数实例化 GOAEncoder 编码器
        # 加载预训练的权重到编码器中，使其可以执行图像反演任务。
        encoder.load_state_dict(
            torch.load(encoder_ckpt, map_location=device))

        # 获取客户端数量
        num_clients = len(self.clients)
        # 初始化一个字典来存储每个客户端的参数
        client_snapshots = []
        client = Client(self.args, 0)
        client.upload_encoder(encoder)  # 上传最新全局模型参数
        client_snapshot = client.neggrad()
        client_snapshots.append(client_snapshot["G_ema"])
        # for cid in range(num_clients):
        # for cid in range(1,num_clients):
        #
        #     # 动态创建新客户端实例（旧实例自动回收）
        #     client = Client(self.args, cid)
        #     client.upload_encoder(encoder)  # 上传最新全局模型参数
        #     client_snapshot = client.train()
        #     client_snapshots.append(client_snapshot["G_ema"])
        #     # 显式删除引用 + 清理显存
        #     del client, client_snapshot
        #     torch.cuda.empty_cache()


        # 假设我们有一个全局模型，可以用来更新
        pretrained_ckpt = "XXXXXXXX/FU-GUIDE/files/ffhqrebalanced512-128.pkl"

        with dnnlib.util.open_url(pretrained_ckpt) as f:
            global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
            # global_generator.state_dict()
        # 打印 global_generator 的所有参数名称和对应的参数值
        for name, param in global_generator.named_parameters():
            if name in client_snapshot:

                param.data.copy_(client_snapshot[name].data)

        exp = self.args.exp
        ckpt_dir = f"experiments/{exp}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        snapshot_data = dict()  # 初始化字典 snapshot_data，用于保存模型和数据
        snapshot_data["G_ema"] = copy.deepcopy(global_generator).eval().requires_grad_(False).cpu()
        with open(os.path.join(ckpt_dir, "fu_negrad.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)
    #负梯度
    def neggrade(self):
        device = torch.device("cuda")
        with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        # 解包传入位置参数。这意味着 g_source.init_args 是一个包含位置参数的列表或元组，它会逐一解包并传递给
        # g_source 存储了从预训练模型文件中提取的生成器参数，这些参数会在后续过程中用来初始化一个新的 TriPlaneGenerator 生成器对象
        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(
            False).to(
            device)  # 创建一个TriPlaneGenerator对象 这是一个生成器类 用于生成三维图像渲染 ，并加载预训练的生成器参数
        copy_params_and_buffers(init_global_generator, generator, require_all=True)  # 将预训练的生成器参数复制到新的生成器对象中
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution  # 设置生成器的渲染分辨率
        generator.rendering_kwargs = init_global_generator.rendering_kwargs  # 设置生成器的渲染参数
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)  # 加载生成器的状态字典
        generator.train()  # 将生成器设置为训练模式

        init_global_generator = copy.deepcopy(
            generator)  # 这行代码对 generator 进行了深度复制，创建了一个完全独立的 g_source 副本。这是为了确保在后续的操作中，修改 generator 不会影响到 g_source
        # g_source 就是init_global_model 中的所有参数都被“冻结”，即在后续的训练过程中，它们的值不会发生变化，也不会计算它们的梯度
        for name, param in init_global_generator.named_parameters():
            param.requires_grad = False
        for name, param in generator.named_parameters():
            if "backbone.synthesis" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        exp = self.args.exp
        exp_dir = f"experiments/{exp}"
        ckpt_dir = f"experiments/{exp}/checkpoints"
        image_dir = f"experiments/{exp}/training/images"
        result_dir = f"experiments/{exp}/training/results"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        # with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        #     for arg in kwargs:
        #         f.write(f"{arg}: {kwargs[arg]}\n")

        # 计算了用于生成图像的相机参数，包括内参矩阵和从摄像机到世界坐标的变换矩阵。
        intrinsics = FOV_to_intrinsics(self.args.fov_deg, device=device)
        cam_pivot = torch.tensor(generator.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device=device)
        cam_radius = generator.rendering_kwargs.get("avg_cam_radius", 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot,
                                                               radius=cam_radius,
                                                               device=device)
        conditioning_params = torch.cat(
            [conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius,
                                              device=device)
        camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        # 设置优化器，用于更新 generator 模型的参数
        optimizer = optim.Adam(generator.parameters(), lr=self.args.lr)
        # 加载一个预训练的均值向量 w_avg，这通常是在生成对抗网络（GAN）中用来初始化或引导生成过程的中间潜在向量。
        w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device)  # [1, 14, 512]
        inversion = self.args.inversion
        # Visualize before unlearning 通过加载并使用一个名为 GOAEncoder 的模型，将图像转换为模型的潜在空间表示
        with torch.no_grad():  # 在反演阶段通常不需要反向传播
            if inversion is not None:
                # assert inversion_image_path is not None, "The path of an image to invert is required."
                assert inversion in ["goae"]
                if inversion == "goae":
                    from goae import GOAEncoder
                    from goae.swin_config import get_config
                    # 配置并加载 GOAE 编码器：一个用于将图像转换为潜在空间的模型
                    swin_config = get_config()  # 获取 Swin Transformer 的配置
                    stage_list = [10000, 20000,
                                  30000]  # 指定训练的 Swin Transformer 阶段 定义一个阶段列表，可能用于编码器的多阶段训练或推理过程。每个数字代表一个训练阶段的迭代次数
                    encoder_ckpt = "./files/encoder_FFHQ.pt"  # 定义编码器的预训练权重文件路径，这个文件可能包含在 FFHQ（人脸高质量数据集）上训练的权重
                    # 使用配置参数实例化 GOAEncoder 编码器，并将其移动到指定的设备（如GPU）
                    encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
                        device)  # 使用配置参数实例化 GOAEncoder 编码器
                    # 加载预训练的权重到编码器中，使其可以执行图像反演任务。
                    encoder.load_state_dict(
                        torch.load(encoder_ckpt, map_location=device))  # 加载预训练的权重到编码器中，使其可以执行图像反演任务。

                    print(f"target_cid is: {self.args.target_cid}")
                    target_client = self.clients[self.args.target_cid]
                    target_client.upload_encoder(encoder)
                    w, flag, length = target_client.compute_latent_vectors(self.args.target_cid)
                    all_w_list = []
                    selected_image_paths = self.select_random_image_from_each_subfolder()
                    # 对每张选取的图像进行编码
                    for image_path in selected_image_paths:
                        # 打开图像并进行预处理（通过image_to_tensor方法）
                        img = self.image_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
                        client_w, _ = encoder(img)
                        all_w_list.append(client_w)
                    all_w_tensor = torch.cat(all_w_list, dim=0)
                    # 对所有的w向量求平均（在维度0上求平均）
                    average_w = torch.mean(all_w_tensor, dim=0)
                    average_w = self.add_gaussian_noise(average_w)

                    if not flag:
                        # 假设 w 的形状为 (N, D, F)，其中 N 是批次大小
                        N = w.shape[0]
                        target_idx = 3
                        # target_idx = self.args.target_idx
                        print(f"target_idx is: {target_idx}")
                        w_origin = w + w_avg  # 为 w 加上均值向量 w_avg，生成原始潜在向量 w_origin
                        w_u = w[[target_idx], :, :] + w_avg  # 从编码器生成的潜在空间表示中选择一个特定索引的潜在向量，并加上均值向量 w_avg，生成反演后的潜在向量 w_u
                    else:  # 如果是单张图像
                        w_u = w + w_avg
                    w_u = self.add_gaussian_noise(w_u)
                else:
                    raise NotImplementedError

            generator.eval()

            if self.args.inversion is None:  # for random 如果没有图像反演，代码将随机生成多个不同视角的图像并保存
                print("none inversionxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                   np.pi / 2 + self.args.angle_p,
                                                                   cam_pivot,
                                                                   radius=cam_radius, device=device)
                    camera_params_view = torch.cat(
                        [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                        dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)[
                        "image"]  # 使用潜在向量 w_u 和相机参数 camera_params_view 生成图像 img_u
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                del img_u
            else:
                if flag is False:  # for OOD 处理目录中的图像
                    # for i in range(length): w.shape[0]
                    for i in range(length):
                        print("flag is Falseoooooooooooooooooooooooooooooooood")
                        for view, angle_y in enumerate(
                                np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs,
                                            self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                           np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius,
                                                                           device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)

                            img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_before_{i}_{view}.png"))
                    del img_origin
                else:  # 处理单张图像
                    print("flag is Trueiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiid")
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                       np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_u = generator.synthesis(w_u, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                    del img_u
            generator.train()

        if self.args.target == "random":
            z_rg = torch.randn(1, 512, device=device)
            w_target = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                         truncation_cutoff=self.args.truncation_cutoff)
            print("w_target is random")
        elif self.args.target == "neg":
            w_target = w_u
        elif self.args.target == "guide":
            with torch.no_grad():
                if self.args.inversion is not None:
                    print("有有有哟")
                    w_id = w[[target_idx], :, :]
                else:
                    print("没有invesion")
                    w_id = w_u - w_avg
                w_target = w_avg - w_id / w_id.norm(p=2) * self.args.target_d
                print("w_target is guide")
        elif self.args.target == "custom_dp":
            w_custom = average_w + w_avg
            print(f"w_custom is :{ w_custom }")
            if self.args.inversion is not None:
                print("正确的")
                w_id = w[[target_idx], :, :]
            else:
                print("没有invesion")
                w_id = w_u - w_avg
            w_target = w_custom - w_id / w_id.norm(p=2) * self.args.target_d
            print("w_target is dp")
        elif self.args.target == "custom":  # 新增的条件判断
            with torch.no_grad():
                # 假设你要做一些自定义操作，比如：
                w_custom = average_w + w_avg
                w_id = w[[target_idx], :, :]
                w_target = w_custom - w_id / w_id.norm(p=2) * self.args.target_d  # 使用自定义操作生成的 w_target
            print("w_target is custom")

        lpips_fn = lpips.LPIPS(net="vgg").to(device)  # 使用 VGG 网络来计算感知损失，用于衡量生成图像和目标图像在感知上的相似度
        id_fn = IDLoss().to(device)
        # 通过逐步调整相机视角，计算生成图像的损失，并为每次迭代初始化损失值
        pbar = tqdm(range(self.args.iter))
        for i in pbar:
            angle_y = np.random.uniform(-self.args.angle_y_abs, self.args.angle_y_abs)
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                      cam_pivot,
                                                      radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)

            loss = torch.tensor(0.0, device=device)
            loss_dict = {}

            # local unlearning loss 局部损失 这个损失包括均方误差（MSE）、感知损失（LPIPS）和身份损失（IDLoss），这些损失项用于衡量生成图像与目标图像之间的差异
            if self.args.local:
                # 计算生成器 generator 的特征图（feat_u）与目标生成器 g_source 的特征图（feat_target）之间的均方误差（MSE）损失
                loss_local = torch.tensor(0.0, device=device)
                feat_u = generator.get_planes(w_u)  # 获取 generator 在输入 w_u 下生成的特征图。
                feat_target = init_global_generator.get_planes(w_target)  # 获取 g_source 在输入 w_target 下生成的特征图。
                loss_local_mse = F.mse_loss(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_mse_lambda * loss_local_mse
                # 生成图像 img_u 和 img_target，并计算它们之间的感知损失（LPIPS）。
                img_u = generator.synthesis(w_u, camera_params)["image"]
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
                loss_local_lpips = lpips_fn(img_u, img_target).mean()
                loss_local = loss_local + self.args.loss_local_lpips_lambda * loss_local_lpips
                # 计算生成图像 img_u 和目标图像 img_target 之间的身份损失（IDLoss）
                loss_local_id = id_fn(img_u, img_target)
                loss_local = loss_local + self.args.loss_local_id_lambda * loss_local_id
                loss = loss + loss_local

                # 计算L1损失
                # loss_local_l1 = torch.nn.L1Loss()(feat_u, feat_target)
                # loss_local = loss_local + self.args.loss_local_l1_lambda * loss_local_l1  # 加权L1损失
                # loss_dict["loss_local"] = loss_local.item()

            # adjacency-aware unlearning loss  邻域
            if self.args.adj:
                loss_adj = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_adj_batch):
                    z_ra = torch.randn(1, 512, device=device)  # 生成一个随机的潜在向量 z_ra，其形状为 (1, 512)
                    w_ra = generator.mapping(z_ra, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff)  # 通过生成器的映射网络将 z_ra 转换为潜在空间表示 w_ra
                    # 计算调整向量 deltas 并应用到生成器的输入 w_u 和目标向量 w_target 上，生成调整后的潜在向量 w_u_adj 和 w_target_adj
                    if self.args.loss_adj_alpha_range_max is not None:
                        loss_adj_alpha = torch.from_numpy(
                            np.random.uniform(self.args.loss_adj_alpha_range_min,
                                              self.args.loss_adj_alpha_range_max,
                                              size=1)).unsqueeze(
                            1).unsqueeze(1).to(device)
                    # 计算 w_ra 和 w_u 之间的归一化差异向量，并乘以系数 loss_adj_alpha，得到调整向量 deltas
                    deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                    w_u_adj = w_u + deltas  # 调整后的
                    w_target_adj = w_target + deltas  # 调整后的邻域向量
                    # 计算调整后的特征图之间的均方误差（MSE）损失
                    feat_u = generator.get_planes(w_u_adj)
                    feat_target = init_global_generator.get_planes(w_target_adj)
                    loss_adj_mse = F.mse_loss(feat_u, feat_target)
                    loss_adj = loss_adj + self.args.loss_adj_mse_lambda * loss_adj_mse
                    # 计算调整后的感知损失
                    # 分别生成调整后的图像 img_u 和 img_target
                    img_u = generator.synthesis(w_u_adj, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_target_adj, camera_params)["image"]
                    # 计算这两张图像之间的感知损失。
                    loss_adj_lpips = lpips_fn(img_u, img_target).mean()
                    loss_adj = loss_adj + self.args.loss_adj_lpips_lambda * loss_adj_lpips
                    # 计算调整后的身份损失
                    loss_adj_id = id_fn(img_u, img_target)
                    loss_adj = loss_adj + self.args.loss_adj_id_lambda * loss_adj_id
                # 累加到总损失并记录调整损失
                loss = loss + self.args.loss_adj_lambda * loss_adj
                loss_dict["loss_adj"] = loss_adj.item()

            loss_fn = -loss
            # 通过反向传播计算梯度，并使用优化器更新生成器的参数
            optimizer.zero_grad()
            loss_fn.backward()  # 通过反向传播计算损失 loss 对生成器参数的梯度
            optimizer.step()
            # 更新进度条并保存中间结果
            pbar.set_postfix(loss=loss.item(), **loss_dict)

            if i % 5 == 0:  # 每 100 次迭代保存一次生成的图像和目标图像
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_u, camera_params_front)["image"]
                    img_u_save = self.tensor_to_image(img_u_save)
                    img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                    img_target_save = init_global_generator.synthesis(w_target, camera_params_front)["image"]
                    img_target_save = self.tensor_to_image(img_target_save)
                    img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                    generator.train()
                del img_u_save, img_target_save
            # 是在训练过程中定期保存当前生成的图像和目标图像
        with torch.no_grad():
            generator.eval()
            img_u_save = generator.synthesis(w_u, camera_params)["image"]
            img_target_save = init_global_generator.synthesis(w_target, camera_params)["image"]
            img_u_save = self.tensor_to_image(img_u_save)
            img_target_save = self.tensor_to_image(img_target_save)
            img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
            img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
            generator.train()
        # 是在训练后或推理时，根据不同的视角生成并保存图像
        with torch.no_grad():
            generator.eval()
            if self.args.inversion is None:  # for random
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                   np.pi / 2 + self.args.angle_p,
                                                                   cam_pivot,
                                                                   radius=cam_radius, device=device)
                    camera_params_view = torch.cat(
                        [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                        dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            else:
                if flag is False:  # for OOD
                    for i in range(length):
                        for view, angle_y in enumerate(
                                np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs,
                                            self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                           np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius,
                                                                           device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)
                            img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
                else:  # for InD
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                       np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_u = generator.synthesis(w_u, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            generator.train()

        snapshot_data = dict()  # 初始化字典 snapshot_data，用于保存模型和数据
        snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        current_time = datetime.datetime.now()
        # 打印当前时间
        print("当前时间是:", current_time)
        with open(os.path.join(ckpt_dir, f"last.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)

    def average_params(self, client_snapshots):
        # """聚合所有客户端的 G_ema 参数"""
        # # 提取所有客户端的 G_ema 参数字典
        # client_params_list = [snapshot["G_ema"] for snapshot in client_snapshots]
        #
        # # 初始化聚合字典
        # averaged_params = {}
        # model_keys = client_params_list[0].keys()  # 安全访问参数键
        #
        # # 参数平均化
        # for key in model_keys:
        #     stacked_params = torch.stack([client_params[key] for client_params in client_params_list])
        #     averaged_params[key] = torch.mean(stacked_params, dim=0)
        # # 初始化聚合字典
        averaged_params = {}

        # 获取模型参数的所有键
        model_keys = set()  # 使用集合以避免重复键

        # 收集所有客户端的参数
        for snapshot in client_snapshots:
            model_keys.update(snapshot.keys())  # 更新键集合

        # 参数平均化
        for key in model_keys:
            # 收集所有客户端对应键的参数
            stacked_params = torch.stack([snapshot[key] for snapshot in client_snapshots if key in snapshot])

            # 计算平均值并存储
            averaged_params[key] = torch.mean(stacked_params, dim=0)

        return averaged_params

    def tensor_to_image(self, t):
        t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(t[0].cpu().numpy(), "RGB")

    def image_to_tensor(self, i, size=256):
        i = i.resize((size, size))
        i = np.array(i)
        i = i.transpose(2, 0, 1)
        i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
        return i

    def fuguide(self):
        device = torch.device("cuda")
        with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        # 解包传入位置参数。这意味着 g_source.init_args 是一个包含位置参数的列表或元组，它会逐一解包并传递给
        # g_source 存储了从预训练模型文件中提取的生成器参数，这些参数会在后续过程中用来初始化一个新的 TriPlaneGenerator 生成器对象
        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(
            False).to(
            device)  # 创建一个TriPlaneGenerator对象 这是一个生成器类 用于生成三维图像渲染 ，并加载预训练的生成器参数
        copy_params_and_buffers(init_global_generator, generator, require_all=True)  # 将预训练的生成器参数复制到新的生成器对象中
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution  # 设置生成器的渲染分辨率
        generator.rendering_kwargs = init_global_generator.rendering_kwargs  # 设置生成器的渲染参数
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)  # 加载生成器的状态字典
        generator.train()  # 将生成器设置为训练模式

        init_global_generator = copy.deepcopy(
            generator)  # 这行代码对 generator 进行了深度复制，创建了一个完全独立的 g_source 副本。这是为了确保在后续的操作中，修改 generator 不会影响到 g_source
        # g_source 就是init_global_model 中的所有参数都被“冻结”，即在后续的训练过程中，它们的值不会发生变化，也不会计算它们的梯度
        for name, param in init_global_generator.named_parameters():
            param.requires_grad = False
        for name, param in generator.named_parameters():
            if "backbone.synthesis" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        exp = self.args.exp
        exp_dir = f"experiments/{exp}"
        ckpt_dir = f"experiments/{exp}/checkpoints"
        image_dir = f"experiments/{exp}/training/images"
        result_dir = f"experiments/{exp}/training/results"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        # with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        #     for arg in kwargs:
        #         f.write(f"{arg}: {kwargs[arg]}\n")

        # 计算了用于生成图像的相机参数，包括内参矩阵和从摄像机到世界坐标的变换矩阵。
        intrinsics = FOV_to_intrinsics(self.args.fov_deg, device=device)
        cam_pivot = torch.tensor(generator.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device=device)
        cam_radius = generator.rendering_kwargs.get("avg_cam_radius", 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                                               device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius, device=device)
        camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        # 设置优化器，用于更新 generator 模型的参数
        optimizer = optim.Adam(generator.parameters(), lr=self.args.lr)
        # 加载一个预训练的均值向量 w_avg，这通常是在生成对抗网络（GAN）中用来初始化或引导生成过程的中间潜在向量。
        w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device)  # [1, 14, 512]
        inversion = self.args.inversion
        # Visualize before unlearning 通过加载并使用一个名为 GOAEncoder 的模型，将图像转换为模型的潜在空间表示
        with torch.no_grad():  # 在反演阶段通常不需要反向传播
            if inversion is not None:
                # assert inversion_image_path is not None, "The path of an image to invert is required."
                assert inversion in ["goae"]
                if inversion == "goae":
                    from goae import GOAEncoder
                    from goae.swin_config import get_config
                    # 配置并加载 GOAE 编码器：一个用于将图像转换为潜在空间的模型
                    swin_config = get_config()  # 获取 Swin Transformer 的配置
                    stage_list = [10000, 20000,
                                  30000]  # 指定训练的 Swin Transformer 阶段 定义一个阶段列表，可能用于编码器的多阶段训练或推理过程。每个数字代表一个训练阶段的迭代次数
                    encoder_ckpt = "./files/encoder_FFHQ.pt"  # 定义编码器的预训练权重文件路径，这个文件可能包含在 FFHQ（人脸高质量数据集）上训练的权重
                    # 使用配置参数实例化 GOAEncoder 编码器，并将其移动到指定的设备（如GPU）
                    encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
                        device)  # 使用配置参数实例化 GOAEncoder 编码器
                    # 加载预训练的权重到编码器中，使其可以执行图像反演任务。
                    encoder.load_state_dict(
                        torch.load(encoder_ckpt, map_location=device))  # 加载预训练的权重到编码器中，使其可以执行图像反演任务。

                    print(f"target_cid is: {self.args.target_cid}")
                    target_client = self.clients[self.args.target_cid]
                    target_client.upload_encoder(encoder)
                    w, flag, length = target_client.compute_latent_vectors(self.args.target_cid)

                    if not flag:
                        # 假设 w 的形状为 (N, D, F)，其中 N 是批次大小
                        N = w.shape[0]
                        target_idx = 3
                        # target_idx = self.args.target_idx
                        print(f"target_idx is: {target_idx}")
                        w_origin = w + w_avg  # 为 w 加上均值向量 w_avg，生成原始潜在向量 w_origin
                        w_u = w[[target_idx], :, :] + w_avg  # 从编码器生成的潜在空间表示中选择一个特定索引的潜在向量，并加上均值向量 w_avg，生成反演后的潜在向量 w_u
                    else:  # 如果是单张图像
                        w_u = w + w_avg
                else:
                    raise NotImplementedError
            # else:  # 当没有进行图像反演时，根据条件生成潜在向量 w_u
            #     w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device).unsqueeze(
            #         0)  # [1, 14, 512]
            #     if inversion_image_path is not None:
            #         w_u = torch.load(inversion_image_path)
            #     else:
            #         z_u = torch.randn(1, 512, device=device)  # 如果没有提供反演路径，则生成一个随机的潜在向量 z_u
            #         w_u = generator.mapping(z_u, conditioning_params, truncation_psi=self.args.truncation_psi,
            #                                 truncation_cutoff=self.args.truncation_cutoff)
            #         # 通过生成器的映射网络将随机潜在向量 z_u 转换为 w_u
            generator.eval()

            if self.args.inversion is None:  # for random 如果没有图像反演，代码将随机生成多个不同视角的图像并保存
                print("none inversionxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                                   cam_pivot,
                                                                   radius=cam_radius, device=device)
                    camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                                   dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)[
                        "image"]  # 使用潜在向量 w_u 和相机参数 camera_params_view 生成图像 img_u
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                del img_u
            else:
                if flag is False:  # for OOD 处理目录中的图像
                    # for i in range(length): w.shape[0]
                    for i in range(length):
                        print("flag is Falseoooooooooooooooooooooooooooooooood")
                        for view, angle_y in enumerate(np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius, device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)

                            img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_before_{i}_{view}.png"))
                    del img_origin
                else:  # 处理单张图像
                    print("flag is Trueiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiid")
                    for view, angle_y in enumerate(np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                                       dim=1)
                        img_u = generator.synthesis(w_u, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                    del img_u
            generator.train()

        if self.args.target == "random":
            z_rg= torch.randn(1, 512, device=device)
            w_target = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                     truncation_cutoff=self.args.truncation_cutoff)
            print("w_target is random")
        #guide
        elif self.args.target == "guide":
            with torch.no_grad():
                if self.args.inversion is not None:
                    print("有有有哟")
                    w_id = w[[target_idx], :, :]
                else:
                    print("没有invesion")
                    w_id = w_u - w_avg
                w_target = w_avg - w_id / w_id.norm(p=2) * self.args.target_d
                print("w_target is guide")

        lpips_fn = lpips.LPIPS(net="vgg").to(device)  # 使用 VGG 网络来计算感知损失，用于衡量生成图像和目标图像在感知上的相似度
        id_fn = IDLoss().to(device)
        # 通过逐步调整相机视角，计算生成图像的损失，并为每次迭代初始化损失值
        pbar = tqdm(range(self.args.iter))
        for i in pbar:
            angle_y = np.random.uniform(-self.args.angle_y_abs, self.args.angle_y_abs)
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p, cam_pivot,
                                                        radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)

            loss = torch.tensor(0.0, device=device)
            loss_dict = {}

            # local unlearning loss 局部损失 这个损失包括均方误差（MSE）、感知损失（LPIPS）和身份损失（IDLoss），这些损失项用于衡量生成图像与目标图像之间的差异
            if self.args.local:
                # 计算生成器 generator 的特征图（feat_u）与目标生成器 g_source 的特征图（feat_target）之间的均方误差（MSE）损失
                loss_local = torch.tensor(0.0, device=device)
                feat_u = generator.get_planes(w_u)  # 获取 generator 在输入 w_u 下生成的特征图。
                feat_target = init_global_generator.get_planes(w_target)  # 获取 g_source 在输入 w_target 下生成的特征图。
                loss_local_mse = F.mse_loss(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_mse_lambda * loss_local_mse
                # 生成图像 img_u 和 img_target，并计算它们之间的感知损失（LPIPS）。
                img_u = generator.synthesis(w_u, camera_params)["image"]
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
                loss_local_lpips = lpips_fn(img_u, img_target).mean()
                loss_local = loss_local + self.args.loss_local_lpips_lambda * loss_local_lpips
                # 计算生成图像 img_u 和目标图像 img_target 之间的身份损失（IDLoss）
                loss_local_id = id_fn(img_u, img_target)
                loss_local = loss_local + self.args.loss_local_id_lambda * loss_local_id
                loss = loss + loss_local
                loss_dict["loss_local"] = loss_local.item()

            # adjacency-aware unlearning loss  邻域
            if self.args.adj:
                loss_adj = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_adj_batch):
                    z_ra = torch.randn(1, 512, device=device)  # 生成一个随机的潜在向量 z_ra，其形状为 (1, 512)
                    w_ra = generator.mapping(z_ra, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff)  # 通过生成器的映射网络将 z_ra 转换为潜在空间表示 w_ra
                    # 计算调整向量 deltas 并应用到生成器的输入 w_u 和目标向量 w_target 上，生成调整后的潜在向量 w_u_adj 和 w_target_adj
                    if self.args.loss_adj_alpha_range_max is not None:
                        loss_adj_alpha = torch.from_numpy(
                            np.random.uniform(self.args.loss_adj_alpha_range_min,
                                              self.args.loss_adj_alpha_range_max,
                                              size=1)).unsqueeze(
                            1).unsqueeze(1).to(device)
                    # 计算 w_ra 和 w_u 之间的归一化差异向量，并乘以系数 loss_adj_alpha，得到调整向量 deltas
                    deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                    w_u_adj = w_u + deltas  # 调整后的
                    w_target_adj = w_target + deltas  # 调整后的邻域向量
                    # 计算调整后的特征图之间的均方误差（MSE）损失
                    feat_u = generator.get_planes(w_u_adj)
                    feat_target = init_global_generator.get_planes(w_target_adj)
                    loss_adj_mse = F.mse_loss(feat_u, feat_target)
                    loss_adj = loss_adj + self.args.loss_adj_mse_lambda * loss_adj_mse
                    # 计算调整后的感知损失
                    # 分别生成调整后的图像 img_u 和 img_target
                    img_u = generator.synthesis(w_u_adj, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_target_adj, camera_params)["image"]
                    # 计算这两张图像之间的感知损失。
                    loss_adj_lpips = lpips_fn(img_u, img_target).mean()
                    loss_adj = loss_adj + self.args.loss_adj_lpips_lambda * loss_adj_lpips
                    # 计算调整后的身份损失
                    loss_adj_id = id_fn(img_u, img_target)
                    loss_adj = loss_adj + self.args.loss_adj_id_lambda * loss_adj_id
                # 累加到总损失并记录调整损失
                loss = loss + self.args.loss_adj_lambda * loss_adj
                loss_dict["loss_adj"] = loss_adj.item()

            # global preservation loss
            if self.args.glob:
                loss_global = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_global_batch):
                    z_rg = torch.randn(1, 512, device=device)
                    w_rg = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff)

                    img_u = generator.synthesis(w_rg, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_rg, camera_params)["image"]
                    loss_global_lpips = lpips_fn(img_u, img_target).mean()
                    loss_global = loss_global + loss_global_lpips
                loss = loss + self.args.loss_global_lambda * loss_global
                loss_dict["loss_global"] = loss_global.item()
            # 通过反向传播计算梯度，并使用优化器更新生成器的参数
            optimizer.zero_grad()
            loss.backward()  # 通过反向传播计算损失 loss 对生成器参数的梯度
            optimizer.step()
            # 更新进度条并保存中间结果
            pbar.set_postfix(loss=loss.item(), **loss_dict)

            if i % 100 == 0:  # 每 100 次迭代保存一次生成的图像和目标图像
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_u, camera_params_front)["image"]
                    img_u_save = self.tensor_to_image(img_u_save)
                    img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                    img_target_save = init_global_generator.synthesis(w_target, camera_params_front)["image"]
                    img_target_save = self.tensor_to_image(img_target_save)
                    img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                    generator.train()
                del img_u_save, img_target_save
            # 是在训练过程中定期保存当前生成的图像和目标图像
        with torch.no_grad():
            generator.eval()
            img_u_save = generator.synthesis(w_u, camera_params)["image"]
            img_target_save = init_global_generator.synthesis(w_target, camera_params)["image"]
            img_u_save = self.tensor_to_image(img_u_save)
            img_target_save = self.tensor_to_image(img_target_save)
            img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
            img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
            generator.train()
        # 是在训练后或推理时，根据不同的视角生成并保存图像
        with torch.no_grad():
            generator.eval()
            if self.args.inversion is None:  # for random
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                   np.pi / 2 + self.args.angle_p,
                                                                   cam_pivot,
                                                                   radius=cam_radius, device=device)
                    camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                                   dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            else:
                if flag is False:   # for OOD
                    for i in range(length):
                        for view, angle_y in enumerate(
                                np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius,
                                                                           device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)
                            img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
                else:  # for InD
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                       np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_u = generator.synthesis(w_u, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            generator.train()

        snapshot_data = dict()  # 初始化字典 snapshot_data，用于保存模型和数据
        snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        with open(os.path.join(ckpt_dir, f"last.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)

    def select_random_image_from_each_subfolder(self,folder='XXXXXXXX/FU-GUIDE/Deep3DFaceRecon/checkpoints/model_name/casia'):
        # 获取所有子文件夹（000 到 127）
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

        # 存储每个子文件夹中选取的图像路径
        selected_images = []

        # 遍历每个子文件夹
        for subfolder in subfolders:
            # 列出该子文件夹中的所有图像文件（.jpg, .jpeg, .png）
            image_files = [f.path for f in os.scandir(subfolder) if
                           f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]

            if image_files:  # 如果子文件夹中有图像文件
                # 随机选择一张图像
                random_image_path = random.choice(image_files)
                selected_images.append(random_image_path)

        return selected_images

    def guide(self):
        device = torch.device("cuda")
        with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        # 解包传入位置参数。这意味着 g_source.init_args 是一个包含位置参数的列表或元组，它会逐一解包并传递给
        # g_source 存储了从预训练模型文件中提取的生成器参数，这些参数会在后续过程中用来初始化一个新的 TriPlaneGenerator 生成器对象
        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(
            False).to(
            device)  # 创建一个TriPlaneGenerator对象 这是一个生成器类 用于生成三维图像渲染 ，并加载预训练的生成器参数
        copy_params_and_buffers(init_global_generator, generator, require_all=True)  # 将预训练的生成器参数复制到新的生成器对象中
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution  # 设置生成器的渲染分辨率
        generator.rendering_kwargs = init_global_generator.rendering_kwargs  # 设置生成器的渲染参数
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)  # 加载生成器的状态字典
        generator.train()  # 将生成器设置为训练模式

        init_global_generator = copy.deepcopy(
            generator)  # 这行代码对 generator 进行了深度复制，创建了一个完全独立的 g_source 副本。这是为了确保在后续的操作中，修改 generator 不会影响到 g_source
        # g_source 就是init_global_model 中的所有参数都被“冻结”，即在后续的训练过程中，它们的值不会发生变化，也不会计算它们的梯度
        for name, param in init_global_generator.named_parameters():
            param.requires_grad = False
        for name, param in generator.named_parameters():
            if "backbone.synthesis" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        exp = self.args.exp
        exp_dir = f"experiments/{exp}"
        ckpt_dir = f"experiments/{exp}/checkpoints"
        image_dir = f"experiments/{exp}/training/images"
        result_dir = f"experiments/{exp}/training/results"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        # with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        #     for arg in kwargs:
        #         f.write(f"{arg}: {kwargs[arg]}\n")

        # 计算了用于生成图像的相机参数，包括内参矩阵和从摄像机到世界坐标的变换矩阵。
        intrinsics = FOV_to_intrinsics(self.args.fov_deg, device=device)
        cam_pivot = torch.tensor(generator.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device=device)
        cam_radius = generator.rendering_kwargs.get("avg_cam_radius", 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot,
                                                               radius=cam_radius,
                                                               device=device)
        conditioning_params = torch.cat(
            [conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius,
                                              device=device)
        camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        # 设置优化器，用于更新 generator 模型的参数
        optimizer = optim.Adam(generator.parameters(), lr=self.args.lr)
        # 加载一个预训练的均值向量 w_avg，这通常是在生成对抗网络（GAN）中用来初始化或引导生成过程的中间潜在向量。
        w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device)  # [1, 14, 512]
        inversion = self.args.inversion
        # Visualize before unlearning 通过加载并使用一个名为 GOAEncoder 的模型，将图像转换为模型的潜在空间表示
        with torch.no_grad():  # 在反演阶段通常不需要反向传播
            if inversion is not None:
                # assert inversion_image_path is not None, "The path of an image to invert is required."
                assert inversion in ["goae"]
                if inversion == "goae":
                    from goae import GOAEncoder
                    from goae.swin_config import get_config
                    # 配置并加载 GOAE 编码器：一个用于将图像转换为潜在空间的模型
                    swin_config = get_config()  # 获取 Swin Transformer 的配置
                    stage_list = [10000, 20000,
                                  30000]  # 指定训练的 Swin Transformer 阶段 定义一个阶段列表，可能用于编码器的多阶段训练或推理过程。每个数字代表一个训练阶段的迭代次数
                    encoder_ckpt = "./files/encoder_FFHQ.pt"  # 定义编码器的预训练权重文件路径，这个文件可能包含在 FFHQ（人脸高质量数据集）上训练的权重
                    # 使用配置参数实例化 GOAEncoder 编码器，并将其移动到指定的设备（如GPU）
                    encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
                        device)  # 使用配置参数实例化 GOAEncoder 编码器
                    # 加载预训练的权重到编码器中，使其可以执行图像反演任务。
                    encoder.load_state_dict(
                        torch.load(encoder_ckpt, map_location=device))  # 加载预训练的权重到编码器中，使其可以执行图像反演任务。

                    print(f"target_cid is: {self.args.target_cid}")
                    target_client = self.clients[self.args.target_cid]
                    target_client.upload_encoder(encoder)
                    w, flag, length = target_client.compute_latent_vectors(self.args.target_cid)
                    all_w_list = []
                    selected_image_paths = self.select_random_image_from_each_subfolder()
                    # 对每张选取的图像进行编码
                    for image_path in selected_image_paths:
                        # 打开图像并进行预处理（通过image_to_tensor方法）
                        img = self.image_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
                        client_w, _ = encoder(img)
                        # print(img.shape)
                        # flops, params = profile(encoder, inputs=(img,))
                        # flops, params = clever_format([flops, params], "%.3f")
                        # print(f"encoder FLOPs: {flops}")
                        # print(f"encoder Params: {params}")
                        all_w_list.append(client_w)
                    all_w_tensor = torch.cat(all_w_list, dim=0)
                    # 对所有的w向量求平均（在维度0上求平均）
                    average_w = torch.mean(all_w_tensor, dim=0)
                    average_w = self.add_gaussian_noise(average_w)

                    if not flag:
                        # 假设 w 的形状为 (N, D, F)，其中 N 是批次大小
                        N = w.shape[0]
                        target_idx = 3
                        # target_idx = self.args.target_idx
                        print(f"target_idx is: {target_idx}")
                        w_origin = w + w_avg  # 为 w 加上均值向量 w_avg，生成原始潜在向量 w_origin
                        w_u = w[[target_idx], :, :] + w_avg  # 从编码器生成的潜在空间表示中选择一个特定索引的潜在向量，并加上均值向量 w_avg，生成反演后的潜在向量 w_u
                    else:  # 如果是单张图像
                        w_u = w + w_avg
                    w_u = self.add_gaussian_noise(w_u)
                else:
                    raise NotImplementedError

            generator.eval()

            if self.args.inversion is None:  # for random 如果没有图像反演，代码将随机生成多个不同视角的图像并保存
                print("none inversionxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                   np.pi / 2 + self.args.angle_p,
                                                                   cam_pivot,
                                                                   radius=cam_radius, device=device)
                    camera_params_view = torch.cat(
                        [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                        dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)[
                        "image"]  # 使用潜在向量 w_u 和相机参数 camera_params_view 生成图像 img_u
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                del img_u
            else:
                if flag is False:  # for OOD 处理目录中的图像
                    # for i in range(length): w.shape[0]
                    for i in range(length):
                        print("flag is Falseoooooooooooooooooooooooooooooooood")
                        for view, angle_y in enumerate(
                                np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs,
                                            self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                           np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius,
                                                                           device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)

                            img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_before_{i}_{view}.png"))
                    del img_origin
                else:  # 处理单张图像
                    print("flag is Trueiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiid")
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                       np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_u = generator.synthesis(w_u, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                    del img_u
            generator.train()
        # 朴素删除 random
        if self.args.target == "random":
            z_rg = torch.randn(1, 512, device=device)
            w_target = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                         truncation_cutoff=self.args.truncation_cutoff)
            print("w_target is random")
        # guide
        elif self.args.target == "guide":
            with torch.no_grad():
                if self.args.inversion is not None:
                    print("有有有哟")
                    w_id = w[[target_idx], :, :]
                else:
                    print("没有invesion")
                    w_id = w_u - w_avg
                w_target = w_avg - w_id / w_id.norm(p=2) * self.args.target_d
                print("w_target is guide")
         #我们的方法
        elif self.args.target == "custom_dp":
            w_custom = average_w + w_avg
            print(f"w_custom is :{ w_custom }")
            if self.args.inversion is not None:
                print("正确的")
                w_id = w[[target_idx], :, :]
            else:
                print("没有invesion")
                w_id = w_u - w_avg
            w_target = w_custom - w_id / w_id.norm(p=2) * self.args.target_d
            print("w_target is dp")
        elif self.args.target == "custom":  # 新增的条件判断
            with torch.no_grad():
                # 假设你要做一些自定义操作，比如：
                w_custom = average_w + w_avg
                w_id = w[[target_idx], :, :]
                w_target = w_custom - w_id / w_id.norm(p=2) * self.args.target_d  # 使用自定义操作生成的 w_target
            print("w_target is custom")

        lpips_fn = lpips.LPIPS(net="vgg").to(device)  # 使用 VGG 网络来计算感知损失，用于衡量生成图像和目标图像在感知上的相似度
        id_fn = IDLoss().to(device)
        # 通过逐步调整相机视角，计算生成图像的损失，并为每次迭代初始化损失值
        pbar = tqdm(range(self.args.iter))
        for i in pbar:
            angle_y = np.random.uniform(-self.args.angle_y_abs, self.args.angle_y_abs)
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + self.args.angle_p,
                                                      cam_pivot,
                                                      radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)

            loss = torch.tensor(0.0, device=device)
            loss_dict = {}

            # local unlearning loss 局部损失 这个损失包括均方误差（MSE）、感知损失（LPIPS）和身份损失（IDLoss），这些损失项用于衡量生成图像与目标图像之间的差异
            if self.args.local:
                # 计算生成器 generator 的特征图（feat_u）与目标生成器 g_source 的特征图（feat_target）之间的均方误差（MSE）损失
                loss_local = torch.tensor(0.0, device=device)
                feat_u = generator.get_planes(w_u)  # 获取 generator 在输入 w_u 下生成的特征图。
                feat_target = init_global_generator.get_planes(w_target)  # 获取 g_source 在输入 w_target 下生成的特征图。
                loss_local_mse = F.mse_loss(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_mse_lambda * loss_local_mse
                # 生成图像 img_u 和 img_target，并计算它们之间的感知损失（LPIPS）。
                img_u = generator.synthesis(w_u, camera_params)["image"]
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
                # w_f = torch.randn(1, 512).to(device)  # 假设输入为一个潜在向量
                #
                # # 假设 camera_params 的形状是 (1, 25) - 批次大小为 1，相机参数为 25 维
                # ccc= torch.randn(1, 25).to(device)
                # flops, params = profile(generator, inputs=(w_f, ccc))
                # flops, params = clever_format([flops, params], "%.3f")
                # print(f"gen FLOPs: {flops}")
                # print(f"gen Params: {params}")
                loss_local_lpips = lpips_fn(img_u, img_target).mean()
                loss_local = loss_local + self.args.loss_local_lpips_lambda * loss_local_lpips
                # 计算生成图像 img_u 和目标图像 img_target 之间的身份损失（IDLoss）
                loss_local_id = id_fn(img_u, img_target)
                loss_local = loss_local + self.args.loss_local_id_lambda * loss_local_id
                loss = loss + loss_local
                # 计算L1损失
                # loss_local_l1 = torch.nn.L1Loss()(feat_u, feat_target)
                # loss_local = loss_local + self.args.loss_local_l1_lambda * loss_local_l1  # 加权L1损失
                # loss_dict["loss_local"] = loss_local.item()

            # adjacency-aware unlearning loss  邻域
            if self.args.adj:
                loss_adj = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_adj_batch):
                    z_ra = torch.randn(1, 512, device=device)  # 生成一个随机的潜在向量 z_ra，其形状为 (1, 512)
                    w_ra = generator.mapping(z_ra, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff)  # 通过生成器的映射网络将 z_ra 转换为潜在空间表示 w_ra
                    # 计算调整向量 deltas 并应用到生成器的输入 w_u 和目标向量 w_target 上，生成调整后的潜在向量 w_u_adj 和 w_target_adj
                    if self.args.loss_adj_alpha_range_max is not None:
                        loss_adj_alpha = torch.from_numpy(
                            np.random.uniform(self.args.loss_adj_alpha_range_min,
                                              self.args.loss_adj_alpha_range_max,
                                              size=1)).unsqueeze(
                            1).unsqueeze(1).to(device)
                    # 计算 w_ra 和 w_u 之间的归一化差异向量，并乘以系数 loss_adj_alpha，得到调整向量 deltas
                    deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                    w_u_adj = w_u + deltas  # 调整后的
                    w_target_adj = w_target + deltas  # 调整后的邻域向量
                    # 计算调整后的特征图之间的均方误差（MSE）损失
                    feat_u = generator.get_planes(w_u_adj)
                    feat_target = init_global_generator.get_planes(w_target_adj)
                    loss_adj_mse = F.mse_loss(feat_u, feat_target)
                    loss_adj = loss_adj + self.args.loss_adj_mse_lambda * loss_adj_mse
                    # 计算调整后的感知损失
                    # 分别生成调整后的图像 img_u 和 img_target
                    img_u = generator.synthesis(w_u_adj, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_target_adj, camera_params)["image"]
                    # 计算这两张图像之间的感知损失。
                    loss_adj_lpips = lpips_fn(img_u, img_target).mean()
                    loss_adj = loss_adj + self.args.loss_adj_lpips_lambda * loss_adj_lpips
                    # 计算调整后的身份损失
                    loss_adj_id = id_fn(img_u, img_target)
                    loss_adj = loss_adj + self.args.loss_adj_id_lambda * loss_adj_id
                # 累加到总损失并记录调整损失
                loss = loss + self.args.loss_adj_lambda * loss_adj
                loss_dict["loss_adj"] = loss_adj.item()

            # global preservation loss
            if self.args.glob:
                loss_global = torch.tensor(0.0, device=device)
                for _ in range(self.args.loss_global_batch):
                    z_rg = torch.randn(1, 512, device=device)
                    w_rg = generator.mapping(z_rg, conditioning_params, truncation_psi=self.args.truncation_psi,
                                             truncation_cutoff=self.args.truncation_cutoff)

                    img_u = generator.synthesis(w_rg, camera_params)["image"]
                    img_target = init_global_generator.synthesis(w_rg, camera_params)["image"]
                    loss_global_lpips = lpips_fn(img_u, img_target).mean()
                    loss_global = loss_global + loss_global_lpips
                loss = loss + self.args.loss_global_lambda * loss_global
                loss_dict["loss_global"] = loss_global.item()
            # 通过反向传播计算梯度，并使用优化器更新生成器的参数
            optimizer.zero_grad()
            loss.backward()  # 通过反向传播计算损失 loss 对生成器参数的梯度
            optimizer.step()
            # 更新进度条并保存中间结果
            pbar.set_postfix(loss=loss.item(), **loss_dict)

            if i % 100 == 0:  # 每 100 次迭代保存一次生成的图像和目标图像
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_u, camera_params_front)["image"]
                    img_u_save = self.tensor_to_image(img_u_save)
                    img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                    img_target_save = init_global_generator.synthesis(w_target, camera_params_front)["image"]
                    img_target_save = self.tensor_to_image(img_target_save)
                    img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                    generator.train()
                del img_u_save, img_target_save
            # 是在训练过程中定期保存当前生成的图像和目标图像
        with torch.no_grad():
            generator.eval()
            img_u_save = generator.synthesis(w_u, camera_params)["image"]
            img_target_save = init_global_generator.synthesis(w_target, camera_params)["image"]
            img_u_save = self.tensor_to_image(img_u_save)
            img_target_save = self.tensor_to_image(img_target_save)
            img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
            img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
            generator.train()
        # 是在训练后或推理时，根据不同的视角生成并保存图像
        with torch.no_grad():
            generator.eval()
            if self.args.inversion is None:  # for random
                for view, angle_y in enumerate(
                        np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                   np.pi / 2 + self.args.angle_p,
                                                                   cam_pivot,
                                                                   radius=cam_radius, device=device)
                    camera_params_view = torch.cat(
                        [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                        dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
                    # 格式化 FLOPs 和参数数量，并打印
                    flops, params = profile(generator, inputs=(w_u, camera_params_view))
                    flops, params = clever_format([flops, params], "%.3f")
                    print(f"FLOPs: {flops}")
                    print(f"Params: {params}")
            else:
                if flag is False:  # for OOD
                    for i in range(length):
                        for view, angle_y in enumerate(
                                np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs,
                                            self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                           np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius,
                                                                           device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)
                            img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
                else:  # for InD
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                       np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_u = generator.synthesis(w_u, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            generator.train()

        snapshot_data = dict()  # 初始化字典 snapshot_data，用于保存模型和数据
        snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        current_time = datetime.datetime.now()
        # 打印当前时间
        print("当前时间是:", current_time)
        with open(os.path.join(ckpt_dir, f"last.pkl"), "wb") as f:
            pickle.dump(snapshot_data, f)


    # 1. 使用高斯噪声实现差分隐私
    def add_gaussian_noise(self,feature_vector):
        """
        给特征向量添加高斯噪声以实现差分隐私。

        参数：
        - feature_vector: 输入的特征向量
        - epsilon: 隐私预算
        - sensitivity: 灵敏度
        - delta: 隐私参数
        """
        # 计算标准差sigma（根据隐私预算和灵敏度来决定噪声的强度）
        epsilon = 0.5  # 隐私预算（较小的epsilon提供较强的隐私保护）
        delta = 1e-5  # 适用于高斯噪声的隐私参数
        sensitivity = 0.001/64 * math.sqrt(2 * math.log(1.25/delta))/epsilon # 假设灵敏度为1.0（可能根据实际应用的不同而变化）
        sigma = (sensitivity/epsilon * math.sqrt(2 * math.log(1.25/delta)))/math.sqrt(5)
        noise = torch.normal(mean=0.0, std=sigma, size=feature_vector.shape, device=feature_vector.device)
        # 在特征向量中添加高斯噪声

        # 返回带噪声的特征向量
        return feature_vector + noise

    # 2. 使用拉普拉斯噪声实现差分隐私
    def add_laplace_noise(self,feature_vector, epsilon, sensitivity):
        """
        给特征向量添加拉普拉斯噪声以实现差分隐私。

        参数：
        - feature_vector: 输入的特征向量
        - epsilon: 隐私预算
        - sensitivity: 灵敏度
        """
        # 计算拉普拉斯分布的尺度（lambda）
        lambda_param = sensitivity / epsilon
        lambda_param = lambda_param * 0.5

        # 在特征向量中添加拉普拉斯噪声
        noise = torch.from_numpy(np.random.laplace(0.0, lambda_param, feature_vector.shape,device=feature_vector.device))

        # 返回带噪声的特征向量
        return feature_vector + noise

    # # 选择使用高斯噪声或拉普拉斯噪声
    # feature_with_gaussian_noise = add_gaussian_noise(feature_vector, epsilon, sensitivity, delta)
    # feature_with_laplace_noise = add_laplace_noise(feature_vector, epsilon, sensitivity)
    #
    # # 输出带噪声的特征向量
    # print("Original Feature Vector:\n", feature_vector)
    # print("Feature with Gaussian Noise:\n", feature_with_gaussian_noise)
    # print("Feature with Laplace Noise:\n", feature_with_laplace_noise)