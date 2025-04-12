import copy
import os
import click
import legacy
import dnnlib
import torch
import random
import pickle
import copy
import lpips
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from FL_server import Server
from torch import optim
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils.misc import copy_params_and_buffers
from arcface import IDLoss


import numpy as np
import torch
from torch import nn
from dataset import get_client_dataloader
from PIL import Image
import swanlab


class Client(object):
    def __init__(self, args, cid):
        torch.manual_seed(0)
        self.cid = cid
        self.args = args
        self.target_id = args.target_cid
        # self.epoch = args.local_epoch

        self.model = copy.deepcopy(args.model)
        self.learning_rate = args.lr
        self.device = args.device
        self.num_clients = args.num_clients
        self.device = args.device
        self.N_client = args.N_client

        self.encoder = None

    def _get_dataloader(self, client_id):
        """
        通过传入的 client_id 获取对应的数据加载器。
        Args:
            client_id (int): 客户端 ID，可能是普通的联邦学习客户端 ID，或者目标客户端 ID。
        """
        return get_client_dataloader(client_id)

    def train(self):
        # 初始化一个新的swanlab实验
        swanlab.init(
            project="my-first-ml",
            config={'learning-rate': 1e-4},
        )
        print(f"current client is（client.py）: {self.cid}")
        device = torch.device("cuda")
        print(f"Checkpoint path （client.py）: {self.args.pretrained_ckpt}")
        with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)

        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(False).to(device)  # 创建一个TriPlaneGenerator对象 这是一个生成器类 用于生成三维图像渲染 ，并加载预训练的生成器参数
        copy_params_and_buffers(init_global_generator, generator, require_all=True)  # 将预训练的生成器参数复制到新的生成器对象中
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution  # 设置生成器的渲染分辨率
        generator.rendering_kwargs = init_global_generator.rendering_kwargs  # 设置生成器的渲染参数
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)  # 加载生成器的状态字典
        generator.train().to(self.device)  # 将生成器设置为训练模式
        for param in generator.parameters():
            param.requires_grad = True
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
        # ./avg_latent_vector.pt    w_avg_ffhqrebalanced512-128.pt.unsqueeze(0)
        w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device)  # [1, 14, 512]
        w, flag, length = self.compute_latent_vectors(self.cid)
        target_idx = self.args.target_idx
        print(f"（client.py）target_idx is: {target_idx}")
        # w_origin = w + w_avg  # 为 w 加上均值向量 w_avg，生成原始潜在向量 w_origin
        w_gen = w[[target_idx], :, :] + w_avg  # 从编码器生成的潜在空间表示中选择一个特定索引的潜在向量，并加上均值向量 w_avg，生成反演后的潜在向量 w_u

        with torch.no_grad():
            w_id = w[[target_idx], :, :]
            w_target = w_id + w_avg





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
                feat_u = generator.get_planes(w_gen)  # 获取 generator 在输入 w_u 下生成的特征图。
                feat_target = init_global_generator.get_planes(w_target)  # 获取 g_source 在输入 w_target 下生成的特征图。
                loss_local_mse = F.mse_loss(feat_u, feat_target)
                loss_local = loss_local + self.args.loss_local_mse_lambda * loss_local_mse
                # 生成图像 img_u 和 img_target，并计算它们之间的感知损失（LPIPS）。
                img_u = generator.synthesis(w_gen, camera_params)["image"]
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
                loss_local_lpips = lpips_fn(img_u, img_target).mean()
                loss_local = loss_local + self.args.loss_local_lpips_lambda * loss_local_lpips
                # 计算生成图像 img_u 和目标图像 img_target 之间的身份损失（IDLoss）
                loss_local_id = id_fn(img_u, img_target)
                loss_local = loss_local + self.args.loss_local_id_lambda * loss_local_id
                loss = loss + loss_local
                loss_dict["loss_local"] = loss_local.item()

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
            swanlab.log({"train/loss": loss.item()})
            pbar.set_postfix(loss=loss.item(), **loss_dict)

            if i % 20 == 0:  # 每 100 次迭代保存一次生成的图像和目标图像
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_gen, camera_params_front)["image"]
                    img_u_save = self.tensor_to_image(img_u_save)
                    img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                    img_target_save = init_global_generator.synthesis(w_target, camera_params_front)["image"]
                    img_target_save = self.tensor_to_image(img_target_save)
                    img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                    generator.train()
                del img_u_save, img_target_save
            # 是在训练过程中定期保存当前生成的图像和目标图像

        # 深拷贝生成器并提取参数
        generator_copy = copy.deepcopy(generator).eval().requires_grad_(False).cpu()

        # 构造参数字典
        snapshot_data = {
            "G_ema": generator_copy.state_dict(),  # 关键修改：存储 state_dict 而非模型对象
        }

        # 释放资源
        del generator, generator_copy
        torch.cuda.empty_cache()

        return snapshot_data  # 返回字典
        # snapshot_data = dict()  # 初始化字典 snapshot_data，用于保存模型和数据
        # snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        #
        # # with open(os.path.join(ckpt_dir, f"finetune.pkl"), "wb") as f:
        # #     pickle.dump(snapshot_data, f)
        #
        # del generator
        # torch.cuda.empty_cache()
        # return snapshot_data

    def invert_image_to_latent(self,image_path, encoder, transform, device):
        img = self.image_to_tensor(Image.open(image_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            w, _ = encoder(img)  # 反演到潜在空间 w
        return w

    def image_to_tensor(self,i, size=256):
        i = i.resize((size, size))
        i = np.array(i)
        i = i.transpose(2, 0, 1)
        i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
        return i

    def convert_tensor(self,t):
        t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(t[0].cpu().numpy(), "RGB")

    def load_encoder(self,encoder_ckpt, device):
        from goae import GOAEncoder  # 假设反演编码器为 GOAE
        from goae.swin_config import get_config
        swin_config = get_config()  # 获取 Swin Transformer 的配置
        stage_list = [10000, 20000,
                      30000]  # 指定训练的 Swin Transformer 阶段 定义一个阶段列表，可能用于编码器的多阶段训练或推理过程。每个数字代表一个训练阶段的迭代次数
        encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(
            device)  # 使用配置参数实例化 GOAEncoder 编码器
        # 加载预训练的权重到编码器中，使其可以执行图像反演任务。
        encoder.load_state_dict(
            torch.load(encoder_ckpt, map_location=device))
        encoder.eval()
        return encoder

    def neggrad(self):
        swanlab.init(
            project="my-first-ml",
            config={'learning-rate': 1e-4},
        )
        print(f"current client is（client.py）: {self.cid}")
        device = torch.device("cuda")
        pretrained_ckpt = "XXXXXXXX/FU-GUIDE/experiments/ft_casia/checkpoints/finetune_final.pkl"
        # with dnnlib.util.open_url(self.args.pretrained_ckpt) as f:
        with dnnlib.util.open_url(pretrained_ckpt) as f:
            init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)

        generator = TriPlaneGenerator(*init_global_generator.init_args,
                                      **init_global_generator.init_kwargs).requires_grad_(False).to(
            device)  # 创建一个TriPlaneGenerator对象 这是一个生成器类 用于生成三维图像渲染 ，并加载预训练的生成器参数
        copy_params_and_buffers(init_global_generator, generator, require_all=True)  # 将预训练的生成器参数复制到新的生成器对象中
        generator.neural_rendering_resolution = init_global_generator.neural_rendering_resolution  # 设置生成器的渲染分辨率
        generator.rendering_kwargs = init_global_generator.rendering_kwargs  # 设置生成器的渲染参数
        generator.load_state_dict(init_global_generator.state_dict(), strict=False)  # 加载生成器的状态字典
        generator.train().to(self.device)  # 将生成器设置为训练模式
        for param in generator.parameters():
            param.requires_grad = True
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
        # optimizer = optim.Adam(generator.parameters(), lr=self.args.lr)
        lr = 0.05
        optimizer = torch.optim.SGD(generator.parameters(), lr=lr, momentum=0.9)

        # 定义图像转换
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 假设生成器支持 256x256 图片
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # 加载一个预训练的均值向量 w_avg，这通常是在生成对抗网络（GAN）中用来初始化或引导生成过程的中间潜在向量。
        w_avg = torch.load("./files/w_avg_ffhqrebalanced512-128.pt", map_location=device)  # [1, 14, 512]
        # w, flag, length = self.compute_latent_vectors(self.cid)
        image_path = "XXXXXXXX/FU-GUIDE/Deep3DFaceRecon/checkpoints/model_name/casia/066/093.png"
        encoder = self.load_encoder(self.args.encoder_ckpt, device)
        w_encoder = self.invert_image_to_latent(image_path, encoder, transform, device)
        w_gen = w_encoder + w_avg
        w_target = w_gen


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
                loss = torch.tensor(0.0, device=device)
                # feat_u = generator.get_planes(w_gen)  # 获取 generator 在输入 w_u 下生成的特征图。
                # feat_target = init_global_generator.get_planes(w_target)  # 获取 g_source 在输入 w_target 下生成的特征图。
                # loss_local_mse = F.mse_loss(feat_u, feat_target)
                # loss= loss + 0.1* loss_local_mse


                img_u = generator.synthesis(w_gen, camera_params)["image"]
                img_target = init_global_generator.synthesis(w_target, camera_params)["image"]
                loss_local_mse = F.mse_loss(img_u, img_target)
                loss1 = loss + loss_local_mse

                loss_local_id = id_fn(img_u, img_target)
                loss = loss1 + loss_local_id
                # 是在训练后或推理时，根据不同的视角生成并保存图像


            # 通过反向传播计算梯度，并使用优化器更新生成器的参数
            optimizer.zero_grad()
            loss_joint = -loss
            loss_joint.backward()  # 通过反向传播计算损失 loss 对生成器参数的梯度
            optimizer.step()
            # swanlab.log({"train/loss": loss.item()})
            # pbar.set_postfix(loss=loss.item(), **loss_dict)
            # 更新进度条并保存中间结果
            # # 修改后的两行代码
            swanlab.log({"train/loss": loss.item(), "train/joint_loss": loss_joint.item()})  # 同时记录两个损失
            pbar.set_postfix(loss=loss.item(), joint_loss=loss_joint.item())  # 直接传递数值，无需解包字典

            if i % 1 == 0:  # 每 100 次迭代保存一次生成的图像和目标图像
                with torch.no_grad():
                    generator.eval()
                    img_u_save = generator.synthesis(w_gen, camera_params_front)["image"]
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
            img_u_save = generator.synthesis(w_gen, camera_params)["image"]
            img_target_save = init_global_generator.synthesis(w_target, camera_params)["image"]
            img_u_save = self.tensor_to_image(img_u_save)
            img_target_save = self.tensor_to_image(img_target_save)
            img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
            img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))

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
                    img_u = generator.synthesis(w_gen, camera_params_view)["image"]
                    img_u = self.tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
            else: # for OOD
                    for i in range(length):
                        for view, angle_y in enumerate(
                                np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                            cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                           np.pi / 2 + self.args.angle_p,
                                                                           cam_pivot, radius=cam_radius,
                                                                           device=device)
                            camera_params_view = torch.cat(
                                [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                                dim=1)
                            img_origin = generator.synthesis(w_gen[[i]], camera_params_view)["image"]
                            img_origin = self.tensor_to_image(img_origin)
                            img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
                  # for InD
                    for view, angle_y in enumerate(
                            np.linspace(-self.args.angle_y_abs, self.args.angle_y_abs, self.args.sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y,
                                                                       np.pi / 2 + self.args.angle_p,
                                                                       cam_pivot,
                                                                       radius=cam_radius, device=device)
                        camera_params_view = torch.cat(
                            [cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)],
                            dim=1)
                        img_u = generator.synthesis(w_gen, camera_params_view)["image"]
                        img_u = self.tensor_to_image(img_u)
                        img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))

        # 深拷贝生成器并提取参数
        generator_copy = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        # 构造参数字典
        snapshot_data = {
            "G_ema": generator_copy.state_dict(),  # 关键修改：存储 state_dict 而非模型对象
        }
        # 释放资源
        del generator, generator_copy
        torch.cuda.empty_cache()

        return snapshot_data  # 返回字典
        # snapshot_data = dict()  # 初始化字典 snapshot_data，用于保存模型和数据
        # snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
        #
        # # with open(os.path.join(ckpt_dir, f"finetune.pkl"), "wb") as f:
        # #     pickle.dump(snapshot_data, f)
        #
        # del generator
        # torch.cuda.empty_cache()
        # return snapshot_data
    def tensor_to_image(self, t):
        t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(t[0].cpu().numpy(), "RGB")

    def image_to_tensor(self, i, size=256):
        i = i.resize((size, size))
        i = np.array(i)
        i = i.transpose(2, 0, 1)
        i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
        return i

    def upload_encoder(self, encoder):
        """
        上传编码器，从服务器端接受编码器并在客户端使用。
        Args:
            encoder (nn.Module): 服务器端下发的编码器模型。
        """
        self.encoder = copy.deepcopy(encoder).to(self.device)

    def compute_latent_vectors(self, target_cid):
        """
        计算潜在空间表示 w
        """

        dataloader, length = self._get_dataloader(target_cid)
        self.encoder.to(self.device)  # 将编码器模型移动到设备上（GPU 或 CPU）
        self.encoder.eval()  # 将编码器设置为评估模式
        print("dataloader 的长度 ", length)
        flag = False
        all_w = []
        with torch.no_grad():
            # 遍历数据加载器中的所有批次
            for batch_idx, (batch_imgs, _) in enumerate(dataloader):
                batch_imgs = batch_imgs.to(self.device)
                batch_w, _ = self.encoder(batch_imgs)
                all_w.append(batch_w)

            all_w = torch.cat(all_w, dim=0)
            return all_w, flag, length  # 返回潜在向量和多张图像的标志