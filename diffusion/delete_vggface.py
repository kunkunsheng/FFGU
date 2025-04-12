import inspect
import logging
import math
import os
import shutil
import datetime
from pathlib import Path
import accelerate
from accelerate.utils import set_seed
import transformers
import datasets
import torch
from torch.utils.data import TensorDataset, ConcatDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
from PIL import Image
from torch import linalg

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import (
    check_min_version,
    is_accelerate_version,
    is_tensorboard_available,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from evaluate import Evaluator

from main import Task
from losses.ddpm_deletion_loss import DDPMDeletionLoss
from data.utils.infinite_sampler import InfiniteSampler
from data.utils.repeat_sampler import RepeatedSampler
from data.src.celeb_dataset import CelebAHQ

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)    

class DeleteCeleb(Task):
    def __init__(
        self,
        cfg: DictConfig
    ):
        self.cfg = cfg        

    def run(self):
        # Logging
        logging_dir = os.path.join(self.cfg.output_dir, self.cfg.logging.logging_dir)

        if self.cfg.logging.logger == "tensorboard":
            if not is_tensorboard_available():
                raise ImportError(
                    "Make sure to install tensorboard if you want to use it for logging during training."
                )
        elif self.cfg.logging.logger == "wandb":
            if not is_wandb_available():
                raise ImportError(
                    "Make sure to install wandb if you want to use it for logging during training."
                )
            import wandb

        # Set up accelerator
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.cfg.output_dir, logging_dir=logging_dir
        )
        kwargs = InitProcessGroupKwargs(
            timeout=datetime.timedelta(seconds=7200)
        )  # a big number for high resolution or big dataset
        accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=self.cfg.mixed_precision,
            log_with=self.cfg.logging.logger,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            wandb_conf = OmegaConf.to_container(self.cfg) # fully converts to dict (even nested keys)
            accelerator.init_trackers(project_name=self.cfg.project_name, config=wandb_conf)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.cfg.random_seed is not None:
            set_seed(self.cfg.random_seed)

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if self.cfg.ema.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                
                # breakpoint()
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        # accelerator.register_load_state_pre_hook(load_model_hook)

        # Instantiate unet
        # if self.cfg.unet._type == "initialize":
        #     unet = hydra.utils.instantiate(self.cfg.unet, _convert_="all")
        # elif self.cfg.unet._type == "pretrained":
        print(f'Loading checkpoint {self.cfg.checkpoint_path}')
        ddpm_pipeline = DDPMPipeline.from_pretrained(self.cfg.checkpoint_path)
        unet = ddpm_pipeline.unet

        # Create EMA for the model.
        if self.cfg.ema.use_ema:

            ema_model = EMAModel.from_pretrained(
                os.path.join(self.cfg.checkpoint_path, self.cfg.subfolders.unet_ema),
                hydra.utils.get_class(self.cfg.unet._target_),
            )


        noise_scheduler = ddpm_pipeline.scheduler #扩散模型的时间步调度器 定义时间步的去噪策略
        print("0000000000")
        # Instantiate optimizer
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, unet.parameters())

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            self.cfg.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            self.cfg.mixed_precision = accelerator.mixed_precision
        print("11111111")
        # ----- END SHARED CODE -----
        def prepare_dataloaders(self, noise_scheduler, accelerator, weight_dtype):
            """
                处理所有客户端的数据，返回训练所需的 DataLoader 和干净图像 Tensor。

                返回:
                    noisy_all_dataloader: DataLoader - 所有客户端的加噪数据
                    noisy_deletion_dataloader: DataLoader - 删除客户端的加噪数据
                    all_imgs: Tensor - 所有客户端的干净图片 (4, C, H, W)
                    deletion_imgs: Tensor - 删除客户端的干净图片 (4, C, H, W)
                """
            transform = hydra.utils.instantiate(self.cfg.transform)
            bsz = 4  # Batch size
            # 验证 deletion loss 相关配置
            if self.cfg.deletion.timestep_del_window is None and self.cfg.deletion.loss_fn == "modified_noise_obj":
                raise ValueError(
                    'Timestep deletion window length must be provided for the modified_noise_obj loss function.'
                )
            start_timestep = (noise_scheduler.config.num_train_timesteps - self.cfg.deletion.timestep_del_window
                      if self.cfg.deletion.loss_fn == "modified_noise_obj" else 0)

            # 存储所有客户端数据集
            noisy_datasets = []
            deletion_datasets = []
            all_imgs_list = []
            deletion_imgs_list = []
            all_noises_list = []  # 统一的噪声
            all_timesteps_list = []  # 统一的时间步
            for client_id in range(self.cfg.federated.num_clients):
                client_id_str = f"{client_id:03d}"  # 例如 "000", "001", ..., "127"
                print(f"Processing client {client_id_str}")
                client_data_path = os.path.join(self.cfg.data_dir, client_id_str)

                # 获取删除客户端 ID 列表
                deletion_client_ids = [str(cid).zfill(3) for cid in self.cfg.deletion.client_id]
                if client_id_str in deletion_client_ids:
                    print(f"Skipping client {client_id_str} (deletion client)")

                    # 处理删除客户端数据
                    dataset_deletion = CelebAHQ(
                        filter="deletion",
                        data_path=client_data_path,
                        remove_img_names=self.cfg.deletion.img_name,
                        transform=transform
                    )

                    image = dataset_deletion[0]  # 取一张图片
                    deletion_imgs = torch.stack([image] * bsz).to(device=accelerator.device, dtype=weight_dtype)
                    if not all_noises_list:
                        noise = torch.randn(deletion_imgs.shape, dtype=weight_dtype, device=deletion_imgs.device)
                        all_noises_list.append(noise)
                    else:
                        noise = all_noises_list[0]
                        # **如果 all_timesteps_list 为空，生成 timesteps 并存入**
                    if not all_timesteps_list:
                        timesteps_client = torch.randint(
                            start_timestep,
                            noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=deletion_imgs.device
                        ).long()
                        all_timesteps_list.append(timesteps_client)
                    else:
                        timesteps_client = all_timesteps_list[0]

                    # 生成加噪后的图片
                    noisy_deletion_images = noise_scheduler.add_noise(deletion_imgs, noise, timesteps_client)

                    # 存储 `TensorDataset‘
                    deletion_datasets.append(TensorDataset(noisy_deletion_images))

                    continue  # 跳过当前客户端，不执行 `dataset_all`
                    # 处理非删除客户端数据
                dataset_all = CelebAHQ(
                    filter="all",
                    data_path=client_data_path,
                    remove_img_names="null",
                    transform=transform
                )

                dataloader = DataLoader(dataset_all, batch_size=bsz, shuffle=False)
                all_imgs = []
                # 提取所有图片
                for batch in dataloader:
                    all_imgs.append(batch)

                # 合并所有 batch，并转换格式
                all_imgs = torch.cat(all_imgs, dim=0).to(device=accelerator.device, dtype=weight_dtype)
                num = all_imgs.shape[0]
                # **如果 all_noises_list 为空，生成 noise 并存入**
                if not all_noises_list:
                    noise = torch.randn(all_imgs.shape, dtype=weight_dtype, device=all_imgs.device)
                    all_noises_list.append(noise)
                else:
                    noise = all_noises_list[0]  # **使用第一次生成的 noise**
                noise = noise.tile((num // noise.shape[0] + 1, 1, 1, 1))[:num]
                print(f"第281行 noise.shape[0] 是 {noise.shape[0]}")
                # **如果 all_timesteps_list 为空，生成 timesteps 并存入**
                if not all_timesteps_list:
                    timesteps_client = torch.randint(
                        start_timestep,
                        noise_scheduler.config.num_train_timesteps,
                        (all_imgs.shape[0],),
                        device=all_imgs.device
                    ).long()
                    all_timesteps_list.append(timesteps_client)
                else:
                    timesteps_client = all_timesteps_list[0]

                
                print(f"第267行 all_imgs.shape[0] 是 {all_imgs.shape[0]}")
                if timesteps_client.shape[0] < all_imgs.shape[0]:
                    extra_timesteps = torch.randint(
                        start_timestep,
                        noise_scheduler.config.num_train_timesteps,
                        (all_imgs.shape[0] - timesteps_client.shape[0],),
                        device=timesteps_client.device
                    ).long()
                    timesteps_client = torch.cat([timesteps_client, extra_timesteps], dim=0)

                print(timesteps_client.shape)  # 确保是 (batch_size,)

                # 生成加噪后的图片
                noisy_images = noise_scheduler.add_noise(all_imgs, noise, timesteps_client)

                # 存储 `TensorDataset`
                noisy_datasets.append(TensorDataset(noisy_images))
                all_imgs_list.append(all_imgs)

            # **合并所有客户端的数据集**
            noisy_all_dataset = ConcatDataset(noisy_datasets)
            deletion_all_dataset = ConcatDataset(deletion_datasets) if deletion_datasets else None

            # **创建合并后的 DataLoader**
            noisy_all_dataloader = DataLoader(
                noisy_all_dataset, batch_size=bsz, num_workers=self.cfg.dataloader_num_workers, shuffle=False
            )

            noisy_deletion_dataloader = None
            if deletion_all_dataset:
                noisy_deletion_dataloader = DataLoader(
                    deletion_all_dataset, batch_size=bsz, num_workers=self.cfg.dataloader_num_workers, shuffle=False
                )

            # **合并所有 batch**
            all_imgs = torch.cat(all_imgs_list, dim=0) if all_imgs_list else None
            deletion_imgs = torch.cat(deletion_imgs_list, dim=0) if deletion_imgs_list else None
            noise = all_noises_list[0] if all_noises_list else None  # 只取第一次生成的 noise
            timesteps = all_timesteps_list[0] if all_timesteps_list else None
            print(f"第327行 all_imgs.shape[0] 是 {all_imgs.shape[0]}")
            print(f"第328行 deletion_imgs.shape[0] 是 {deletion_imgs.shape[0]}")
            print(f"第329行 noise.shape[0] 是 {noise.shape[0]}")
            print(f"第330行 timesteps.shape[0] 是 {timesteps.shape[0]}")

            return noisy_all_dataloader, noisy_deletion_dataloader, all_imgs, deletion_imgs, noise, timesteps

        # # Preprocessing the datasets and DataLoaders creation.
        # transform = hydra.utils.instantiate(self.cfg.transform)
        # # 遍历每个客户端 (000 - 127)
        # print("3333333")
        # bsz = 4
        # if self.cfg.deletion.timestep_del_window is None and self.cfg.deletion.loss_fn == "modified_noise_obj":
        #     raise ValueError(
        #         'Timestep deletion window length must be provided for the modified_noise_obj loss function.')
        # start_timestep = noise_scheduler.config.num_train_timesteps - self.cfg.deletion.timestep_del_window if self.cfg.deletion.loss_fn == "modified_noise_obj" else 0
        # noisy_datasets = []
        # for client_id in range(self.cfg.federated.num_clients):
        #     client_id_str = f"{client_id:03d}"  # 转换成三位数格式，如 000, 001, ..., 127
        #     print(client_id_str)
        #     client_data_path = os.path.join(self.cfg.data_dir, client_id_str)
        #
        #     deletion_client_ids = [str(client_id).zfill(3) for client_id in self.cfg.deletion.client_id]
        #     if client_id_str in deletion_client_ids:
        #         print(f"Skipping client {client_id_str} (unlearn_id)")
        #         dataset_deletion = CelebAHQ(
        #             filter="deletion",
        #             data_path=client_data_path,
        #             remove_img_names= self.cfg.deletion.img_name, #["10000.jpg", "10002.jpg"]
        #             transform=transform
        #         )
        #         image = dataset_deletion[0]
        #         deletion_images= torch.stack([image] * 4)
        #         deletion_images = deletion_images.to(device=accelerator.device, dtype=weight_dtype)
        #
        #         noise = torch.randn(deletion_images.shape, dtype=weight_dtype, device= deletion_images.device)
        #         timesteps_client = torch.randint(
        #             start_timestep,
        #             noise_scheduler.config.num_train_timesteps,
        #             (bsz,),
        #             device=deletion_images.device
        #         ).long()
        #
        #         noisy_deletion_images = noise_scheduler.add_noise(deletion_images, noise, timesteps_client)
        #         dataset = TensorDataset(noisy_deletion_images)  # 将 Tensor 转换为 Dataset
        #         noisy_deletion_dataloader = torch.utils.data.DataLoader(
        #             dataset, batch_size=4, num_workers=self.cfg.dataloader_num_workers, shuffle=False
        #         )
        #         continue
        #
        #     dataset_all = CelebAHQ(
        #         filter="all",
        #         data_path=client_data_path,
        #         remove_img_names= "null",
        #         transform=transform
        #     )
        #     dataloader = DataLoader(dataset_all, batch_size=4, shuffle=False)
        #     all_imgs = []
        #
        #     # 遍历 DataLoader，提取所有图片
        #     for batch in dataloader:
        #         all_imgs.append(batch)  # 每个 batch 是 (batch_size, C, H, W)s
        #
        #     # 合并所有 batch，形成 (总样本数, C, H, W) 的 Tensor
        #     all_imgs = torch.cat(all_imgs, dim=0).to(device=accelerator.device, dtype=weight_dtype)
        #     noise = torch.randn(all_imgs.shape, dtype=weight_dtype, device=all_imgs.device)
        #     timesteps_client = torch.randint(
        #         start_timestep,
        #         noise_scheduler.config.num_train_timesteps,
        #         (all_imgs.shape[0],),
        #         device=all_imgs.device
        #     ).long()
        #     # 6. 生成加噪后的图片
        #     noisy_images = noise_scheduler.add_noise(all_imgs, noise, timesteps_client)
        #     noisy_dataset = TensorDataset(noisy_images)
        #
        #     # 8. 存储当前客户端的数据集
        #     noisy_datasets.append(noisy_dataset)
        #
        # # **合并所有客户端的数据集**
        # noisy_all_dataset = ConcatDataset(noisy_datasets)
        #
        # # **创建合并后的 DataLoader**
        # noisy_all_dataloader = DataLoader(
        #     noisy_all_dataset, batch_size=4, num_workers=self.cfg.dataloader_num_workers, shuffle=False
        # )
        # 初始化调度器的学习率 Initialize the learning rate scheduler
        lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.cfg.training_steps
        )
        noisy_all_dataloader, noisy_deletion_dataloader, all_imgs, deletion_imgs, noise, timesteps= prepare_dataloaders(self,
                                                                                                       noise_scheduler,
                                                                                                       accelerator,
                                                                                                       weight_dtype)

        print("All images shape:", all_imgs.shape if all_imgs is not None else "None")
        print("Deletion images shape:", deletion_imgs.shape if deletion_imgs is not None else "None")
        print(f"接收到 noise 是 {noise}")
        print(f"接收到 timesteps 是 {timesteps}")

        # Prepare everything with our `accelerator` (except optimizer and lr_scheduler)
        unet, optimizer, lr_scheduler, noisy_all_dataloader, noisy_deletion_dataloader = accelerator.prepare(
            unet, optimizer, lr_scheduler, noisy_all_dataloader, noisy_deletion_dataloader
        )

        if self.cfg.ema.use_ema:
            ema_model.to(accelerator.device)

        total_batch_size = (
            self.cfg.train_batch_size
            * accelerator.num_processes
            * self.cfg.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        # logger.info(f"Num all examples: {len(dataset_all)}")
        # logger.info(f"Num deletion examples: {len(dataset_deletion)}")
        logger.info(f"  Num training steps = {self.cfg.training_steps}")
        logger.info(
            f"  Instantaneous batch size per device = {self.cfg.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.cfg.gradient_accumulation_steps}"
        )
        # logger.info(f"  Total optimization steps = {max_train_steps}")

        global_step = 0
        # first_epoch = 0

        gamma = noise_scheduler.alphas_cumprod ** 0.5
        gamma = gamma.to(accelerator.device)

        sigma = (1 - noise_scheduler.alphas_cumprod) ** 0.5
        sigma = sigma.to(accelerator.device)

        loss_class = DDPMDeletionLoss(gamma=gamma, sigma=sigma) #, frac_deletion=self.cfg.deletion.frac_deletion)
        loss_fn = getattr(loss_class, self.cfg.deletion.loss_fn)

        def evaluate_model(unet, num_imgs_to_generate=self.cfg.eval_batch_size, batch_size=self.cfg.eval_batch_size, set_generator=False):

            print("evaluate_model")
            unet.eval()
            unet = accelerator.unwrap_model(unet)

            if self.cfg.ema.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            evaluator = Evaluator(self.cfg)
            evaluator.load_model(unet, noise_scheduler) #加载模型和扩散调度器 调度器决定了“扩散模型如何去噪生成图像”
            
            generated_images = []
            remaining_images = num_imgs_to_generate

            while remaining_images > 0:
                current_batch_size = min(batch_size, remaining_images)
                images = evaluator.sample_images(current_batch_size, set_generator=set_generator)
                generated_images.append(images)
                remaining_images -= current_batch_size
            
            generated_images = np.concatenate(generated_images, axis=0)

            if self.cfg.ema.use_ema:
                ema_model.restore(unet.parameters()) #恢复unet原始参数

            unet.train()
            return generated_images
        #在特定时间步 timestep 评估 UNet 的去噪能力
        def evaluate_unlearning_timestep(unet, timestep, clean_image, num_imgs_to_generate=self.cfg.eval_batch_size, batch_size=self.cfg.eval_batch_size):
            print("evaluate_unlearning_timestep")
            unet.eval()
            unet = accelerator.unwrap_model(unet)

            if self.cfg.ema.use_ema:
                ema_model.store(unet.parameters())
                ema_model.copy_to(unet.parameters())

            evaluator = Evaluator(self.cfg)
            evaluator.load_model(unet, noise_scheduler)
            
            generated_images = []
            remaining_images = num_imgs_to_generate
            generator = torch.Generator(device=accelerator.device).manual_seed(self.cfg.random_seed)

            while remaining_images > 0:
                current_batch_size = min(batch_size, remaining_images)

                noise = torch.randn((current_batch_size, *clean_image.shape), generator=generator, device=accelerator.device)
                timestep_expanded = torch.full((current_batch_size,), timestep, device=accelerator.device)
                noisy_image_batch = noise_scheduler.add_noise(clean_image, noise, timestep_expanded)
                images = evaluator.denoise_images(noisy_image_batch, timestep)
                # breakpoint()
                generated_images.append(images)
                remaining_images -= current_batch_size
            
            generated_images = np.concatenate([image for image in generated_images], axis=0)

            if self.cfg.ema.use_ema:
                ema_model.restore(unet.parameters())

            unet.train()
            return generated_images
        #计算成员损失 在不同时间步计算loss 观察unet遗忘能力 绘制mloss曲线 可视化删除数据的恢复难度 如果mloss低说明数据仍能恢复

        if self.cfg.metrics.membership_loss: 
            membership_class = hydra.utils.instantiate(self.cfg.metrics.membership_loss.class_cfg, dataset_all=dataset_all, dataset_deletion=dataset_deletion, noise_scheduler=noise_scheduler, unet=unet, device=accelerator.device)
            membership_class.sample_images()
            membership_class.sample_noises()

            plot_params = self.cfg.metrics.membership_loss.plot_params
            if plot_params is not None:
                time_freq = plot_params.time_frequency
                timesteps = list(range(0, self.cfg.scheduler.num_train_timesteps, time_freq))
                losses = membership_class.compute_membership_losses(timesteps)
                all_membership_losses = [loss[0] for loss in losses]

                # Create plot
                fig, ax = plt.subplots()
                ax.plot(timesteps, all_membership_losses)

                # Set the labels and title
                ax.set_xlabel('Timestep')
                ax.set_ylabel('All membership loss')
                ax.set_title('Loss over time')

                # Save the plot to a file
                plt.savefig(plot_params.save_path)

                exit()

             #用于计算分类准确率、Inception Score（IS）
        if self.cfg.metrics.classifier_cfg:
            classifier = hydra.utils.instantiate(self.cfg.metrics.classifier_cfg, device=accelerator.device)
        
        if self.cfg.metrics.fid:
            fid_calculator = hydra.utils.instantiate(self.cfg.metrics.fid.class_cfg, device=accelerator.device)
            fid_calculator.load_celeb()
        #测试特定噪声时间步 t 下模型的恢复能力。


        #遗忘的目标图片在时间步250步加噪并还原
        if self.cfg.metrics.denoising_injections:
            injection_timestep = torch.tensor(self.cfg.metrics.denoising_injections.timestep, device=accelerator.device)
            target_image = transform(Image.open(self.cfg.metrics.denoising_injections.img_path)).to(accelerator.device)
        #似然计算（Likelihood Estimation）衡量生成数据的概率密度。
        if self.cfg.metrics.likelihood:
            likelihood_calculator = hydra.utils.instantiate(self.cfg.metrics.likelihood.class_cfg)

        deletion_tracker = {
            'deletion_reached': False,
            'deletion_steps': None
        }
        def log_metrics(unet):
            print("log_metrics")
            wandb_batch = {}
            if global_step % self.cfg.sampling_steps == 0:
                images = evaluate_model(unet, set_generator=True) # fix randomness 
                grid = Evaluator.make_grid_from_images(images)
                wandb_batch["Sampled Images"] = wandb.Image(grid)

                images = torch.from_numpy(images).permute(0, 3, 1, 2).to(accelerator.device) # convert to (N, C, H, W) format
                if self.cfg.metrics.fraction_deletion:
                    frac_deletion = classifier.compute_class_frequency(images, self.cfg.deletion.class_label) 
                    wandb_batch["metrics/deletion_class_fraction"] = frac_deletion
                    if frac_deletion == 0 and not deletion_tracker['deletion_reached']:
                        wandb.run.summary['deletion_steps'] = global_step
                        deletion_tracker['deletion_steps'] = global_step
                        deletion_tracker['deletion_reached'] = True
                
                if self.cfg.metrics.denoising_injections:
                    targeted_image_generations = evaluate_unlearning_timestep(unet, injection_timestep, target_image)
                    target_image_grid = Evaluator.make_grid_from_images(targeted_image_generations)
                    wandb_batch[f"Target Image Generations (t={injection_timestep})"] = wandb.Image(target_image_grid)
                
                if self.cfg.metrics.likelihood and (global_step % self.cfg.metrics.likelihood.step_frequency == 0):
                    print('Computing likelihood...')
                    likelihood = likelihood_calculator.evaluate_likelihood(unet, target_image.unsqueeze(0))[0][0].item()
                    wandb_batch["metrics/likelihood"] = likelihood
                    print(f'Likelihood: {likelihood}')
            
                
            if self.cfg.metrics.membership_loss and (global_step % self.cfg.metrics.membership_loss.step_frequency == 0):
                timesteps = self.cfg.metrics.membership_loss.timesteps
                membership_losses = membership_class.compute_membership_losses(timesteps)
                for idx, timestep in enumerate(timesteps):
                    wandb_batch[f"membership_loss/all_membership_loss_t={timestep}"] = membership_losses[idx][0]
                    wandb_batch[f"membership_loss/deletion_membership_loss_t={timestep}"] = membership_losses[idx][1]
                    wandb_batch[f"membership_loss/membership_ratio_t={timestep}"] = membership_losses[idx][1] / membership_losses[idx][0]

            if self.cfg.metrics.inception_score and (global_step % self.cfg.metrics.inception_score.step_frequency == 0 or deletion_tracker['deletion_steps'] == global_step):
                images = evaluate_model(unet, num_imgs_to_generate=self.cfg.metrics.inception_score.num_imgs_to_generate, batch_size=self.cfg.metrics.inception_score.batch_size)
                images = torch.from_numpy(images).permute(0, 3, 1, 2).to(accelerator.device) # convert to (N, C, H, W) format

                inception_calculator = hydra.utils.instantiate(self.cfg.metrics.inception_score.class_cfg, classifier=classifier)
                inception_calculator.update(images)
                is_mean, is_std = inception_calculator.compute()
                wandb_batch['metrics/is_mean'] = is_mean.item()
                wandb_batch['metrics/is_std'] = is_std.item()
                print(f'IS_mean: {is_mean}')
                print(f'IS_std: {is_std}')
            
            if self.cfg.metrics.fid:
                should_compute_fid = self.cfg.metrics.fid.step_frequency and (global_step % self.cfg.metrics.fid.step_frequency == 0)
                if should_compute_fid or deletion_tracker['deletion_steps'] == global_step:
                    print('Computing FID...')
                    images = evaluate_model(unet, num_imgs_to_generate=self.cfg.metrics.fid.num_imgs_to_generate, batch_size=self.cfg.metrics.fid.batch_size)
                    images = torch.from_numpy(images).permute(0, 3, 1, 2).to(accelerator.device) # convert to (N, C, H, W) format

                    fid_calculator.add_fake_images(images)
                    fid = fid_calculator.compute(verbose=True)
                    wandb_batch['metrics/fid'] = fid.item()
                    print(f'FID: {fid}')

            if self.cfg.logging.logger == "wandb":
                accelerator.get_tracker("wandb").log(wandb_batch, step=global_step)

        # Train!
        progress_bar = tqdm(
            total=self.cfg.training_steps * len(self.cfg.deletion.img_name),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Steps")
        log_metrics(unet)

        accum_loss_x = {}   #是目标的
        accum_loss_a = {}  #是保留的数据集
        # 新增：用于汇总每个客户端的数据
        print("train!!!")

        #服务器端

        all_noisy_images = []
        deletion_noisy_images = []
        # 遍历 DataLoader，提取所有加噪数据
        for batch in noisy_all_dataloader:
            all_noisy_images.append(batch[0])  # batch 是 (batch_size, C, H, W)
        # 合并所有批次，形成完整的 Tensor
        noisy_all_images = torch.cat(all_noisy_images, dim=0)  # 形状变为 (N, C, H, W)
        # 存入字典
        all_samples_dict = {
            "noisy_latents": noisy_all_images,
        }
        for batch in noisy_deletion_dataloader:
            deletion_noisy_images.append(batch[0])
        noisy_deletion_images = torch.cat(deletion_noisy_images, dim=0)  # 形状变为 (N, C, H, W)
        # 存入字典
        deletion_samples_dict = {
            "noisy_latents": noisy_deletion_images
        }


        while global_step < self.cfg.training_steps * len(self.cfg.deletion.img_name):
            unet.train()
            target_image = deletion_imgs[0].squeeze()

            assert noisy_deletion_images.shape == noisy_all_images.shape
            print(f"传入loss的 noise 是 {noise}")
            print(f"传入loss的 timesteps 是 {timesteps}")
            with accelerator.accumulate(unet):
                # Predict the noise residual
                # model_output = unet(noisy_all_images, timesteps).sample

                if self.cfg.scheduler.prediction_type == "epsilon": #控制模型的输出类型，这里选择 epsilon (预测噪声)

                    items = loss_fn(unet, timesteps, noise, {}, all_samples_dict, deletion_samples_dict, **self.cfg.deletion.loss_params)
                    loss, loss_x, loss_a, importance_weight_x, importance_weight_a, weighted_loss_x, weighted_loss_a = (
                        items
                    )
                    batch_stats = {}
                    if loss is not None:
                        print("loss:", loss.mean().item())
                        batch_stats["loss/mean"] = loss.mean().item()
                        batch_stats["loss/max"] = loss.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss/min"] = loss.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss/std"] = loss.mean(dim=[1, 2, 3]).std().item()

                    if loss_x is not None:
                        batch_stats["loss_x/mean"] = loss_x.mean().item()
                        batch_stats["loss_x/max"] = loss_x.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss_x/min"] = loss_x.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss_x/std"] = loss_x.mean(dim=[1, 2, 3]).std().item()
                    
                    if loss_a is not None:
                        batch_stats["loss_a/mean"] = loss_a.mean().item()
                        batch_stats["loss_a/max"] = loss_a.mean(dim=[1, 2, 3]).max().item()
                        batch_stats["loss_a/min"] = loss_a.mean(dim=[1, 2, 3]).min().item()
                        batch_stats["loss_a/std"] = loss_a.mean(dim=[1, 2, 3]).std().item()
                    
                    if importance_weight_x is not None:
                        batch_stats["importance_weight_x/mean"] = importance_weight_x.mean().item()
                        batch_stats["importance_weight_x/max"] = importance_weight_x.max().item()
                        batch_stats["importance_weight_x/min"] = importance_weight_x.min().item()
                        batch_stats["importance_weight_x/std"] = importance_weight_x.std().item()

                    if importance_weight_a is not None:
                        batch_stats["importance_weight_a/mean"] = importance_weight_a.mean().item()
                        batch_stats["importance_weight_a/max"] = importance_weight_a.max().item()
                        batch_stats["importance_weight_a/min"] = importance_weight_a.min().item()
                        batch_stats["importance_weight_a/std"] = importance_weight_a.std().item()

                    if "superfactor" in self.cfg.deletion.loss_params:
                        batch_stats["superfactor"] = self.cfg.deletion.loss_params.superfactor
                        decay = self.cfg.deletion.superfactor_decay
                        if decay is not None:
                            self.cfg.deletion.loss_params.superfactor *= self.cfg.deletion.superfactor_decay
                    wandb.log(batch_stats, step=global_step)
                # elif self.cfg.scheduler.prediction_type == "sample":
                #     alpha_t = _extract_into_tensor(
                #         noise_scheduler.alphas_cumprod,
                #         timesteps,
                #         (all_images.shape[0], 1, 1, 1),
                #     )
                #     snr_weights = alpha_t / (1 - alpha_t)
                #     # use SNR weighting from distillation paper
                #     print(" 用到了model_output")
                #     loss = snr_weights * F.mse_loss(
                #         model_output.float(), all_images.float(), reduction="none"
                #     )
                #     loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {self.cfg.scheduler.prediction_type}"
                    )
                
                # breakpoint()
                if loss is not None:
                    loss = loss.sum() / self.cfg.train_batch_size
                    accelerator.backward(loss)
                else:
                    weighted_loss_x = weighted_loss_x.sum() / self.cfg.train_batch_size
                    weighted_loss_a = weighted_loss_a.sum() / self.cfg.train_batch_size

                    # only keep graph in case of SISS since double forward and erasediff dont mix noise_preds between loss_x and loss_a
                    retain_graph = self.cfg.deletion.loss_fn == "importance_sampling_with_mixture"
                    accelerator.backward(weighted_loss_x, retain_graph=retain_graph)

                    gradients_loss_x = {}
                    for name, param in unet.named_parameters():
                        # if param.grad is not None:
                        gradients_loss_x[name] = param.grad.clone()
                        # if name not in accum_loss_x:
                        #     accum_loss_x[name] = param.grad.clone()  # First time, initialize with clone of the gradient
                        # else:
                        #     accum_loss_x[name] += param.grad.clone()  # Add gradients for manual accumulation

                    accelerator.backward(weighted_loss_a)

                    # gradients_loss_a = {}
                    for name, param in unet.named_parameters():
                        # if param.grad is not None:
                        true_grad = param.grad.clone() - gradients_loss_x[name]
                        if name not in accum_loss_a:
                            accum_loss_a[name] = true_grad  # First time, initialize with clone of the gradient
                        else:
                            accum_loss_a[name] += true_grad  # Add gradients for manual accumulation
                    print('accumulating')
                
                if accelerator.sync_gradients:
                    print('prepare to step!')
                    if loss is None:
                        for name, param in unet.named_parameters():
                            accum_loss_x[name] = param.grad.clone() - accum_loss_a[name]

                        # Initialize variables to hold the sum of squared norms for both accum_loss_x and accum_loss_a
                        total_l2_norm_x = 0.0
                        total_l2_norm_a = 0.0

                        # Compute L2 norm for accum_loss_x
                        for grad in accum_loss_x.values():
                            total_l2_norm_x += torch.norm(grad, p=2) ** 2  # Squared L2 norm for each gradient tensor

                        # Compute L2 norm for accum_loss_a
                        for grad in accum_loss_a.values():
                            total_l2_norm_a += torch.norm(grad, p=2) ** 2  # Squared L2 norm for each gradient tensor

                        # Take the square root of the sum to get the final L2 norms
                        total_l2_norm_x = torch.sqrt(total_l2_norm_x)
                        total_l2_norm_a = torch.sqrt(total_l2_norm_a)

                        print(f"L2 norm of accum_loss_x: {total_l2_norm_x}")
                        print(f"L2 norm of accum_loss_a: {total_l2_norm_a}")


                        if self.cfg.deletion.loss_fn == 'erasediff':
                            scaling_factor = self.cfg.deletion.eta - sum(torch.sum(accum_loss_x[name] * accum_loss_a[name]) for name, _ in unet.named_parameters()) / (total_l2_norm_a ** 2)
                            scaling_factor = -max(scaling_factor, 0)
                        else:
                            # gradient norm fixing 
                            scaling_factor = self.cfg.deletion.scaling_norm / total_l2_norm_a # if total_l2_norm_a > self.cfg.deletion.loss_params.clipping_norm else 1
                        
                        wandb.log({'gradient/norm_loss_x': total_l2_norm_x, 'gradient/norm_loss_a': total_l2_norm_a, 'gradient/scaling_factor': scaling_factor}, step=global_step)
                        for name, param in unet.named_parameters():
                            param.grad = accum_loss_x[name] - scaling_factor * accum_loss_a[name]  # Set the gradient to the clipped difference

                        accum_loss_x = {}
                        accum_loss_a = {}

                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()

                
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                 if self.cfg.ema.use_ema:
                    ema_model.step(unet.parameters())
                 progress_bar.update(1)
                 global_step += 1
                 print(f"Current global_step: {global_step}")
                 if accelerator.is_main_process:
                    # Log metrics
                    log_metrics(unet)

                    # Save model
                    if self.cfg.checkpointing_steps and global_step % self.cfg.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if self.cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(self.cfg.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= self.cfg.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints)
                                    - self.cfg.checkpoints_total_limit
                                    + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        self.cfg.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            self.cfg.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            if loss is not None:
                logs["loss"] = loss.detach().item()
            if self.cfg.ema.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            accelerator.wait_for_everyone()
        # The end
        progress_bar.close()
        accelerator.end_training()