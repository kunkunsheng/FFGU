# dataset.py
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import os
import torch
from PIL import Image, UnidentifiedImageError


class FaceDataset(Dataset):
    def __init__(self, root_dir, client_id, transform=None):
        self.root_dir = root_dir
        self.client_id = f"{client_id:03d}"
        self.transform = transform
        # List all image files in the root directory (assuming all files in the folder are images)
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if
                            img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            # 如果打开图片失败，打印错误并返回一个默认值或空值
            print(f"Warning: Unable to open image {img_path}. Error: {e}")
            return None, self.client_id
        if self.transform:
            image = self.transform(image)
        return image, self.client_id


def get_client_dataloader(client_id,
                          root='XXXXXXXX/FU-GUIDE/Deep3DFaceRecon/checkpoints/model_name/celeb',
                          batch_size=8,
                          num_samples=40):
    """
        为指定的客户端创建 DataLoader，用于加载该客户端的数据（仅随机选择 num_samples 张图片）。

        Args:
            client_id (int): 客户端 ID，确定要加载哪个客户端的数据。
            root (str): 根目录，包含所有客户端的数据文件夹。
            batch_size (int): 批次大小。
            num_samples (int): 要随机选择的图片数量。

        Returns:
            DataLoader: 用于加载该客户端数据的数据加载器。
        """
    client_folder = os.path.join(root, f"{client_id:03d}")  # 客户端文件夹，例如 '000', '001'
    print(f'dataset.py中的cid :{client_id:03d}')

    # 定义图像转换操作（可以根据需要自定义）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])

    client_dataset = FaceDataset(client_folder, client_id, transform=transform)

    # 创建一个随机选择的索引列表，选取 `num_samples` 个样本
    indices = torch.randperm(len(client_dataset)).tolist()[:num_samples]

    # 创建一个 SubsetRandomSampler 来只从这些索引中采样
    sampler = SubsetRandomSampler(indices)

    # 创建 DataLoader，使用 sampler 来选择数据
    dataloader = DataLoader(client_dataset, batch_size=batch_size, sampler=sampler)

    return dataloader, len(indices)

def get_client_dataloader_init(client_id,
                          root='XXXXXXXX/FU-GUIDE/Deep3DFaceRecon/checkpoints/model_name/celeb',
                          batch_size=4):
    """
    为指定的客户端创建 DataLoader，用于加载该客户端的数据。

    Args:
        client_id (int): 客户端 ID，确定要加载哪个客户端的数据。
        root (str): 根目录，包含所有客户端的数据文件夹。
        batch_size (int): 批次大小。

    Returns:
        DataLoader: 用于加载该客户端数据的数据加载器。
    """
    client_folder = os.path.join(root, f"{client_id:03d}")  # 客户端文件夹，例如 '000', '001'
    print(f'dataset.py中的cid :{client_id:03d}')
    # 定义图像转换操作（可以根据需要自定义）
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
    ])

    client_dataset = FaceDataset(client_folder, client_id, transform=transform)

    # 创建 DataLoader
    dataloader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)

    return dataloader, len(client_dataset)

