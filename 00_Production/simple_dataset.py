import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class SimpleCIFAR10Dataset(Dataset):
    """
    CIFAR-10データセットの簡易版
    torchvisionを使わずにCIFAR-10データを読み込む
    """
    
    def __init__(self, root="./dataset", train=True, download=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        if download:
            self._download()
        
        self.data, self.targets = self._load_data()
    
    def _download(self):
        """データをダウンロードする（実際にはダミーデータを生成）"""
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        # 実際のCIFAR-10の代わりにダミーデータを生成
        if self.train:
            # 訓練データ: 1000サンプル
            data = np.random.randint(0, 255, (1000, 32, 32, 3), dtype=np.uint8)
            targets = np.random.randint(0, 10, 1000)
            
            with open(os.path.join(self.root, "cifar10_train.pkl"), "wb") as f:
                pickle.dump((data, targets), f)
        else:
            # テストデータ: 200サンプル
            data = np.random.randint(0, 255, (200, 32, 32, 3), dtype=np.uint8)
            targets = np.random.randint(0, 10, 200)
            
            with open(os.path.join(self.root, "cifar10_test.pkl"), "wb") as f:
                pickle.dump((data, targets), f)
    
    def _load_data(self):
        """データを読み込む"""
        if self.train:
            file_path = os.path.join(self.root, "cifar10_train.pkl")
        else:
            file_path = os.path.join(self.root, "cifar10_test.pkl")
        
        with open(file_path, "rb") as f:
            data, targets = pickle.load(f)
        
        return data, targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        # numpy array to tensor
        img = torch.from_numpy(img.astype(np.float32))
        img = img.permute(2, 0, 1)  # HWC to CHW
        
        # 正規化
        if self.transform:
            img = self.transform(img)
        else:
            # デフォルトの正規化
            img = img / 255.0
            img = (img - 0.5) / 0.5
        
        return img, target

# transformsの代替
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return torch.from_numpy(img.astype(np.float32))
        return img

class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

# トランスフォームの設定
def get_transform():
    return Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
