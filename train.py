import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import sys
import glob
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
midas_dir = os.path.join(current_dir, 'MiDaS')
sys.path.append(midas_dir)

from midas.transforms import Resize, NormalizeImage, PrepareForNet
from midas.model_loader import default_models, load_model
from torchvision.transforms import Compose
import torch.nn.functional as F

def load_midas_model(model_path, model_type, device, optimize=False, height=None, square=False):
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)
    return model, transform, net_w, net_h

class CustomTransform:
    def __init__(self, midas_transform, normalize_transform):
        self.midas_transform = midas_transform
        self.normalize_transform = normalize_transform

    def __call__(self, sample):
        sample = self.midas_transform(sample)
        
        if isinstance(sample['image'], np.ndarray):
            sample['image'] = torch.from_numpy(sample['image'])
        
        if sample['image'].dtype != torch.float32:
            sample['image'] = sample['image'].float()
        
        sample['image'] = self.normalize_transform(sample['image'])
        
        return sample

class DTUDataset(Dataset):
    def __init__(self, root_dir, transform=None, net_w=256, net_h=256):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'Rectified')
        self.depth_dir = os.path.join(root_dir, 'Depths')
        self.img_size = (net_w, net_h)

        if transform is None:
            self.transform = Compose([
                Resize(
                    net_w,
                    net_h,
                    resize_target=True,
                    keep_aspect_ratio=False,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                PrepareForNet()
            ])
        else:
            self.transform = transform

        if not os.path.exists(self.image_dir):
            raise ValueError(f"错误：图像目录不存在 {self.image_dir}")
        if not os.path.exists(self.depth_dir):
            raise ValueError(f"错误：深度目录不存在 {self.depth_dir}")

        self.data_pairs = []

        rect_scans = set(os.listdir(self.image_dir))
        depth_scans = set(d for d in os.listdir(self.depth_dir) if d.endswith('_train'))

        common_scans = rect_scans.intersection(depth_scans)
        print(f"找到 {len(common_scans)} 个共同的训练扫描文件夹")

        for scan in common_scans:
            img_dir = os.path.join(self.image_dir, scan)
            depth_dir = os.path.join(self.depth_dir, scan)

            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('_r5000.png')])
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.pfm') and f.startswith('depth_map')])

            min_files = min(len(img_files), len(depth_files))
            if min_files == 0:
                print(f"警告：{scan} 中没有找到匹配的图像和深度图对")
                continue

            if len(img_files) != len(depth_files):
                print(f"警告：{scan} 中的图像数量 ({len(img_files)}) 与深度图数量 ({len(depth_files)}) 不匹配")
                print(f"将只使用前 {min_files} 对数据")
                img_files = img_files[:min_files]
                depth_files = depth_files[:min_files]

            for img_file, depth_file in zip(img_files, depth_files):
                self.data_pairs.append({
                    'image': os.path.join(img_dir, img_file),
                    'depth': os.path.join(depth_dir, depth_file)
                })

        if not self.data_pairs:
            raise ValueError("没有找到有效的训练数据对！")

        print(f"总共找到 {len(self.data_pairs)} 个训练数据对")

    def read_pfm(self, path):
        with open(path, 'rb') as file:
            header = file.readline().decode('utf-8').rstrip()
            if header not in ['PF', 'Pf']:
                raise ValueError('Not a PFM file')
            
            dim_line = file.readline().decode('utf-8').rstrip()
            w, h = map(int, dim_line.split())
            
            scale_line = file.readline().decode('utf-8').rstrip()
            scale = float(scale_line)
            
            data = np.fromfile(file, np.float32)
            data = data.reshape((h, w))
            
            if scale < 0:
                data = np.flipud(data)
                scale = -scale
            
            return data

    def __getitem__(self, idx):
        data_pair = self.data_pairs[idx]

        image = cv2.imread(data_pair['image'])
        if image is None:
            raise ValueError(f"Failed to load image: {data_pair['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        depth = self.read_pfm(data_pair['depth'])
        depth = np.ascontiguousarray(depth)

        depth = cv2.resize(depth, self.img_size, interpolation=cv2.INTER_LINEAR)

        sample = {
            "image": image,
            "depth": depth
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        image = sample["image"]  
        depth = torch.from_numpy(sample["depth"]).unsqueeze(0).float()  

        return image, depth

    def __len__(self):
        return len(self.data_pairs)

def plot_losses(losses, save_path='training_loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    losses = []  
    
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0
        
        for i, (images, depths) in enumerate(train_loader):
            images = images.to(device)
            depths = depths.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            pred_depths = outputs.squeeze(1)
            target_depths = depths.squeeze(1)
            
            loss = criterion(pred_depths, target_depths)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / batch_count
        losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}')
        
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(save_dir, f'midas_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'losses': losses,  
            }, checkpoint_path)
            
            model_path = os.path.join(save_dir, f'midas_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            
            plot_losses(losses, os.path.join(save_dir, 'training_loss.png'))
            
            print(f'Saved model checkpoint and loss plot at epoch {epoch + 1}')

def load_checkpoint(checkpoint_path, model, optimizer=None):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        losses = checkpoint.get("losses", [])
        print(f"Resuming from epoch {start_epoch}")
    else:
        model.load_state_dict(checkpoint)
        start_epoch = 0
        losses = []
        print("Loaded model state dict")
    
    return model, optimizer, start_epoch, losses

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 50
    
    model_type = "dpt_swin2_tiny_256"
    checkpoint_path = "./midas_checkpoint_epoch_5.pth"
    initial_model_path = "/home/work3d/dpt_swin2_tiny_256.pt"
    
    print(f"Loading initial model {model_type}")
    model, midas_transform, net_w, net_h = load_midas_model(initial_model_path, model_type, device, optimize=False)
    model = model.train()  
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, losses = load_checkpoint(checkpoint_path, model, optimizer)
    else:
        start_epoch = 0
        losses = []
        print("Starting training from scratch")
    
    print(f"Model ready. Input size: {net_w}x{net_h}")
    
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    transform = CustomTransform(midas_transform, normalize_transform)
    
    dataset = DTUDataset(root_dir='/home/DTU', transform=transform, net_w=net_w, net_h=net_h)
    print(f"Found {len(dataset)} training samples")
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    
    criterion = nn.MSELoss()
    
    print("start train!")
    train_model(model, train_loader, criterion, optimizer, 
                device, num_epochs=num_epochs-start_epoch)

if __name__ == '__main__':
    main()