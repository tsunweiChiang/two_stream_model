import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import random


class TwoStreamDataset(Dataset):
    def __init__(self, samples, class_to_idx):
        """
        run once when instantiating the Dataset object.
        """
        # samples: list of (spatial_mp4_path, temporal_npy_path, label)
        self.samples = samples
        #紀錄每個 class 對應的 indedx
        self.class_to_idx = class_to_idx
        #將圖片resize成符合resnet大小，並正規化
        self.spatial_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        """
        returns the number of samples in our dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        loads and returns a sample from the dataset at the given index idx. 
        """
        spatial_path, temporal_path, label = self.samples[idx]

        # Spatial: 取中間 frame
        cap = cv2.VideoCapture(spatial_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        mid_frame = frames[len(frames) // 2]
        mid_frame = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB)
        # (H,W,3) -> (3, 224, 224)
        spatial_input = self.spatial_transform(mid_frame)   

        # Temporal: 載入 optical flow npy (H, W, 20)
        flow = np.load(temporal_path)
        num_channels = flow.shape[2]
        # (H, W, 20) uint8 -> (20, 224, 224)
        flow_resized =np.stack(
            [cv2.resize(flow[...,c],(224,224),interpolation=cv2.INTER_AREA)for c in range(num_channels)],axis=0
        )
        temporal_input = torch.from_numpy(flow_resized).float()/255.0

        return spatial_input, temporal_input, torch.tensor(label, dtype=torch.long)


def _build_samples(spatial_root, temporal_root, expected_channels=20):
    """掃描資料夾，回傳 (samples, class_to_idx)，過濾 channel 數不符的檔案。"""
    #看有多少 class, 並且建立 index
    classes = sorted([
        d for d in os.listdir(spatial_root)
        if os.path.isdir(os.path.join(spatial_root, d))
    ])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    samples = []
    skipped = 0

    for cls in classes:
        label = class_to_idx[cls]
        spatial_cls_dir  = Path(spatial_root) / cls
        temporal_cls_dir = Path(temporal_root) / cls

        for mp4_file in sorted(spatial_cls_dir.glob("*.mp4")):
            npy_file = temporal_cls_dir / f"{mp4_file.stem}_flow.npy"
            if not npy_file.exists():
                continue
            if np.load(npy_file, mmap_mode='r').shape[2] != expected_channels:
                skipped += 1
                continue
            samples.append((str(mp4_file), str(npy_file), label))

    if skipped:
        print(f"Warning: 跳過 {skipped} 個 channel 數不符的檔案")

    return samples, class_to_idx


def build_train_val_loaders(spatial_root, temporal_root,
                             val_ratio=0.2, batch_size=16,
                             num_workers=4, seed=42):
    """
    隨機切出 val_ratio 比例當 val，其餘當 train。
    回傳 (train_loader, val_loader)。
    """
    samples, class_to_idx = _build_samples(spatial_root, temporal_root)

    random.seed(seed)
    random.shuffle(samples)

    n_val        = int(len(samples) * val_ratio)
    val_samples  = samples[:n_val]
    train_samples = samples[n_val:]

    print(f"Classes : {class_to_idx}")
    print(f"Train   : {len(train_samples)} 筆")
    print(f"Val     : {len(val_samples)} 筆")

    train_loader = DataLoader(
        TwoStreamDataset(train_samples, class_to_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        TwoStreamDataset(val_samples, class_to_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
