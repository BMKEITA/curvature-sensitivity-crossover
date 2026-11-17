#!/usr/bin/env python3
"""
extract_dino_features.py
Extracts frozen DINO ViT-B/16 (d=768) L2-normalized features for MNIST, CIFAR-10, and CIFAR-100
on GPU (NVIDIA TITAN Xp compatible) and saves them to .npz archives.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as T


def get_dataset(name: str, data_dir: str = "./data"):
    transform_rgb = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    transform_gray = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    if name == "mnist":
        train_ds = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform_gray)
        test_ds = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform_gray)
    elif name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_rgb)
        test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_rgb)
    elif name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_rgb)
        test_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_rgb)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return ConcatDataset([train_ds, test_ds])


@torch.no_grad()
def extract_and_save(dataset_name: str, out_path: str, batch_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DINO Extractor] Loading DINO ViT-B/16 (d=768) on {device}...")
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16", pretrained=True)
    model.eval().to(device)

    ds = get_dataset(dataset_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    feats_list, labels_list = [], []
    total = len(ds)
    seen = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        emb = model(imgs)
        emb = F.normalize(emb, p=2, dim=-1)  # L2-normalized features (||x_i||_2 = 1)
        feats_list.append(emb.cpu().numpy().astype(np.float32))
        labels_list.append(labels.numpy().astype(np.int64))
        seen += imgs.size(0)
        if seen % 5000 == 0 or seen == total:
            print(f"  Processed {seen}/{total} samples for {dataset_name}...")

    X = np.concatenate(feats_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    np.savez_compressed(out_path, X=X, y=y, features=X, labels=y)
    print(f"[Done] Saved {out_path}: X.shape={X.shape}, y.shape={y.shape}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "cifar100"])
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    for dname in args.datasets:
        out_file = f"dino_{dname}.npz"
        if os.path.exists(out_file):
            print(f"[Skip] {out_file} already exists.")
        else:
            extract_and_save(dname, out_file, args.batch_size)