import argparse
import copy
import os
import pathlib
import sys
from typing import List, Tuple

import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

import dataset
import network


def setup_train_val_split(count: int) -> Tuple[np.ndarray, np.ndarray]:
    train_count = int(round(count * 0.8))
    train_indices = np.arange(train_count)
    val_indices = np.arange(train_count, count)

    return train_indices, val_indices

def setup_train_val_datasets(data_dir: str) -> Tuple[dataset.ImageDataset, dataset.ImageDataset]:
    train_indices, val_indices = setup_train_val_split(10000)

    image_x0_path = f"{data_dir}/image_x0"
    image_x1_path = f"{data_dir}/image_x1"
    image_y_path = f"{data_dir}/image_y"
    train_dataset = dataset.ImageDataset(train_indices, image_x0_path, image_x1_path, image_y_path)
    val_dataset = dataset.ImageDataset(val_indices, image_x0_path, image_x1_path, image_y_path)
    
    return train_dataset, val_dataset

def setup_train_val_loaders(data_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = setup_train_val_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=8
    )

    return train_loader, val_loader

def train_1epoch(model: nn.Module, train_loader: DataLoader, lossfun, optimizer, lr_scheduler, device) -> float:
    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader):
        x = data["x"].to(device)
        y = data["y"].to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = lossfun(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        lr_scheduler.step()

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss


def validate_1epoch(model: nn.Module, val_loader: DataLoader, lossfun, device) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in tqdm(val_loader):
            x = data["x"].to(device)
            y = data["y"].to(device)

            out = model(x)
            loss = lossfun(out.detach(), y)

            total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss


def train(
    model: nn.Module, 
    optimizer,
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    lr_scheduler,
    n_epochs: int, 
    device):
    lossfun = torch.nn.L1Loss()

    for epoch in tqdm(range(n_epochs)):
        train_loss = train_1epoch(
            model, train_loader, lossfun, optimizer, lr_scheduler, device
        )
        val_loss = validate_1epoch(model, val_loader, lossfun, device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch}, train loss={train_loss}, val loss={val_loss}, lr={lr}"
        )

def predict(model, loader, device):
    preds = []
    for data in tqdm(loader):
        with torch.set_grad_enabled(False):
            x = data["x"].to(device)
            y = model(x)
        
        y = y.cpu().numpy()
        for i in range(len(y)):
            preds.append(y[i, 0, :, :])
    
    return preds

def train_unet(data_dir: str, batch_size: int, device: torch.device) -> nn.Module:
    model = network.UNet_2D()
    model.to(device)

    n_epochs = 10
    train_loader, val_loader = setup_train_val_loaders(data_dir, batch_size)
    n_iterations = len(train_loader) * n_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)
    train(model, optimizer, train_loader, val_loader, n_epochs=n_epochs, lr_scheduler=lr_scheduler, device=device)
    return model

def predict_unet(data_dir: str, model: nn.Module, batch_size: int, device) -> List[np.ndarray]:
    # めんどくさいしvalでええか……
    _, val_loader = setup_train_val_loaders(data_dir, batch_size)
    preds = predict(model, val_loader, device)
    return preds

def run(data_dir: str, device: torch.device):
    batch_size = 32
    model = train_unet(data_dir, batch_size, device)
    preds = predict_unet(data_dir, model, batch_size, device)

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()

    print(f"params: {params}")

    for i, pred in enumerate(preds):
        pred = np.round(pred * 255)
        img = Image.fromarray(pred.astype(np.uint8))
        img.save(f"{data_dir}/pred/{i:0>4}.bmp")       


if __name__ == "__main__":
    run("data", "cuda:0")