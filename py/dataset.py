import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 40

class ImageDataset(Dataset):
  def __init__(
      self,
      indices: np.ndarray,
      path_x0: str,
      path_x1: str,
      path_y: str,
      ):
    self.path_x0 = path_x0
    self.path_x1 = path_x1
    self.path_y = path_y
    self.indices = indices

  def __getitem__(self, i):
    index = self.indices[i]
    img_x0_path = f"{self.path_x0}/{index:0>4}.bmp"
    img_x0 = np.array(Image.open(img_x0_path)) / 255
    img_x0 = img_x0[np.newaxis, :, :]
    img_x0 = torch.from_numpy(img_x0.astype(np.float32)).clone()

    img_x1_path = f"{self.path_x1}/{index:0>4}.bmp"
    img_x1 = np.array(Image.open(img_x1_path)) / 255
    img_x1 = img_x1[np.newaxis, :, :]
    img_x1 = torch.from_numpy(img_x1.astype(np.float32)).clone()
    img_x = torch.cat([img_x0, img_x1])

    img_y_path = f"{self.path_y}/{index:0>4}.bmp"
    img_y = np.array(Image.open(img_y_path)) / 255
    img_y = img_y[np.newaxis, :, :]
    img_y = torch.from_numpy(img_y.astype(np.float32)).clone()

    data = {"x": img_x, "y": img_y}
    return data
  
  def __len__(self):
    return len(self.indices)