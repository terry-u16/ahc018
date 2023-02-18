import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 40

class ImageDataset(Dataset):
  def __init__(
      self,
      indices: np.ndarray,
      path_x: str,
      path_y: str,
      ):
    self.path_x = path_x
    self.path_y = path_y
    self.indices = indices

  def __getitem__(self, i):
    index = self.indices[i]
    img_x_path = f"{self.path_x}/{index:0>4}.bmp"
    img_x = np.array(Image.open(img_x_path)) / 255
    img_x = img_x[np.newaxis, :, :]
    img_x = torch.from_numpy(img_x.astype(np.float32)).clone()

    img_y_path = f"{self.path_y}/{index:0>4}.bmp"
    img_y = np.array(Image.open(img_y_path)) / 255
    img_y = img_y[np.newaxis, :, :]
    img_y = torch.from_numpy(img_y.astype(np.float32)).clone()

    data = {"x": img_x, "y": img_y}
    return data
  
  def __len__(self):
    return len(self.indices)