import itertools
import random
import torch
from torch.utils.data import Dataset

class HairDataset(Dataset):
  def __init__(self, filepath):
    self.images = []

  def load(self, filepath):
    pass

  # Return review
  def __getitem__(self, idx):
      return self.images[idx]

  # Return the number of elements of the dataset.
  def __len__(self):
      return len(self.images)
