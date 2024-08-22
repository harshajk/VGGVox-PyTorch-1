| PyTorch | TensorFlow |
| ------- | ---------- |
| import torch | import tensorflow as tf |
| import torch.nn as nn | |
| import torch.optim as optim | |
| import torch.nn.functional as F | |
| from torch.optim import lr_scheduler, SGD, Adam |  |
| from torch.utils.data import Subset, Dataset, DataLoader | |
| from torchvision import transforms | |
| transformers=transforms.ToTensor() | |
