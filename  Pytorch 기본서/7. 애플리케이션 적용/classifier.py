import torch
from torch import nn, optim
from torch.utils.data import (Dataset,
                            DataLoader,
                            TensorDataset)
import tqdm

from torchvision import models

def create_network():
    
    # resnet18 기반의 이종 분류 식별 모델
    net = models.resnet18()
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim, 2)
    return net