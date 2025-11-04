"""Модуль конфигурации нейронной сети"""
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as f


class MyModel(nn.Module):

    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.layer2(x)
        return x


class MyDataset(Dataset):

    def __init__(self, list_dataset: np.ndarray) -> None:
        self.list_dataset = list_dataset
        self.targets = torch.eye(5)
        self.dict_of_targets = {'Black': 0, 'Dark gray': 1, 'Gray': 2, 'Light Gray': 3, 'White': 4}

    def __len__(self):
        return len(self.list_dataset)

    def __getitem__(self, idx):

        t = self.list_dataset[idx][-1]
        target = 0

        if isinstance(t, str):

            target = self.targets[self.dict_of_targets[t]]

        data = self.list_dataset[idx][0]
        light_pix = torch.tensor(data, dtype=torch.float32)

        return light_pix, target
