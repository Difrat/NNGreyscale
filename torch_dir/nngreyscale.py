"""Модуль обучения нейронной сети"""
import torch
from torch.utils.data import DataLoader

from torch_file import MyDataset, MyModel
from data_processing_module import read_data_file
from save_load_model import save_model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class NNGreyscale:

    def __init__(self, date, model):
        self.date = date
        self.model = model

    def train_nn(self, path):
        d_train = MyDataset(self.date)
        train_data = DataLoader(d_train, batch_size=1, shuffle=False, drop_last=False)

        optim_f = optim.Adam(self.model.parameters(), lr=0.01)
        loss_func = nn.CrossEntropyLoss()

        epochs = 5
        self.model.train()

        for _e in range(epochs):
            loss_mean = 0
            lm_count = 0
            train_tqdm = tqdm(train_data, leave=True)

            for x_train, y_train in train_tqdm:
                prediction = self.model(x_train)
                loss = loss_func(prediction, y_train.view(-1))

                optim_f.zero_grad()
                loss.backward()
                optim_f.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
                train_tqdm.set_description(f'Epoch {_e + 1}/{epochs}, loss_mean: {loss_mean:.3f}')

        save_model(self.model, path)


    def test_nn(self, d_test):
        test_data = DataLoader(d_test, batch_size=1, shuffle=False, drop_last=False)
        Q = 0

        self.model.eval()

        for x_test, y_test in test_data:
            with torch.no_grad():
                p = self.model(x_test)
                p = torch.argmax(p, dim=0)
                y = torch.argmax(y_test, dim=1)
                Q += torch.sum(p == y).item()

        Q /= len(d_test)



# dataset = read_data_file('Dataset/dataset.csv')
# count = len(dataset)
# len_train_data = round(count * .85)
# train_dataset = dataset[:len_train_data]
# test_dataset = dataset[len_train_data:]
# less_lst_val = []
# loss_lst = []
# d_test = MyDataset(test_dataset)
# model = MyModel(1, 30, 5)