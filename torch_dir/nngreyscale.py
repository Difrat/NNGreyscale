"""Модуль обучения нейронной сети"""
import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch_file import MyDataset, MyModel


from save_load_model import save_model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class NNGreyscale:

    def __init__(self, data=None, model_nn=None):
        self.data = data
        self.__model_nn = model_nn
        self.__train_data = None
        self.__test_data = None

    def set_date(self, data=None) -> None:
        """
        Метод, который позволяет установить данные для нейросети и разбить их на тренировочную и тестовую часть

        :param data: должен получить массив данных numpy

        :return: None
        """

        if data is not None:

            self.date = data

        else:

            try:

                count = len(self.data)

                len_train_data = round(count * .85)

                self.__train_data = MyDataset(self.data[:len_train_data])
                self.__test_data = MyDataset(self.data[len_train_data:])

            except TypeError as e:

                print(
                    f'Произошла ошибка TypeError: не были указаны данные для обучения нейросети, укажите данные в поле date')

        return None

    def train_nn(self, model_name: str) -> None:
        """
        Данный метод применяется для обучения модели на тренировочных данных

        :param model_name: Использует строковое значение для указания мени модели

        :return: None
        """

        train_data = DataLoader(self.__train_data, batch_size=1, shuffle=False, drop_last=False)

        optim_f = optim.Adam(self.__model_nn.parameters(), lr=0.01)
        loss_func = nn.CrossEntropyLoss()

        epochs = 5
        self.__model_nn.train()

        for _e in range(epochs):
            loss_mean = 0
            lm_count = 0
            train_tqdm = tqdm(train_data, leave=True)

            for x_train, y_train in train_tqdm:
                prediction = self.__model_nn(x_train)
                loss = loss_func(prediction, y_train.view(-1))

                optim_f.zero_grad()
                loss.backward()
                optim_f.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
                train_tqdm.set_description(f'Epoch {_e + 1}/{epochs}, loss_mean: {loss_mean:.3f}')

        date_time = datetime.now().strftime("%Y_%m_%d %H_%M")

        save_model(self.__model_nn, os.path.join(os.getcwd(), f'models\\{model_name} {date_time}.zip'))

        return None

    def test_nn(self):

        test_data = DataLoader(self.__test_data, batch_size=1, shuffle=False, drop_last=False)
        Q = 0

        self.__model_nn.eval()

        for x_test, y_test in test_data:
            with torch.no_grad():
                p = self.__model_nn(x_test)
                p = torch.argmax(p, dim=0)
                y = torch.argmax(y_test, dim=1)
                Q += torch.sum(p == y).item()

        Q /= len(self.__test_data)
        print(Q)

    def run_nn(self):
        pass



