"""Модуль для обработки данных"""

import os
from os.path import isfile

import numpy as np
import pandas as pd
from PIL import Image


# def convert_data_to_percent(num: int) -> int:
#     """Преобразования диапазона 0-255 RGB каналов к процентам """
#     return int(num / 255 * 100)

def get_data_from_image(file_name: str) -> list:
    """Функция получает характеристики каждого пикселя из изображения и отдает список. Список содержит RGB цветовые
    координаты и координаты пикселя


    :param file_name: Путь к расположению файла
    :return: Функция возвращает результат в виде списка состоящего из цветовых координат пикселя и его координат на изображении
    """

    data_list = [['brightness', 'X', 'Y']]

    if os.path.isfile(os.path.join(os.getcwd(), f'image\\{file_name}')):

        img_object = Image.open(os.path.join(os.getcwd(), f'image\\{file_name}'))

        width, height = img_object.size

        for x in range(width):
            for y in range(height):
                rgb_pixel = img_object.getpixel((x, y))
                data_list.append([rgb_pixel, x, y])

    return data_list


def write_data_file(file_name: str, list_of_data: list) -> None:
    """Функция для записи значений в CVS файл

    :param file_name: Использует строковое значение для указания пути до файла куда нужно записать данные

    :param list_of_data: Использует список в качестве атрибута
    """

    if isfile(os.path.join(os.getcwd(), f'dataset\\{file_name}')):

        df = pd.read_csv(os.path.join(os.getcwd(), f'dataset\\{file_name}'))
        df[list_of_data[0]] = list_of_data[1:]
        df.to_csv(os.path.join(os.getcwd(), f'dataset\\{file_name}'), index=False)

    else:
        np_data = np.array(list_of_data[1:])
        df = pd.DataFrame(np_data, columns=list_of_data[0])
        df.to_csv(os.path.join(os.getcwd(), f'dataset\\{file_name}'), index=False)

    return None


def move_file(old_file_name: str, new_file_name: str) -> None:

    os.rename(os.path.join(os.getcwd(), f'dataset\\{old_file_name}'), os.path.join(os.getcwd(), f'results\\{new_file_name}'))

    return None


def read_data_file(file_name: str) -> np.ndarray or str:
    """Функция получения dataset данных из csv файла


    :param file_name: Использует строковое значение для указания пути до файла куда нужно записать данные
    """

    if os.path.isfile(os.path.join(os.getcwd(), f'dataset\\{file_name}')):
        loaded_df = pd.read_csv(os.path.join(os.getcwd(), f'dataset\\{file_name}'))

        np_dataset = loaded_df.to_numpy()
    else:
        return print(f'File {file_name} does not exist')

    return np_dataset


def mark_pixel(dataset: np.ndarray) -> list:
    """
    Функция размечает каждый пиксель в градациях серого.

    :param dataset: Получает в качестве параметра список с немаркированными пикселями
    :return: Функция возвращает результат в виде списка
    """
    color = []
    for _ in dataset:

        if 0 <= _[0] <= 50:
            color.append('Black')
        elif 51 <= _[0] <= 101:
            color.append('Dark gray')
        elif 102 <= _[0] <= 152:
            color.append('Gray')
        elif 153 <= _[0] <= 203:
            color.append('Light Gray')
        else:
            color.append('White')

    return color