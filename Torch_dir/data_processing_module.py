"""Модуль для обработки данных"""

import os
import numpy as np
import pandas as pd
from PIL import Image


def convert_data_to_percent(num: int) -> int:
    """Преобразования диапазона 0-255 RGB каналов к процентам """
    return int(num / 255 * 100)


def write_data_file(file_name: str, list_of_data: list) -> None:
    """Функция для записи значений в CVS файл

    Атрибуты:

    file_name -> использует строковое значение для указания пути до файла куда нужно записать данные

    list_of_data -> использует список в качестве атрибута
    """

    np_data = np.array(list_of_data[1:])
    df = pd.DataFrame(np_data, columns=list_of_data[0])
    df.to_csv(file_name, index=False)

    return None


def read_data_file(file_name: str) -> np.ndarray or str:
    """Функция получения dataset данных из csv файла

    Атрибуты:

    file_name -> использует строковое значение для указания пути до файла куда нужно записать данные
    """
    if os.path.isfile(file_name):
        loaded_df = pd.read_csv(file_name)

        np_dataset = loaded_df.to_numpy()
    else:
        return print(f'File {file_name} does not exist')

    return np_dataset


def get_data_from_image(path: str) -> list:
    """Функция получает характеристики каждого пикселя из изображения и отдает список. Список содержит RGB цветовые
    координаты и координаты пикселя

    Атрибуты:

    Path -> Путь к расположению файла
    """

    data_list = [['brightness', 'X', 'Y']]

    img_object = Image.open(path)

    width, height = img_object.size

    for x in range(width):
        for y in range(height):
            rgb_pixel = img_object.getpixel((x, y))
            data_list.append([rgb_pixel, x, y])

    return data_list

def mark_pixel(dataset: np.ndarray) -> list:
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


