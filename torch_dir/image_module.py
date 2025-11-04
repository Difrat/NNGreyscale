"""Это модуль для генерации данных для нейронной сети"""
import os
from os.path import isfile

from PIL import Image
import random
from datetime import datetime


def creat_random_pixel_image(mode: str, size: tuple[int, int] = (256, 256), file_name: str = 'random_image') -> None:
    """Функция создает изображение из набора пикселей которые получают случайные цвета

    :param mode: Использует строковое значение с помощью которого указывается необходимый режим. Список режимов смотреть в
    документации библиотеки pillow (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)

    :param size: Картеж длиной 2, элементы картежа имеют тип int. Тут мы задаем размер изображения

    :param file_name: Принимает строковое значение в качестве имени сохраняемого файла.
    """

    random_image = Image.new(mode=mode, size=size)

    for x in range(size[0]):

        for y in range(size[1]):
            random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            random_image.putpixel((x, y), tuple(random_color))

    date_time = datetime.now().strftime("%Y_%m_%d %H_%M")

    random_image.save(os.path.join(os.getcwd(), f'image\\{file_name}_{size[0]}x{size[1]} {date_time}.png'))

    return None


def convert_image_to_grayscale(file_name: str) -> None:
    """Функция переводит цветные пиксели в оттенки серого


    :param file_name: Использует строковое значение в качестве имени сохраняемого файла. Расширение файла не указывается
    """
    if isfile(os.path.join(os.getcwd(), f'image\\{file_name}.png')):

        img = Image.open(os.path.join(os.getcwd(), f'image\\{file_name}.png')).convert('L')

        date_time = datetime.now().strftime("%Y_%m_%d %H_%M")

        img.save(os.path.join(os.getcwd(), f'image\\{file_name}_greyscale {date_time}.png'))

    else:
        print('Указан неверный путь')

    return None

