"""Это модуль для генерации данных для нейронной сети"""

from PIL import Image
import random
from datetime import datetime


def creat_random_pixel_image(mode: str, size: tuple[int, int] = (256, 256), path: str = '') -> None:
    """Функция создает изображение из набора пикселей которые получают случайные цвета

    Атрибуты:

    Mode -> использует строковое значение с помощью которого указывается необходимый режим. Список режимов смотреть в
    документации библиотеки pillow (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)

    Size -> использует картеж длиной 2, элементы картежа имеют тип int. Тут мы задаем размер изображения

    Path -> использует строковое значение для указания пути где будет создан файл. Если путь не указывать, файл с
    изображением будет создан в текущей директории модуля
    """

    random_image = Image.new(mode=mode, size=size)

    for x in range(size[0]):

        for y in range(size[1]):
            random_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            random_image.putpixel((x, y), tuple(random_color))

    date_time = datetime.now().strftime("%Y_%m_%d %H_%M")

    random_image.save(f'{path}random_image_{size[0]}x{size[1]} {date_time}.png')

    return None


def convert_image_to_grayscale(path: str) -> None:

    img = Image.open(path).convert('L')

    date_time = datetime.now().strftime("%Y_%m_%d %H_%M")

    img.save(f'grayscale_image {date_time}.png')

    return None
