import os
from os.path import isdir


def create_sys_folder():
    folder_name = ['dataset', 'images', 'models', 'results']

    for dir_name in folder_name:
        os.mkdir(f'{os.path.join(os.getcwd(), dir_name)}')

    return None


print(os.path.join(os.getcwd(), 'models'))
print(isdir(os.path.join(os.getcwd(), 'models')))