from torch_dir.data_processing_module import read_data_file
from torch_dir.nngreyscale import NNGreyscale
from torch_dir.torch_file import MyModel
import os

if __name__ == '__main__':

    dataset = read_data_file('dataset.csv')
    model = MyModel(1, 30, 5)
    nn_greyscale = NNGreyscale(data=dataset, model_nn=model)
    # nn_greyscale.set_date()
    # nn_greyscale.train_nn('Test')
    # nn_greyscale.test_nn()
    nn_greyscale.set_model_nn('Greyscale 2025_11_02 11_58')
    nn_greyscale.run_model()
