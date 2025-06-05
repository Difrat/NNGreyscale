# import torch
# from torch.utils.data import DataLoader
#
# from torch_file import MyDataset, MyModel
# from data_processing_module import read_data_file
# import torch.optim as optim
# import torch.nn as nn
# from tqdm import tqdm
#
# dataset = read_data_file('Dataset/dataset.csv')
#
# count = len(dataset)
# len_train_data = round(count * .85)
# train_dataset = dataset[:len_train_data]
# test_dataset = dataset[len_train_data:]
#
# model = MyModel(1, 30, 5)
#
# d_train = MyDataset(train_dataset)
#
# train_data = DataLoader(d_train, batch_size=1 , shuffle=False, drop_last=False)
#
# optim = optim.Adam(model.parameters(), lr=0.01)
# loss_func = nn.CrossEntropyLoss()
#
# epochs = 5
# model.train()
#
# for _e in range(epochs):
#     loss_mean = 0
#     lm_count = 0
#     train_tqdm = tqdm(train_data, leave=True)
#
#     for x_train, y_train in train_tqdm:
#         prediction = model(x_train)
#         loss = loss_func(prediction, y_train.view(-1))
#
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#
#         lm_count += 1
#         loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
#         train_tqdm.set_description(f'Epoch {_e + 1}/{epochs}, loss_mean: {loss_mean:.3f}')
#
# model.load_state_dict(torch.load('MyModel.zip', weights_only=True))
# d_test = MyDataset(test_dataset)
# test_data = DataLoader(d_test, batch_size=1, shuffle=False, drop_last=False)
# Q = 0
#
# model.eval()
#
# for x_test, y_test in test_data:
#     with torch.no_grad():
#         p = model(x_test)
#         probabilities = torch.softmax(p, dim=0)
#         predicted_class = torch.argmax(probabilities, dim=0)
#         print(x_test, predicted_class)
#
# Q /= len(d_test)
#
# print(Q)