import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_openml

import _timetool
t_p = _timetool.Stopwatch()

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
torch.manual_seed(1)
if USE_CUDA: torch.cuda.manual_seed_all(1)

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
mnist.target = mnist.target.astype(np.int8)
X = mnist.data / 255
Y = mnist.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/7, random_state=0)
X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
Y_train = torch.LongTensor(Y_train).to(device)
Y_test = torch.LongTensor(Y_test).to(device)

batch_size = 1024
ds_train = TensorDataset(X_train, Y_train)
ds_test = TensorDataset(X_test, Y_test)
loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=True)

print(f'Data loading done in {t_p.term():3.3f}s')

model = nn.Sequential().to(device)
model.add_module('fc1', nn.Linear(784, 100).to(device))
model.add_module('relu1', nn.ReLU().to(device))
model.add_module('fc2', nn.Linear(100, 100).to(device))
model.add_module('relu2', nn.ReLU().to(device))
model.add_module('fc3', nn.Linear(100, 10).to(device))

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    global t_e
    t_e.reset()

    for data, targets in loader_train:
        predict = model(data)
        loss = loss_fn(predict, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tmp = t_e.term()
    print(f'E: {epoch:5d} | TE: {tmp:3.3f}s  '
          f'TB: {1000*tmp*batch_size/len(loader_train.dataset):3.3f}ms')


def test():
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in loader_test:

            predict = model(data)
            _, predict = torch.max(predict.data, 1)
            correct += predict.eq(targets.data.view_as(predict)).sum()

    data_num = len(loader_test.dataset)
    print(f'AC: {correct}/{data_num} {100.*correct/data_num:3.2f}%')


print(f'Pre-training settings done in {t_p.term():3.3f}s')

t_e = _timetool.Stopwatch()

for ep in range(3):
    train(ep)
    test()
