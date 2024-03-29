import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

'''

Low-level

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros((1, 3), requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for ep in range(nb_epochs + 1):

    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if ep % 100 == 0:
        print(f'{ep:4d}/{nb_epochs} Cost: {cost.item():.6f}')
'''

'''

Used nn.Module

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

model = nn.Linear(4, 3)

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for ep in range(nb_epochs + 1):

    z = model(x_train)
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if ep % 100 == 0:
        print(f'{ep:4d} C: {cost.item():.6f}')
'''

''''''

# Used Class

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for ep in range(nb_epochs + 1):

    z = model(x_train)
    cost = F.cross_entropy(z, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if ep % 100 == 0:
        print(f'{ep:4d} C: {cost.item():.6f}')