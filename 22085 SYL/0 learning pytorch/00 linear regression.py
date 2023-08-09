import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

opt = optim.SGD([W, b], lr=0.01)

nb_epoch = 12000

for epoch in range(nb_epoch+1):

    hypo = x_train*W + b
    cost = torch.mean((hypo - y_train)**2)

    opt.zero_grad()  # gradient reset
    cost.backward()  # calculate gradient
    opt.step()       # change W, b

    if epoch % 100 == 0:
        print(f'{epoch:4d}/{nb_epoch} W: {W.item():.3f}, b: {b.item():.3f} Cost: {cost.item():.6f}')
