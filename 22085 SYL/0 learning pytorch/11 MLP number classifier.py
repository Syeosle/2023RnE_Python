import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
torch.manual_seed(1)
if USE_CUDA: torch.cuda.manual_seed_all(1)

digits = load_digits()
X = digits.data
Y = digits.target
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y, dtype=torch.int64).to(device)


model = nn.Sequential(
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 16), nn.ReLU(),
    nn.Linear(16, 10), nn.ReLU()
).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters())
l_losses = []

for ep in range(1001):

    predict = model(X)
    cost = loss_fn(predict, Y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if ep % 50 == 0: print(f'{ep:4d} C: {cost.item():.6f}')

    l_losses.append(cost.item())

plt.plot(l_losses)
plt.show()
