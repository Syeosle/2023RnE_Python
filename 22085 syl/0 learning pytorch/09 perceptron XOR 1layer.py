import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
if USE_CUDA: torch.cuda.manual_seed(1)

X = torch.FloatTensor([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

linear = nn.Linear(2, 1)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid).to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

for ep in range(10001):
    hypo = model(X)

    optimizer.zero_grad()
    cost = criterion(hypo, Y)
    cost.backward()
    optimizer.step()

    if ep % 100 == 0: print(f'{ep:5d} C: {cost.item():.6f}')
