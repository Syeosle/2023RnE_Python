import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
torch.manual_seed(1)
if USE_CUDA: torch.cuda.manual_seed_all(1)

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

model = nn.Sequential(
    nn.Linear(2, 10), nn.Sigmoid(),
    nn.Linear(10, 10), nn.Sigmoid(),
    nn.Linear(10, 10), nn.Sigmoid(),
    nn.Linear(10, 1), nn.Sigmoid()
).to(device)

criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1)

for ep in range(10001):
    hypo = model(X)
    cost = criterion(hypo, Y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if ep % 100 == 0: print(f'{ep:5d} C: {cost.item():.6f}')

# ===========================================================

with torch.no_grad():
    hypo = model(X)
    predict = (hypo > 0.5).float()
    acc = (predict==Y).float().mean()
    print(f'hypo   : {hypo.detach().cpu().squeeze().numpy()}\n'
          f'predict: {predict.detach().cpu().squeeze().numpy()}\n'
          f'Y      : {Y.cpu().squeeze().numpy()}\n'
          f'Acc    : {acc.item()}\n')
