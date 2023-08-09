import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import _timetool
sw = _timetool.Stopwatch()

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
torch.manual_seed(777)
if USE_CUDA: torch.cuda.manual_seed_all(777)

lr = 0.001
total_epochs = 15
batch_size = 128

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader_train = DataLoader(dataset=mnist_train,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5

        # ImgIn = (?, 28, 28, 1) --(Conv)-> (?, 28, 28, 32) --(Pool)-> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # (?, 14, 14, 32) -> (?, 14, 14, 64) -> (?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # (?, 7, 7, 64) -> (?, 7, 7, 128) -> (?, 4, 4, 128)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        # 4x4x128 -> 625
        self.fc1 = nn.Linear(4*4*128, 625)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1-self.keep_prob)
        )

        # 7*7*64 input to 10 output
        self.fc2 = nn.Linear(625, 10)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out


model = CNN().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

total_batch = len(data_loader_train)

print(f'Settings done in {sw.term():.3f}s\n'
      f'Total batch count: {total_batch}')

preload = []
for X, Y in data_loader_train:
    X, Y = X.to(device), Y.to(device)
    preload.append((X, Y))
X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
Y_test = mnist_test.targets.to(device)

print(f'Data preloading done in {sw.term():.3f}s\n')

acc_log = []

for ep in range(total_epochs + 1):
    avg_cost = 0

    for X, Y in preload:
        X = X.to(device)
        Y = Y.to(device)

        hypo = model(X)
        cost = criterion(hypo, Y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost

    avg_cost /= total_batch
    epoch_length = sw.term()

    with torch.no_grad():
        prediction = model(X_test)
        correct = torch.argmax(prediction, 1) == Y_test
        acc = correct.float().mean().detach().cpu() * 100

    acc_log.append(acc)

    print(f'E: {ep:2d} C: {avg_cost:.6f} AC: {acc:3.3f}% | TE: {epoch_length:.3f}s')
    sw.term()

plt.plot(acc_log)
plt.show()
