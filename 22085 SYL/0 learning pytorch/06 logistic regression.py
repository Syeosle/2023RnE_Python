import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(1)

'''
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0],    [0],    [0],    [1],    [1],    [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):

    hypo = torch.sigmoid(x_train.matmul(W) + b)
    cost = F.binary_cross_entropy(hypo, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}')
'''
'''
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0],    [0],    [0],    [1],    [1],    [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    hypo = model(x_train)
    cost = F.binary_cross_entropy(hypo, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 20 == 0:
        prediction = hypo >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f} '
              f'Accuracy {accuracy*100:3.2f}%')
'''


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
        self.y_data = [[0], [0], [0], [1], [1], [1]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):

        x_train, y_train = samples

        hypo = model(x_train)
        cost = F.binary_cross_entropy(hypo, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 20 == 0:
            prediction = hypo >= torch.FloatTensor([0.5])
            correct_prediction = prediction.float() == y_train
            accuracy = correct_prediction.sum().item() / len(correct_prediction)
            print(f'Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f} '
                  f'Accuracy {accuracy*100:3.2f}%')

print(model(torch.FloatTensor(dataset.x_data)))
print(torch.FloatTensor(dataset.y_data))