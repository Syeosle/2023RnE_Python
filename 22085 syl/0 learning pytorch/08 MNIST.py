import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else "cpu")
print('Device set to', device)

random.seed(1)
torch.manual_seed(1)
if USE_CUDA: torch.cuda.manual_seed_all(1)

training_epochs = 15
batch_size = 512

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

preload_data = []

t_p = time.time()
for X, Y in data_loader:
    X = X.view(-1, 28 * 28).to(device)
    Y = Y.to(device)
    preload_data.append((X, Y))

print(f'preloading: {time.time() - t_p}s')

X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
Y_test = mnist_test.test_labels.to(device)
acc_logging = []

linear = nn.Linear(784, 10, bias=True).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(linear.parameters(), lr=0.1)

for ep in range(training_epochs + 1):
    avg_cost = 0
    total_batch = len(data_loader)
    t_l = time.time()

    for X, Y in preload_data:
        t_d = time.time()
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)

        t_XY = time.time()
        hypo = linear(X)
        cost = criterion(hypo, Y)

        t_m = time.time()
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        t_o = time.time()
        avg_cost += cost / total_batch

    with torch.no_grad():
        prediction = linear(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        acc_logging.append(accuracy.item() * 100)

    # print(f'E: {ep+1:4d} C: {avg_cost:.6f} A: {acc_logging[-1]:3.2f} | T: d:{t_d-t_l:.3f}s '
    #       f'X:{t_XY-t_d:.3f}s m:{t_m-t_XY:.3f}s o:{t_o-t_m:.3f}s')

print('Learning finished!')

plt.plot(acc_logging)
plt.show()

'''
# testing
with torch.no_grad():
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # random picking test
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
'''