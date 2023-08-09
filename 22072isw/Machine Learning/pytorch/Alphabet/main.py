import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from classes import NeuralNetwork, ReadFromCsv

device = 'cuda'
learning_rate = 1e-4
shuffle = True
batch_size = int(input("Enter Batch Size : "))
epochs = int(input("Enter Epochs : "))

training_data = ReadFromCsv(
    csv_path='./data/AlphabetData.csv',
    train = True
)

test_data = ReadFromCsv(
    csv_path='./data/AlphabetData.csv',
    train = False
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

model = NeuralNetwork()

saved_path = 'model_weights.pth'
if os.path.exists(saved_path) :
    model.load_state_dict(torch.load(saved_path))
    model.train()
    
model.to(device)
    
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        
        # 예측(prediction)과 손실(loss) 계산
        X, y = X.to(device), y.to(device)
        pred, y = model(X), torch.flatten(y)
        loss = loss_fn(pred, y)
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % (10000 // batch_size) == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred, y = model(X), torch.flatten(y)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Saved!")
    test_loop(test_dataloader, model, loss_fn)
print("Done!")