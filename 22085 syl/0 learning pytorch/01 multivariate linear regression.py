import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련 데이터
x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])


# 가중치 w와 편향 b 초기화
w = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w, b], lr=1e-5)

nb_epochs = 20000
for epoch in range(nb_epochs + 1):

    hypo = x_train.matmul(w) + b
    cost = torch.mean((hypo - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'{epoch:4d}/{nb_epochs:4d} Hypo: {hypo.squeeze().detach()} Cost: {cost.item():.6f}')
