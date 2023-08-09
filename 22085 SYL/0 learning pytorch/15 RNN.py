import torch
import torch.nn as nn

input_size = 5
hidden_size = 8


# (batch size, time steps, input size)
inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)

outputs, _status = cell(inputs)

print(_status.shape)

# ================

inputs = torch.Tensor(1, 10, 5)
cell = nn.RNN(input_size=5, hidden_size=8, num_layers=2,
              batch_first=True, bidirectional=True)

outputs, _status = cell(inputs)

print(outputs.shape)

# ================

# GRU 또한 RNN 함수와 같은 방법으로 쓸 수 있다.
# 일반적으로 데이터가 많은 경우 LSTM, 적은 경우 GRU를 쓰긴 한다
