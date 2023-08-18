import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 26),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class ReadFromCsv(Dataset):
    def __init__(self, csv_path, train=True, inp=(1, None), outp=(None, 1), train_ratio=0.7):
        df = pd.read_csv(csv_path)
        sz = int(len(df)* train_ratio)
        inp = df.iloc[:, inp[0]:inp[1]].values
        outp = df.iloc[:, outp[0]:outp[1]].values
        
        train_inp, test_inp, train_outp, test_outp = train_test_split(
            inp,
            outp,
            train_size=train_ratio,
            shuffle=True
        )
        
        if train :
            self.inp = train_inp
            self.outp = train_outp
        else :
            self.inp = test_inp
            self.outp = test_outp
	
    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inp = torch.FloatTensor(self.inp[idx])
        outp = torch.LongTensor(self.outp[idx])
        return inp, outp