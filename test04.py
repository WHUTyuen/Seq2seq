import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNNCell(28,64)
        self.fc = nn.Linear(64,10)
    def forward(self, x):
        x = x.reshape(-1,28,28)
        hx = torch.zeros(x.shape[0],64)

        for s in range(28):
            input = x[:,s,:]
            h = self.rnn(input,hx)
        output = self.fc(h)
        return output