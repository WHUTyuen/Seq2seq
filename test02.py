import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(28,64,1,batch_first=True)
        self.fc = nn.Linear(64,10)

    def forward(self, x):
        x = x.reshape(-1,28,28)
        h0 = torch.zeros(1,x.shape[0],64)

        out_put,_ = self.rnn(x,h0)
        out = out_put[:,-1,:]
        out = self.fc(out)
        return torch.softmax(out,dim=1)