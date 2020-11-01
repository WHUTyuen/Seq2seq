import os
import numpy as np
import torch
from torch import nn
import  torch.utils.data as data
from data import MyDataset

img_path = "data"
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCH = 100
save_path = r"params/seq2seq.pkl"

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.lstm = nn.LSTM(128,128,2,batch_first=True)
    def forward(self, x):
        x = x.reshape(-1,180,120).permute(0,2,1)
        x = x.reshape(-1,180)
        fc1_out = self.fc1(x)
        fc1_out = fc1_out.reshape(-1,120,128)
        lstm_out,(h_n,h_c)=self.lstm(fc1_out)
        out = lstm_out[:,-1,:]#N,128
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(128,128,2,batch_first=True)
        self.out = nn.Linear(128,10)
    def forward(self, x):
        x = x.reshape(-1,1,128)
        x = x.expand(BATCH_SIZE,4,128)#N,4,128
        lstm_out,(h_n,h_c) = self.lstm(x)#N,4,128
        lstm_out = lstm_out.reshape(-1,128)#N*4,128
        out = self.out(lstm_out)#N*4,10
        out = out.reshape(-1,4,10)#N,4,10
        return out

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        encoder = self.encoder(x)
        out = self.decoder(encoder)
        return out
if __name__ == '__main__':
    net = Net().cuda()
    opt = torch.optim.Adam([{"params":net.encoder.parameters()},{"params":net.decoder.parameters()}])
    loss_func = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    train_data = MyDataset(root="data")
    train_loader = data.DataLoader(train_data,BATCH_SIZE,shuffle=True,drop_last=True,num_workers=NUM_WORKERS)

    for epoch in range(EPOCH):
        for i,(x,y) in enumerate(train_loader):
            bathc_x = x.cuda()
            bathc_y = y.float().cuda()

            out = net(bathc_x)

            loss = loss_func(out,bathc_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 5 == 0:
                test_y = torch.argmax(y, 2).detach().cpu().numpy()
                pred_y = torch.argmax(out, 2).cpu().detach().numpy()
                acc = np.mean(np.all(pred_y == test_y, axis=1))
                print("epoch:", epoch, "Loss:", loss.item(), "acc:", acc)
                print("test_y:", test_y[0])
                print("pred_y", pred_y[0])
        torch.save(net.state_dict(), save_path)