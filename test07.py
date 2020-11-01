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
save_path = r"params/test07.pkl"

class RnnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU()
        )
        self.lstm = nn.LSTM(128,128,2,batch_first=True)
        self.out = nn.Linear(128,10)

    def forward(self, x):
        x = x.reshape(-1,180,120).permute(0,2,1)#NCHW-->N,180,120-->N,120,180
        x = x.reshape(-1,180)#N,120,180-->N,180
        fc1 = self.fc1(x)#N,128
        fc1 = fc1.reshape(-1,120,128)#N,128-->N,120,128
        lstm,(h_n,h_c) = self.lstm(fc1)
        out = lstm[:,-1,:]#NV

        out = out.reshape(-1,1,128)#NV-->N,1,V
        out = out.expand(BATCH_SIZE,4,128)#N,1,V-->N,4,128
        lstm,(h_n,h_c) = self.lstm(out)#N,4,128
        y1 =  lstm.reshape(-1,128)#N,4,128-->N*4,128
        out = self.out(y1)#N*4,10
        out = out.reshape(-1,4,10)#N*4,10-->N,4,10
        return out

if __name__ == '__main__':
    net = RnnNet().cuda()
    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))

    train_data = MyDataset(root="data")
    train_loader = data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,drop_last=True)

    for epoch in range(EPOCH):
        for i,(x,y) in enumerate(train_loader):
            batch_x = x.cuda()
            batcn_y = y.float().cuda()

            out = net(batch_x)

            loss = loss_func(out,batcn_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i%5 == 0:
                test_y = torch.argmax(y,2).detach().cpu().numpy()
                pred_y = torch.argmax(out,2).cpu().detach().numpy()
                acc = np.mean(np.all(pred_y==test_y,axis=1))
                print("epoch:",epoch,"Loss:",loss.item(),"acc:",acc)
                print("test_y:",test_y[0])
                print("pred_y",pred_y[0])
        torch.save(net.state_dict(),save_path)
