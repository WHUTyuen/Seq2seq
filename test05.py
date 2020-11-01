import torch
import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torch.nn as nn
import torch.utils.data as data

class RnnNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn_layer = nn.LSTM(28,64,1,batch_first=True)
        self.output_layer = nn.Linear(64,10)

    def forward(self, x):
        input = x.reshape(-1,28,28)#NCHW-->NSV

        h0 = torch.zeros(1,1000,64)
        c0 = torch.zeros(1,1000,64)
        outputs,(hn,cn) = self.rnn_layer(input,(h0,c0))
        #outputs--->NSV我们往线性层输入的是最后一个S的V
        output = outputs[:,-1,:]#只要NSV的S的最后一个输出：NV
        output = self.output_layer(output)
        return output

if __name__ == '__main__':
    train_dataset = dataset.MNIST(root='/data',train=True,transform=transforms.ToTensor(),download=True)
    train_dataloader = data.DataLoader(train_dataset,batch_size=1000,shuffle=True)

    test_dataset = dataset.MNIST(root='/data', train=False, transform=transforms.ToTensor())
    test_dataloader = data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

    net = RnnNet()
    opt = torch.optim.Adam(net.parameters())

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10000):
        for xs,ys in train_dataloader:
            output = net(xs)
            opt.zero_grad()
            loss = loss_fn(output,ys)
            loss.backward()
            opt.step()
        print(loss.item())
        for xs,ys in test_dataloader:
            test_output = net(xs)
            test_ids = torch.argmax(test_output,dim=1)
            print(torch.mean(torch.eq(test_ids, ys).float()))