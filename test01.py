import torch.nn as nn
import torch

rnn = nn.RNN(7, 3, 2,batch_first=True,bias=False)
# input = torch.randn(50, 30, 10)
input  = torch.arange(1,29).reshape(1,4,7).float()
# h0 = torch.randn(6, 50, 5)
output, hn = rnn(input)
print(output.shape)
# print(hn.shape)
# print(input)
# print(output)
# print(hn)
for i in rnn.parameters():
    print(i)