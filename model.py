import torch
import torch.nn as nn

#设置随机种子，保证每次运行结果都相同
torch.manual_seed(2020)
#设计神经网络模型

class MyNetwork(nn.Module):
    def __init__(self, inchannel = 1, outchannel = 1, kernel = 2, stride = 2):
        super(MyNetwork,self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel, stride),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel, kernel, stride),
            nn.ReLU(), 
            )
        self.linear = nn.Sequential(
            nn.Linear(36,12),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(12,10),
            )

    def forward(self, x):
        out = self.Conv(x)
        out = out.view(-1, 36)
        out = self.linear(out)
        return out

if __name__ == '__main__':
    pass