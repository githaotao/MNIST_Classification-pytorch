import torch
import torch.nn as nn

#设置随机种子，保证每次运行结果都相同
torch.manual_seed(2020)
#设计神经网络模型

class MyNetwork(nn.Module):
    def __init__(self, inchannel = 1, kernel = 2, stride = 2):
        super(MyNetwork,self).__init__()
        # inchannel, outchannel, kernel, stride
        self.Conv = nn.Sequential(
            nn.Conv2d(inchannel, 16, kernel, stride),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel, stride),
            nn.ReLU(), 
            )
        self.linear = nn.Sequential(
            nn.Linear(576,64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64,10),
            )

    def forward(self, x):
        out = self.Conv(x)
        out = out.view(-1, 576)
        out = self.linear(out)
        return out

if __name__ == '__main__':
    pass
