import torch
import torch.nn as nn

from model import MyNetwork
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
#设置随机种子，保证每次运行结果都相同
torch.manual_seed(2020)

#超参数
#测试样本总个数
total_test = 1000
use_cuda = torch.cuda.is_available()
cuda_num = 0
#对图像转换
transform = T.Compose([
    #PIL默认读取bmp图像为4通道，这里转为灰度图像，变为1通道
    T.Lambda(lambda img: img.convert("L")),
    T.CenterCrop(24),
    #T.RandomHorizontalFlip(),
    T.ToTensor(),
    #ToTensor的值在（0,1）之间，用下面的Normalize转为（-1,1）之间
    T.Normalize(mean = [0.5], std = [0.5])
    ])
test_dataset = ImageFolder(root = './dataset/test/', transform = transform)
test_loader = DataLoader(test_dataset, batch_size = total_test, shuffle = False)

model = MyNetwork()
model.load_state_dict(torch.load('./best_model.pth'))

softmax = nn.Softmax(dim = 1)

def test():
    if use_cuda:
        model.cuda(cuda_num)
    model.eval()
    acc_num = 0
    all_num = 0 
    for _, (imgs, labels) in enumerate(test_loader):
        if use_cuda:
            imgs = imgs.cuda(cuda_num)  
            labels = labels.cuda(cuda_num) 
        output = model(imgs)
        output = softmax(output)
        output = output.argmax(dim = 1)
        acc_num += (output == labels).sum().cpu().item()
        all_num += len(labels)
    accuracy = acc_num / all_num
    print('准确率为：', accuracy)

if __name__ == '__main__':
    test()