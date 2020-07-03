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
batch_size = 256
eval_batch = 128
EPOCH = 500
use_cuda = torch.cuda.is_available()
cuda_num = 0
learning_rate = 0.001
#对图像转换
transform = T.Compose([
    #PIL默认读取bmp图像为4通道，这里转为灰度图像，变为1通道
    T.Lambda(lambda img: img.convert("L")),
    T.RandomCrop(24),
    #T.RandomHorizontalFlip(),
    T.ToTensor(),
    #ToTensor的值在（0,1）之间，用下面的Normalize转为（-1,1）之间
    T.Normalize(mean = [0.5], std = [0.5])
    ])
train_dataset = ImageFolder(root = './dataset/train/', transform = transform)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

eval_dataset = ImageFolder(root = './dataset/eval/', transform = transform)
eval_loader = DataLoader(eval_dataset, batch_size = eval_batch, shuffle = False)

model = MyNetwork()
#更改continue_flag为True使网络接着训练，否则改为False
continue_flag = True
if continue_flag:
    model.load_state_dict(torch.load('./best_model.pth'))
loss_func = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim = 1)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    train_loss_best = 99999
    eval_accuracy_best = 0
    if use_cuda:
        model.cuda(cuda_num)
    model.train()
    for epoch in range(EPOCH):
        for _, (imgs, labels) in enumerate(train_loader):
            if use_cuda:
                imgs = imgs.cuda(cuda_num)  
                labels = labels.cuda(cuda_num) 
            output = model(imgs)
            loss = loss_func(output, labels)
            if loss.item() < train_loss_best:
                train_loss_best = loss.item()
                print('epoch:',epoch,' train_best_loss:',train_loss_best)
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

        #在验证集上测量损失，保存最好结果
        acc_num = 0
        all_num = 0
        model.eval()
        for _, (imgs, labels) in enumerate(eval_loader):
            if use_cuda:
                imgs = imgs.cuda(cuda_num)  
                labels = labels.cuda(cuda_num) 
            output = model(imgs)
            output = softmax(output)
            output = output.argmax(dim = 1)
            acc_num += (output == labels).sum().cpu().item()
            all_num += len(labels)
        accuracy = acc_num / all_num

        if accuracy > eval_accuracy_best:
            torch.save(model.state_dict(), './best_model.pth')
            eval_accuracy_best = accuracy
            print('epoch:',epoch,' best_accuracy:',accuracy)

    print('Finish')


if __name__ == '__main__':
    train()
