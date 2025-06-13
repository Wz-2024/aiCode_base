import torch
import torch.nn as nn


device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data=torch.ones((10,3,229,229))

'''
    首先明确一点,nn.Sequential是一个类,可以包含多个神经网络层,
    并且这神经网络就是顺次执行的,
    以前的写法中,relu只在forward中写,在Sequential中,直接按顺序来写
 
    inplace=True表示在原数据上进行操作,否则会创建新的数据
    我们的目标是让一个 X 流过这些网络,因此这里要写成True

    此外,用nn.Sequential这种写法,代码的组织思路一般是按块的(块内像是一种执行流的组织)
    以CNN代码来讲,可以分为 feature,fc 两个大Sequential块,
     
'''
net = nn.Sequential(
        nn.Conv2d(3,6,3)
       
        ,nn.ReLU(inplace=True)
        ,nn.Conv2d(6,4,3)
        ,nn.ReLU(inplace=True)
        ,nn.MaxPool2d(2)
        ,nn.Conv2d(4,16,5,2,1)
        ,nn.ReLU(inplace=True)
        ,nn.Conv2d(16,3,5,3,2)
        ,nn.ReLU(inplace=True)
        ,nn.MaxPool2d(2)
)

data=net(data)
print(data.shape)