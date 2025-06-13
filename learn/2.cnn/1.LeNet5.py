import torch
from torch import nn
#在糕端写法中,其实不用F
from torch.nn import functional as F
from torchinfo import summary

data=torch.ones(size=(10,1,32,32))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,5)
        '''
            这里本应该填kernel_size=2,stride=2,
            只填一个2表示卷积核大小为2 , 步长自动与卷积核大小一致
        '''
        self.pool1=nn.AvgPool2d(2)
        self.conv2=nn.Conv2d(6,16,5)
        self.pool2=nn.AvgPool2d(2)
        #与线性层链接的时候,线性层接收的是特征图中全部像素拉平到一维的,因此是通道数*高*宽
        self.fc1=nn.Linear(5*5*16,120)
        self.fc2=nn.Linear(120,84)
    def forward(self,x):
        x=self.pool1(F.tanh(self.conv1(x)))
        x=self.pool2(F.tanh(self.conv2(x)))
        #这里需要展平
        x=x.view(-1,5*5*16)
        x=F.tanh(self.fc1(x))#samples,Features
        output=F.softmax(self.fc2(x),1)
    
'''
    torchinfo可以显示模型的结构,以及参数
'''

if __name__=="__main__":
    model=Model()
    model(data)#相当于执行了forward,本质上是执行了__call__方法
    print(model)
    '打印得非常详细'
    print(summary(model,input_size=(10,1,32,32)))

        