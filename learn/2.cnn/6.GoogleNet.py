import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

"""

    基础卷积块,inception,辅助分类器
    这三个模块是串联的,因此可以用nn.Sequential()串联

    考虑参数的问题:
        现在用到的卷积层,不仅是普通的Conv2d(),而是自定义的,复合的卷积层,BasicConv2d()
        与之前最大的不同是,BasicConv2d()是需要复用的,参数不能写死,应该留出传入参数的接口

        因此可能需要一些参数,这些参数都应该写进‘类的参数当中’
        (所谓'写进类的参数中'表示应该在构造函数说明清楚需要用到的所有参数)
        但是参数太多,如果不确定当前到底需要几个参数,可以用**kwargs代替
        如果__init__()用到的参数和其中某个方法的参数几乎一致,则可以直接用**kwargs代替

"""


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),  # 这里应该填上一个卷积层的输出
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,  # 3x3卷积核,用来降采样
        ch3x3: int,  # 3x3卷积核,用来提取特征
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ):
        super().__init__()
        """
            这里需要有四个分支,由于是并行的,因此四个分支肯定不直接都包在一个Sequential里面,而是要分别定义四个分支,
            然后再通过nn.ModuleList来组合成一个Module

            需要说明的是,由于这几个branch是并联的,因此他们的输入应该保持一致,即in_channels应该是一致的
            但是输出很可能是不一样的,这个比较麻烦,因此每个输出都需要一个独立的参数来控制

        """
        # 1*1
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1*1+3*3
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )
        # 1*1+5*5
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )
        # pool+1*1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1,ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        # 对于这个cat()函数,如果维度1一样那就在维度2上合并
        return torch.cat(outputs, 1)  # 并行操作后拼接在一起


# 辅助分类器 auxiliary classifier
class AuxClf(nn.Module):
    def __init__(self,in_channels,num_classes,**kwargs):
        super().__init__()

        '''
            注意,这里卷积和 fc虽然是串联的,但是要把提取特征和分类分来
            (详见之前讨论过的Sequential可以解决计算维度的问题)
        '''

        self.feature_ = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3)
            ,BasicConv2d(in_channels,128,kernel_size=1)
        )
        self.clf_=nn.Sequential(
            nn.Linear(4*4*128,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024,num_classes)
        )

    def forward(self,x):
        x=self.feature_(x)
        x=x.view(-1,4*4*128)
        x=self.clf_(x)
        return x

'''
    上边是GooGLeNet的所有组件,
    接下来开始组装真正的GoogleNet
'''

class GoogLeNet(nn.Module):
    def __init__(self,num_classes:int=1000,blocks=None):
        super().__init__()

        '''
            假设下面的模块是可替换的,比如 现在使用的是inception,如果后期有更好的inceptionV3可以代替
            那逐个修改就会比较费劲
            这样的话可以用一个列表来维护这些类的名称,,后期只需要修改列表中的元素即可
        '''
        if blocks is None:
            blocks=[BasicConv2d,Inception,AuxClf]

        conv_block=blocks[0]
        inception_block=blocks[1]
        auxclf_block=blocks[2]


        #block1
        self.conv1=conv_block(3,64,kernel_size=7,stride=2,padding=3)
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)#表示向上取整

        #block2
        self.conv2=conv_block(64,64,kernel_size=1)
        self.conv3=conv_block(64,192,kernel_size=3,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        #block3
        self.inception3a=inception_block(192,64,96,128,16,32,32)
        self.inception3b=inception_block(256,128,128,192,32,96,64)
        self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        #block4
        self.inception4a=inception_block(480,192,96,208,16,48,64)
        self.inception4b=inception_block(512,160,112,224,24,64,64)
        self.inception4c=inception_block(512,128,128,256,24,64,64)
        self.inception4d=inception_block(512,112,144,288,32,64,64)
        self.inception4e=inception_block(528,256,150,320,32,128,128)
        self.maxpool4=nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)

        #block5
        self.inception5a=inception_block(832,256,160,320,32,128,128)
        self.inception5b=inception_block(832,384,192,384,48,128,128)

        #clf 
        #全局平均池化
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))#需要的特征图尺寸是多少
        self.dropout=nn.Dropout(0.4)
        self.fc=nn.Linear(1024,num_classes)

        # 辅助分类器
        self.aux1=auxclf_block(512,num_classes)#4a
        self.aux2=auxclf_block(528,num_classes)

    def forward(self,x):
        #block1
        x=self.maxpool1(self.conv1(x))


        #block2
        x=self.maxpool2(self.conv3(self.conv2(x)))


        #block3
        x=self.inception3a(x) 
        x=self.inception3b(x)
        x=self.maxpool3(x)


        #block4
        x=self.inception4a(x)
        aux1=self.aux1(x)

        x=self.inception4b(x)
        x=self.inception4c(x)
        x=self.inception4d(x)
        aux2=self.aux2(x)
        x=self.inception4e(x)
        x=self.maxpool4(x)

        #block5
        x=self.inception5a(x)
        x=self.inception5b(x)

        #clf
        x=self.avgpool(x)#在全局平均池化之前,特征图就变成1*1了
        x=torch.flatten(x,1)
        x=self.dropout(x)
        x=self.fc(x)
        return x,aux2,aux1






if __name__ == "__main__":
    # 注意,这里暂且不需要张量,因为只有前向传播时候才需要
    convolution = BasicConv2d(2, 10, kernel_size=3)
    print(convolution)
    inception = Inception(192, 64, 96, 128, 16, 32, 32)
    data = torch.ones(100, 192, 28, 28)  # 样本量,通道数,28*28的特征
    data = inception(data)
    print(data.shape)
    auxClf=AuxClf(512,1000)
    #表示4a后的辅助分类器
    print(auxClf)


    data=torch.ones(10,3,224,224)
    model=GoogLeNet(num_classes=1000)
    fc2,fc1,fc0=model(data)
    for i in [fc2,fc1,fc0]:
        print(i.shape)
    
    summary(model,(10,3,224,224))

