import torch
import torch.nn as nn
from typing import List, Union , Type, Optional
from torchinfo import summary

#basic_conv - Conv2d + BatchNorm2d + ReLU  ---——>(3x3的一个  1x1的一个)
#Residual Unit, Bottlenck
'''
    先定义以上的四个类,最终希望讲他们组织起来从而搭建起残差网络
'''


def conv3x3(in_,out_,stride=1,initialzero=False):
    bn=nn.BatchNorm2d(out_)
    #需要进行判断:是否需要对BN进行初始化? 是最后一层就初始化,不是则不需要改变gamma,beta
    if initialzero==True:
        nn.init.constant_(bn.weight,0)#gamma=0
    return nn.Sequential(
        nn.Conv2d(in_, out_, kernel_size=3, stride=stride, padding=1,bias=False),
        bn
    )

def conv1x1(in_,out_,stride=1,initialzero=False):
    bn=nn.BatchNorm2d(out_)
    #需要进行判断:是否需要对BN进行初始化? 是最后一层就初始化,不是则不需要改变gamma,beta
    if initialzero==True:
        nn.init.constant_(bn.weight,0)#gamma=0
    return nn.Sequential(
        nn.Conv2d(in_, out_, kernel_size=1, stride=stride, padding=0,bias=False),
        bn
    )


#残差单元
class ResidualUnit(nn.Module):
    def __init__(self,out_,stride1:int=1,in_:Optional[int]=None):
        super().__init__()
        #这里需要判断stride1是否等于2,如果为2,则特征图尺寸会发生变化
        #需要在 skip connection 中需要增加1x1卷积层来调整特征图的尺寸
        #若为1则无需处理
        self.stride1=stride1

        #当特征图尺寸需要缩小时,卷积层的输出特征图数量_out等于输入特征图的数量
        #当特征图尺寸不需要缩小时,out_==in_
        if stride1 != 1:
            in_=int(out_/2)
        else: in_=out_


        #拟合部分,输出F(x)
        self.fit_=nn.Sequential(
            conv3x3(in_,out_,stride=stride1),
            nn.ReLU(inplace=True),
            conv3x3(out_,out_,initialzero=True),
        )
        #跳跃连接部分,输出H(x)
        self.skipconv=conv1x1(in_,out_,stride=stride1)

        #单独定义放在H(x)后来使用的激活函数ReLU
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        #拟合结果
        fx=self.fit_(x)
        if self.stride1!=1:
            x=self.skipconv(x)#这里需要增加一个1x1卷积层来调整特征图的尺寸
        '''
            这部分表示两部分的加和,,
            但是,这里必须要求fx和x的维度一样,只要不一样,就无法完成点对点,像素对像素的加和
            其实torch中只要求列一样
        '''
        hx=self.relu(fx+x)
        return hx


#瓶颈结构
class Bottleneck(nn.Module):
    def __init__(self,middle_out,stride1:int =1,in_ :Optional[int]=None):
        super().__init__()
        out_=4*middle_out

        '''
            如果是特征图的尺寸需要缩小的场合
            即 conv2_x - conv3_x - conv4_x - conv5_x
            每次都需要将特征图尺寸折半,同时卷积层上的middle_out=1/2 in_
        '''
        #in_这个option的参数是用来判断当前处于架构的哪一部分的
        if in_==None:#这里其实只有64 和 None两种
            if stride1!=1:   #缩小特征图的场合,是每个layers的第一个瓶颈结构
                in_=middle_out*2
            else:           #不缩小特征图的
                in_=middle_out*4
     


        self.fit_=nn.Sequential(
            conv1x1(in_,middle_out,stride=stride1)
            ,nn.ReLU(inplace=True)
            ,conv3x3(middle_out,middle_out)
            ,nn.ReLU(inplace=True)
            ,conv1x1(middle_out,out_,initialzero=True)
        )
        self.skipconv=conv1x1(in_,out_,stride=stride1)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        fx=self.fit_(x)
        x=self.skipconv(x)
        hx=self.relu(fx+x)
        return hx

class ResNet(nn.Module):
    def __init__(self,
                 block:Type[Union[ResidualUnit,Bottleneck]],
                 layers:List[int],
                 num_classes:int
                ):
        super().__init__()
        #layer1 卷积+池化 组合
        self.layer1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
        )
        #layer2 - layer5:残差块/瓶颈结构
        self.layer2_x=make_layers(block,64,num_blocks=layers[0],afterconv1=True)
        self.layer3_x=make_layers(block,128,num_blocks=layers[1])
        self.layer4_x=make_layers(block,256,num_blocks=layers[2])
        self.layer5_x=make_layers(block,512,num_blocks=layers[3])

        #全局平均池化
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))


        #全连接层的layer 并不通用
        if block==ResidualUnit:
            self.fc=nn.Linear(512,num_classes)
        else:
            self.fc=nn.Linear(2048,num_classes)
    def forward(self,x):
        #Layer1 普通卷积+池化 的输出
        x=self.layer1(x)
        #Layer2-5
        x=self.layer5_x(self.layer4_x(self.layer3_x(self.layer2_x(x))))
        #全局平均池化
        x=self.avgpool(x)
        x=torch.flatten(x,1)#表示从第一维度开始展平(0维度是样本维度,不需要动)
        x=self.fc(x)

   


#使用统一的函数来打包层--因为他们有相同的特征,第一层特殊,其他层完全一样
def make_layers(
        block: Type[Union[ResidualUnit,Bottleneck]],
        middle_out: int,
        num_blocks: int,
        afterconv1:bool=False,
        ):
    layers=[]
    if afterconv1 is True:
        layers.append(block(middle_out,in_=64))
    else:
        layers.append(block(middle_out,stride1=2))
    
    for i in range(num_blocks-1):
        layers.append(block(middle_out))
    '''
    此时layer中包裹的是很多歌实体化好的类,,
    接下来应该给这些类中传入x,完成__call__,也就是forward()

    具体应该怎么传入x?
    解包->利用nn.Sequential()打包这些类,打包成一个seq=Sequential(),
    再把x传给seq,即seq(x),,这就完成了forward()

    因此这里直接用nn.Sequential()返回
    这样以来,返回的就是一个网络类,传入x就能完成这个网络类的forward()
    '''
    return nn.Sequential(*layers)


if __name__ == '__main__':
    num_blocks_conv3x = 4
    ru0=ResidualUnit(out_=128,stride1=2)
    for i in range(num_blocks_conv3x-1):
        pass
    
    conv2_x_101=make_layers(Bottleneck,64,3,afterconv1=True)
    datashape=(10,64,56,56)
    print(summary(conv2_x_101,datashape,depth=1,device='cpu'))
    conv4x_101=make_layers(Bottleneck,256,3,afterconv1=True)
    print(summary(conv4x_101))

    print('--------------------------')
    datashape=(10,3,224,224)
    #建一个34层的,,在建个101层的
    res32=ResNet(ResidualUnit,[3,4,6,3],num_classes=1000)
    res101=ResNet(Bottleneck,[3,4,23,3],num_classes=1000)
    print(summary(res32,datashape,depth=1,device='cpu'))
    print(summary(res101,datashape,depth=1,device='cpu'))