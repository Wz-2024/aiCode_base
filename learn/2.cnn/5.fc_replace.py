import torch
import torch.nn as nn
from torchinfo import summary

data=torch.ones((10,3,32,32))

'''
    这里主要用到了一个1*1卷积核,主要是熟悉nn.Sequential()的写法
'''

class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        '''

        '''
        self.block1=nn.Sequential(
            nn.Conv2d(3,192,5,padding=2),nn.ReLU(inplace=True)
            ,nn.Conv2d(192,160,1),nn.ReLU(inplace=True)
            ,nn.Conv2d(160,96,1),nn.ReLU(inplace=True)
            ,nn.MaxPool2d(kernel_size=3,stride=2)
            ,nn.Dropout(0.25)
        )

        self.block2=nn.Sequential(
            nn.Conv2d(96,192,5,padding=2),nn.ReLU(inplace=True)
            ,nn.Conv2d(192,192,1),nn.ReLU(inplace=True)
            ,nn.Conv2d(192,192,1),nn.ReLU(inplace=True)
            ,nn.MaxPool2d(kernel_size=3,stride=2)
            ,nn.Dropout(0.25)
        )

        self.block3=nn.Sequential(
            nn.Conv2d(192,192,3,padding=1),nn.ReLU(inplace=True)
            ,nn.Conv2d(192,192,1),nn.ReLU(inplace=True)
            ,nn.Conv2d(192,10,1),nn.ReLU(inplace=True)
            #全局平均池化
            ,nn.AvgPool2d(7,stride=1)
            ,nn.Softmax(dim=1)
        )

    def forward(self,x):
        output=self.block3(self.block2(self.block1(x)))    
        return output
    
if __name__=='__main__':
    model=NiN()
    data=model(data)
    #输出表示是10个特征图,每个特征图的尺寸是1*1
    print(data.shape)
    info=summary(model,input_size=(10,3,32,32))
    print(info)
