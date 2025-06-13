import torch
import torch.nn as nn
from torchvision import models as m



vgg16_bn_ = m.vgg16_bn()
resnet18_=m.resnet18()
class MyNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block2=vgg16_bn_.features[7:14]
        self.block3=resnet18_.layer3
        self.avgpool=resnet18_.avgpool
        self.fc=nn.Linear(in_features=256,out_features=10,bias=True)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.block3(self.block2(x))
        x=self.avgpool(x)
        x=x.view(-1,256)
        x=self.fc(x)
        return x

    

if __name__=='__main__':
    model=MyNet1()
    data=torch.ones(10,1,32,32)
    data=model(data)
    print(data)
    

