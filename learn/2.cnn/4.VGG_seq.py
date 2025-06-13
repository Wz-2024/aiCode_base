import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        #表示特征提取部分的Sequential
        self.feature_=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),nn.ReLU(inplace=True),  
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256,512,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512,512,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    
        self.clf_=nn.Sequential(
            nn.Dropout(p=0.5),
            #这里的512*7*7要怎么算呢?可以先将上面的Seqtuential打印出来查看
            nn.Linear(512*7*7,4096),nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,1000),nn.Softmax(dim=1)
        )

    def forward(self,x):
        x=self.feature_(x)#调用第一个Sequential,完成特征提取
        x=x.view(-1,512*7*7)#将特征图展平
        ooutput=self.clf_(x)
        return ooutput
    
model=VGG16()
print(model)

model_info=summary(model,(10,3,224,224),device=device)
