import torch.nn as nn
from torchinfo import summary
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch


class DoubleConv2d(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,3,1,0,bias=False)
                                 ,nn.BatchNorm2d(out_channels)
                                 ,nn.ReLU(inplace=True)
                                 ,nn.Conv2d(out_channels, out_channels,3,1,0,bias=False)
                                 ,nn.BatchNorm2d(out_channels)
                                 ,nn.ReLU(inplace=True)
                                 )
    def forward(self,x):
        return self.conv(x)
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        #下采样(Encoder部分),包含四个双卷积层

        #4个块
        self.encoder_conv = nn.Sequential(DoubleConv2d(1,64)
                                          ,DoubleConv2d(64,128)
                                          ,DoubleConv2d(128,256)
                                          ,DoubleConv2d(256,512)
                                         )
        #每个块之间用pool链接
        self.encoder_down = nn.MaxPool2d(2)
    
        #上采样
        #4个块
        self.decoder_up = nn.Sequential(nn.ConvTranspose2d(1024,512,4,2,1)
                                       ,nn.ConvTranspose2d(512,256,4,2,1)
                                       ,nn.ConvTranspose2d(256,128,4,2,1)
                                       ,nn.ConvTranspose2d(128,64,4,2,1)
                                       )
        #每个块之间还得反卷积,每个反卷积的输入输出不同,所以得定义Sequential
        self.decoder_conv = nn.Sequential(DoubleConv2d(1024,512)
                                          ,DoubleConv2d(512,256)
                                          ,DoubleConv2d(256,128)
                                          ,DoubleConv2d(128,64)
                                         )
        
        self.bottleneck = DoubleConv2d(512,1024)
        
        self.output = nn.Conv2d(64,2,3,1,1)
    
    def forward(self,x):
        
        #encoder：保存每一个DoubleConv的结果为跳跃链接做准备，同时输出codes
        skip_connection = []
        
        for conv in self.encoder_conv:
            x = conv(x)
            skip_connection.append(x)
            x = self.encoder_down(x)
        
        x = self.bottleneck(x)
        
        #刚才从上到下存下来的,decoder是从下到上,因此调换顺序
        skip_connection = skip_connection[::-1]
        
        #decoder：codes每经过一个转置卷积，就需要与跳跃链接中的值合并
        #合并后的值进入DoubleConv
        
        for idx in range(4):
            #反卷积
            x = self.decoder_up[idx](x)
            #跳跃链接
            skip_connection[idx] = transforms.functional.resize(skip_connection[idx],size=x.shape[-2:])
            x = torch.cat((skip_connection[idx],x),dim=1)
            #双层卷积
            x = self.decoder_conv[idx](x)
        
        x = self.output(x)
        return x
    
if __name__=="__main__":
    model=Unet()
    summary(model,input_size=(10,1,572,572),device="cuda")
