import torch 
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
br=os.path.exists('./0')

print(br)
'''
    文件结构 train
                -male
                    -1.jpg
                    -2.jpg
                -female
                    -1.jpg
                    -2.jpg
'''
train_img=torchvision.datasets.ImageFolder(
    root='./train',
    transform=transforms.ToTensor()
)
print(train_img)
print(len(train_img))
print(train_img[0])#第0个样本的 Tensor()+标签
print(train_img[0][0].shape)#第0个样本图像对应的Tensor()的形状
'''
    输出表示前33个都是/male 中的
    后几个都是/female 中的,,,
    打包方式比较规整
    注意自动打好的标签是 [0,1,2,3...],,并不是文件夹的名称
    文件夹信息存储在 classes 中
    for idx,(x,y) in enumerate(train_img):
    print(y)
    if(y==1):
        print(idx)
        break
'''

'其他可调用的方法/属性'
print(train_img.classes)#标签列表,,,就是文件夹的名称

print(np.unique(train_img.targets))
