import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision #包含数据的模块
import torchvision.transforms as transforms #用来处理数据的模块
import matplotlib.pyplot as plt
import numpy as np

#实例化数据
mnist=torchvision.datasets.FashionMNIST(
    root='./data', #数据存放位置
    train=True, #True则表示用训练集(很大),False则表示用测试集(小)
    download=True, #是否下载数据,,如果在给定的路径下找到了数据集,那就不会再下载了
    transform=transforms.ToTensor() #在数据集导入后进行统一的处理,都转化为torch能够处理的张量
)
def print_message():
    print('直接打印mnist')
    print(mnist)
    print('----')
    print('length,即样本数',len(mnist))
    '''
        对于图像数据集来讲,shape一般有四个维度
        sample_size:样本数
        H-height:图像高度
        W-width:图像宽度
        C-channel:图像通道数

        但是对于Mnist中的数据来讲,它是单通道的[60000,28,28],神经网络无法识别三维的,因此需要reshape一下
    '''
    print('----')
    print('shape:',mnist.data.shape)
    print('----')
    print('目标分类',mnist.targets.unique())
    print('----')
    print('打印分类类别的具体意义',mnist.classes)
def show_image():
    '''
        这里需要解释,plt是py的原生库,torch作为第三方库,它的tensor不能被plt识别
        因此需要先将tensor转化为numpy
    '''
    mnist[0]#中有两个元素,第一个是(转化为tensor的)图片,第二个是标签
    print('图像的形状',mnist[0][0].shape)
    plt.imshow(mnist[0][0].view(28,28).numpy())
    # plt.savefig('./image.png')

if __name__=='__main__':
    # print_message()
    show_image()