import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from skimage import io
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import os

def plotsample(data):
    fig,axs=plt.subplots(1,5,figsize=(10,10))
    for i in range(5):
        num=random.randint(0,len(data)-1)#随机一个数
        #抽取数据中对应的图像对象,make_grid可以将任意格式的图像通道数提升为3
        npimg=torchvision.utils.make_grid(data[num][0]).numpy()
        nplable=data[num][1]#读取标签
        #将图像由(3,h,w)变为(h,w,3)
        npimg=np.transpose(npimg,(1,2,0))
        axs[i].imshow(npimg)
        axs[i].set_title(nplable)
        axs[i].axis('off')

class CustomDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):#root_dir下存的是png图像
        super().__init__()
        #注意这里应当是个性化的
        self.identity=pd.read_csv(csv_file,sep=' ',header=None)
        self.root_dir=root_dir
        self.transform=transform
    def __len__(self):
        return len(self.identity)
    
    def __info__(self):
        print("\t number of samples:P{}".format(len(self.identity)))
        print("\t number of lable:{}".format(self.identity.iloc[:,1].nunique()))
        print('\t root_dir:{}'.format(self.root_dir))



    def __getitem__(self,idx):
        #这里需要保证一下idx不是一个张量
        if torch.is_tensor(idx):
            idx=idx.tolist()
        img_dic=os.path.join(self.root_dir,self.identity.iloc[idx,0])
        img=io.imread(img_dic)
        lable=self.identity.iloc[idx,1]
        if self.transform:
            img=self.transform(img) 


        sample=(img,lable)
        return sample

class AttributeDataset(Dataset):
    def __init__(self, csv_file, root_dir, labelname, transform=None):   
        super().__init__()
        self.root_dir = root_dir
        self.labelname = labelname
        self.transform = transform
        
        # 在初始化时解析 CSV 文件
        attr = pd.read_csv(csv_file, header=None)
        # 提取列名（属性名）和数据行
        attrs = attr.iloc[0, 0].split()  # 第一行是列名
        data_rows = attr.iloc[1:, 0].str.split().tolist()  # 剩余行是数据
        self.attr = pd.DataFrame(data_rows, columns=attrs)  # 创建 DataFrame

    def __len__(self):
        return len(self.attr)

    def __info__(self):
        print('Number of Samples: {}'.format(len(self.attr)))
        print('Root Dir: {}'.format(self.root_dir))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 直接使用预解析的 DataFrame
        imgdic = os.path.join(self.root_dir, self.attr.iloc[idx, 0])  # 第一列是图像文件名
        image = io.imread(imgdic)
        lable = int(self.attr.loc[idx, self.labelname])  # 获取指定属性的标签

        if self.transform:
            image = self.transform(image)
        
        sample = (image, lable)
        return sample


def identity_():
    img_path=r'/data_disk/dyy/python_projects/bili_dif/data2/datasets4/picturestotensor/celebAsubset/Img/Img_celeba.7z/img_celeba'
    csv_path=r'/data_disk/dyy/python_projects/bili_dif/data2/datasets4/picturestotensor/celebAsubset/Anno/identity_CelebA_1000.txt'
 
    
    data=CustomDataset(csv_file=csv_path,root_dir=img_path,transform=transforms.ToTensor())
    print(len(data))
    # print(data[0][0])
    data.__info__()

def test():
    img_path=r'/data_disk/dyy/python_projects/bili_dif/data2/datasets4/picturestotensor/celebAsubset/Img/Img_celeba.7z/img_celeba'
    csv_path=r'/data_disk/dyy/python_projects/bili_dif/data2/datasets4/picturestotensor/celebAsubset/Anno/list_attr_celeba_1000.txt'
    #csv
    attr=pd.read_csv(csv_path,header=None)
    print('长度为',len(attr))
    #现在读进来的情况非常非常蛋疼,是n行一列的
    #接下来,应该取出每一行,然后让这一行按照空格split()
    attrs=attr.iloc[0][0].split()
    lists=attr.iloc[1:,0].str.split().tolist()
    # print(lists)
    attr=pd.DataFrame(lists,columns=attrs)
    print(attr)
    print(attr.loc[:,'Arched_Eyebrows'])

def attribute():
    img_path=r'/data_disk/dyy/python_projects/bili_dif/data2/datasets4/picturestotensor/celebAsubset/Img/Img_celeba.7z/img_celeba'
    csv_path=r'/data_disk/dyy/python_projects/bili_dif/data2/datasets4/picturestotensor/celebAsubset/Anno/list_attr_celeba_1000.txt'

    data=AttributeDataset(csv_file=csv_path,root_dir=img_path,labelname='Attractive',transform=transforms.ToTensor())

    print(data[500][0])

    print(data[500][1])#这个展示出的是给定属性第500个图像的信息,两个元素,图像,第二个表示标签
 
    print(len(data))



if __name__=='__main__':
    # identity_()
    attribute()
    
