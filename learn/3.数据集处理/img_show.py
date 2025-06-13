import torchvision
import torch
import lmdb
from torchvision import transforms


#下载的LSUN只有训练集和验证集,没有测试集
data_train=torchvision.datasets.LSUN(
    root='/data_disk/dyy/python_projects/bili_dif/data2/lsun-master/data',
    classes=['church_outdoor_train'],
    transform=transforms.ToTensor()
)
data_val=torchvision.datasets.LSUN(
    root='/data_disk/dyy/python_projects/bili_dif/data2/lsun-master/data',
    classes=['church_outdoor_val']
    # transform=transforms.ToTensor()
)

print(data_train)
data_val[2][0].save('./test.png')
