"""
需求:假设有500条数据,20个特征,标签为三类
现在实现一个三层的神经网络
其中第层1层13个神经元,第二层8个神经元,第三层是输出层
第一层激活函数是relu,第二层是sigmoid

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# 先设出数据
torch.manual_seed(1)
# 相当于制造样本,500个样本,每个样本20个特征
X = torch.randn((500, 20), dtype=torch.float32)
# 相当于制造了标签,500个样本,每个样本一个标签,标签的取值为{0,1,2}
y = torch.randint(low=0, high=3, size=(500, 1), dtype=torch.float32)


# 继承nn.Module类 来定义网络结构
class Model(nn.Module):
    def __init__(self, in_features=10, out_features=3):
        super(Model, self).__init__()#其实就是找到nn.Module的构造函数   
        # 这里定义了所需的网络
        self.linear1 = nn.Linear(in_features, 13)
        self.linear2 = nn.Linear(13, 8)
        self.output = nn.Linear(8, out_features)
    # 前向传播 总体的思路就是全连接1->激活函数->全连接2->激活函数-> output层（也是一个线性层）->softmax
    def forward(self, x):
        z1=self.linear1(x)
        sigma1=F.relu(z1)
        z2=self.linear2(sigma1)
        sigma2=F.sigmoid(z2)

        z3=self.output(sigma2)
        sigma3=F.softmax(z3,dim=1)
        return sigma3

input_=X.shape[1]#特征的数量-20
output_=len(y.unique())#分类的数目,,
''''
    这个tensor的形状是500*1,500个batch,每个batch一列
    先取unique()一个去重,返回的是一个tensor,
    然后len()就是这个tensor的长度,也就是分类的数目
    这样的话留下来的肯定就只有单个的 012了,取个len那肯定是3
'''


# 实例化模型
torch.manual_seed(1)
net=Model(in_features=input_,out_features=output_)
#到这里,所有的层就已经被实例化了,所有的w,b等也已经被初始化了

net.forward(X)#这里就是前向传播了,这里的X就是输入的数据,这里的net就是我们定义的网络结构

'''
    这里还有个骚写法
    net(X),直接调用net的__call__方法,这会执行除了init之外的所有方法
'''


sigma=net.forward(X)#这里的sigma就是输出的结果,也就是预测的结果
'''
    思考这里最终的输出是什么?
    应该是500*3,因为已经明确了out_features是3
'''
print(sigma.shape)
print(sigma)

#其实可以通过net.linear1.weight,,net.linear1.bias查看每一个层的矩阵
#矩阵一定是满足相乘要求的,但是略有奇怪,不需要记

