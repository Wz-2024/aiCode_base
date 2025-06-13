import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as CEL



torch.random.manual_seed(1)
X=torch.rand((500,20),dtype=torch.float32)
y=torch.randint(0,3,(500,),dtype=torch.float32)

input_=X.shape[1]#特征的数量-20
output_=len(y.unique())#分类的数目,,



#定义神经网络架构
class Model(nn.Module):
    def __init__(self,in_features=40,out_features=2):
        super().__init__()
        self.linear1=nn.Linear(in_features,13)
        self.linear2=nn.Linear(13,8)
        self.output=nn.Linear(8,out_features)
    def forward(self,x):
        sigma1=F.relu(self.linear1(x))
        sigma2=F.sigmoid(self.linear2(sigma1))
        zhat=self.output(sigma2)
        #调用封装好的Loss function 不需要最后一层激活
        return zhat
    
#开始实例化当前这个神经网络
torch.manual_seed(220)
model=Model(in_features=input_,out_features=output_)

zhat=model(X)
'''
    到目前为止,已经拿到了计算完的结果,但是还没用到标签y
'''

criterion=CEL()
loss=criterion(zhat,y.long())
print(loss)
#此时直接打印求导,,发现没有值,因为目前还没开始求导
loss.backward()
print(model.linear1.weight.shape)#这个形状就是权重的

print(model)    