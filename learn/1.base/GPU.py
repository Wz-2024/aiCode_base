import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# 设置超参数
lr = 0.01  # 调整学习率
gamma = 0.8
epochs = 10
bs = 128  # batch_size

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mnist = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

# 加载数据集后就放到DataLoader中
batched_data = DataLoader(mnist, batch_size=bs, shuffle=True)


def print_message():
    print("打印dataLoader封装后的 用 x,y 来索引的结构")
    for x, y in batched_data:
        print(x.shape)
        print(y.shape)
        break  # 只打印一个批次
    print("当前batch中第一个样本中有多少个元素:", mnist.data[0].shape)
    # 当前batch有128个样本,每个样本有28*28=784个元素


# 我们对二维数据的转化比较熟悉,因此需要先转换到二维来进行操作
# mnist(128,1,28,28)------>mnist(128,1*28*28) 这样他就有28*28个特征,作为featuer_input

input_ = mnist.data[0].numel()  # 输入神经元的个数
oput_ = len(mnist.targets.unique())  # 目标的分类数


# 定义需要用到的网络结构
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 128)
        self.output = nn.Linear(128, out_features)
        
        # Xavier 初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        """
        这里需要先整理神经元的结构,让它在第二维变成28*28(预设好的)维度
        """
        x = x.view(-1, input_)
        sigma1 = torch.relu(self.linear1(x))
        # 因为要算准确率,这里需要用到softmax,并且是加了log版本的
        sigma2 = F.log_softmax(self.output(sigma1), dim=1) 
        return sigma2


# 定义一个训练函数,其中包括损失函数,优化算法等,双循环的训练,梯度下降 流程等
# 参数方面,因为batch_size在DataLoader中已经用过了,这里不必要再传一遍
def fit(model, batched_data, lr=0.01, epochs=10, gamma=0.9):
    "因为上边用了softmax,这里就只能用NLLLoss了"
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=gamma)

    # 训计数器,表示训练开始前模型查看的样本数为0,预测对了的数量为0
    samples = 0
    acc_num = 0
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(batched_data):
            """
            #进来第一件事是对y降维,为什么降维?
            #因为它作为一个标签,就应该是一个一维张量,表示当前批次每个样本的标签
            """
            x, y = x.to(device), y.to(device)  # 将数据移动到GPU
            y = y.view(x.shape[0])
            # 正向传播
            # 这里有骚写法,直接model(x),利用的是nn.Module的__call__
            sigma = model(x)
            loss = criterion(sigma, y)
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #求解准确的个数
            yhat = torch.max(sigma, 1)[1]
            acc_num += torch.sum(yhat == y)
            #求当前看过的样本数
            samples += x.shape[0]
            # 分子表示已经看过的数据有多少
            # 分母表示一躬需要看多少数据,如果eopch=10,样本为60,那分母就是600
            if (batch_idx + 1) % 100 == 0 or batch_idx == len(batched_data) - 1:
                print(
                    "Epoch{}:[{}/{} {:.0f}%],Loss:{:.6f},AccRate{:.3f}%".format(
                        epoch + 1,
                        samples,
                        epochs * len(batched_data.dataset),
                        samples / (epochs * len(batched_data.dataset)) * 100,
                        loss.data.item(),
                        float(100 * acc_num.item() / samples)
                    )
                )


if __name__ == "__main__":
    # print_message()
    # 训练与评估
    print(3)
    torch.manual_seed(112)
    model = Model(in_features=input_, out_features=oput_).to(device)  # 将模型移动到GPU
    fit(model, batched_data, lr, epochs, gamma)