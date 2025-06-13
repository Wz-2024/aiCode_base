import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

data = torch.ones(size=(10, 3, 227, 227))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11,stride= 4)  # in out kernel stride
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride= 2)

        self.conv2 = nn.Conv2d(96, 256,kernel_size= 5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride= 2)

        # 前两个部分是在尽可能缩小,现在开始提取特征
        """
            (kernel_size,stride)=(5,3)(3,1)时,能够维持住特征图的大小
        """
        self.conv3 = nn.Conv2d(256, 384,kernel_size= 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384,kernel_size= 3,padding= 1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3,padding= 1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 进入全链接层
        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        "把数据拉平"
        x = x.view(-1, 6 * 6 * 256)
       
        x=F.relu(F.dropout(self.fc1(x), p=0.5))

        x = F.relu(F.dropout(self.fc2(x),p=0.5))
        output = F.softmax(self.fc3(x), dim=1)
        return output


if __name__ == "__main__":
    model = Model()
    model(data)
    print(data)
    print(summary((model), input_size=(10, 3, 227, 227)))
