import torch
import torch.nn as nn


X=torch.arange(9).reshape(3,3).float()

batch_normalization=nn.BatchNorm1d(3)
X_bn=batch_normalization(X)
print(X)
print(X_bn)