## 概述
本项目包含了以下内容
### learn
（1）learn中从最基础的网络搭建开始，介绍了如何写前向传播、反向传播、loss的构造，optim的初始化、normalization的写法等<br>
（2）演示不同版本的分类cnn的搭建方式，包括最简单的LeNet5，AlexNet还有稍复杂的VGG，ResNet<br>
（3）介绍了数据集预处理的方式，但只是简单的归一化，增强、封装而已。没有设计到Diffusion相关的预处理算法<br>
（4）（5）利用上述内容搭建了比较完备的训练过程，包括了如何搭建并初始化网络，训练代码，保存损失信息，保存模型等。<br>
### Diffusion
  diffusion文件夹中是DDPM的简单复现，用到的架构是UNet
