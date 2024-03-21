# IC neuron: An efficient unit to construct neural networks

## 背景

基于物理学中的弹性碰撞模型，本文提出了一种能表现更复杂分布的新的神经网络模型。将其称为Inter-layer collision neuron。我们知道通过神经递质，可以将信息从一个神经元传递到另一个神经元。

## IC Neuron介绍

以物理学中的弹性碰撞为基础。输入可以被拆分成不同的子空间，表示不同的线性变换。

$$y = wx + \sigma((w - 1)x).$$

![Fig 1](./fig/IC%20Neuron.png)

神经元的计算公式如下所示：

$$\begin{aligned}
y &= \text{f}\left( \sum_{i=1}^n w_i x_i + \sigma\left(\sum_{i=1}^n (w_i - 1)x_i + b_1\right)+b_2\right)\\
&= \text{f}\left( \sum_{i=1}^n w_i x_i + \sigma\left(\sum_{i=1}^n w_i x_i - x_{sum} + b_1\right)+b_2\right)
\end{aligned}$$

其中，$\text{f}$代表一个激活函数，$b_1,b_2$是偏置。为了更加提升非线性的表达能力，可以把常数1放松。

$$\begin{aligned}
y &= \text{f}\left( \sum_{i=1}^n w_i x_i + \sigma\left(\sum_{i=1}^n (w_i - w')x_i + b_1\right)+b_2\right)\\
&= \text{f}\left( \sum_{i=1}^n w_i x_i + \sigma\left(\sum_{i=1}^n w_i x_i - w'x_{sum} + b_1\right)+b_2\right)
\end{aligned}$$

根据证明，IC神经元可以表示整个超平面平行，提供更为灵活的决策边界。

此外，还可以将IC神经元进行组装，组装成全连接结构、循环结构和卷积结构都是可以的。

通过比较IC神经元和MP神经元的复杂度是差不多的。

## 结果

![Fig2](./fig/IC%20Exp.png)

相比于传统神经网络，通过使用相同的超参，IC网络能提升准确率和加速训练过程。
