# Human-level few-shot concept induction through minimax entropy learning

## 背景

人类是同时从监督学习和无监督的模式观察中进行学习的。但是目前的机器学习就是从标记的大量数据中进行训练，不相人类那样仅靠少量的无监督例子就可以归纳出一个相关联的概念。

最近几十年，对于感知任务，机器学习取得了很大的进展，但是需要大量的、精心挑选的数据集用于训练。尽管有很多成就，但是关联性归纳领域并没有被彻底解决，尽管这对人类智力起到了关键的作用。人类不仅可以通过监督学习、能进行物体识别，还可以通过一些例子的观察来归纳没有见过的关系。比如，婴儿可以识别类似ABB结构的模式，小孩可以快速了解逻辑上的关联性和非关联性。甚至缺乏教育亚马逊土著人有能力解决IQ测试中的复杂的集合概念归纳问题。这种归纳的熟练性与智力水平息息相关，尤其是是智商。

目前很多的相关方法都用的是监督学习的方法，这与人类的学习方法并不一致。并且展现的归纳推理水平也相较人类差距甚远。

人类和机器的学习效率也不一样，人类能够从一个单一的上下文实例归纳出新的概念，但是机器需要大量的数据才能所谓“归纳”出概念。似乎机器学习只是将推理退化为了“死记硬背”。

像人类一样的概念理解的核心问题是：人是如何仅通过观察事件序列归纳出一个不熟悉的概念呢？人类是如何从那么少的例子中通过生成不熟悉的概念从而解释现象呢？

## Minimax Entropy

本文提出一种基于极大极小商在IQ测试中小样本概念归纳和抽象推理的统一的计算框架。与之前做的一些工作不同，本文使用一个全局的优化策略：初始化，将能量模型与滤波器结合。之后动态增加删除滤波器，利用极大极小熵原则加权结合。

这种策略会比单纯的贪心策略要好，但是也会引出一个问题：如何处理极端情况下处理大量甚至可以覆盖连续空间的滤波器呢？本文使用双层优化来解决这个问题：在内层优化中，本文确定连续滤波器函数的参数，并在每个滤波器中球的优化参数来拟合观测。在外层优化中，本文提出极大极小熵学习。大致上，极小熵学习与最小化观测分布和建模分布的KL散度相等价。同时，极大熵原则会找到与观测数据相符的优化分布。

在实际中，极大极小熵结构共有三个学习阶段：在第一阶段，本文对每个连续滤波器函数的优化参数进行求解，并将其保持固定；在第二阶段，本文进行极小熵学习，通过最大化$\log$似然来表示选择最佳的编码方案；第三阶段进行极大熵学习，在上述两个阶段的结果下，最佳分布被解为Gibbs形式。内部的滤波器学习阶段可以用任何领域特定解算机独立实现。外部的极大极小熵学习过程可以用最大$\log$似然实现，参数包括所有滤波器内部的指示变量和计算滤波器重要性的权重系数。

![Fig1](./fig/Minimax%20Entropy%20Learning.png)

具体而言，对每个问题的上下文语境集合$C=\{x_i\}$，包含着物体为中心的表示，可以是两到三个简短序列或静态图像。本文的极大极小熵模型应该在物体为中心的表示空间中，学习到最能表示隐藏概念特征的分布$p(x)$。假如说，隐藏概念可以通过一系列响应方程$\{H_j(\cdot)\}$或滤波器捕捉，那么就需要满足下面的最大熵原则：

$$\begin{aligned}
\max_p &-\int p(x) \log p(x) \mathrm{d}x \\
\text{subject to} &\underset{x\sim p(x)}{\mathbb{E}}[H_j(x)] = \mu_j^{\text{obs}}, \forall j\\
&\int p(x)\mathrm{d}x=1
\end{aligned}$$

其中，$\mu_j^{\text{obs}}$代表在上下文中的平均滤波器响应。此优化可以接受一个分析性的解决方案，如下所示：

$$p(x) = \frac{1}{Z} \exp\left[-\sum_j\lambda_jH_j(x)\right]$$

其中，$Z=\int\exp [-\sum_j\lambda_jH_j(x)]\mathrm{d}x$，是标准化基。$\lambda_j$是优化拉格朗日乘数，可以通过上述最大似然学习得到。

在最小熵学习阶段，本文在最大熵学习得到的结果的基础上最小化模型的熵。其等效于最小化受到隐藏概念限制的真实分布$p^*(x)$和本文刚得到的近似分布$p(x)$之间的KL散度：

$$\underset{p}{\min}-\int p(x)\log p(x) \mathrm{d}x = \underset{p}{\min} \mathbf{KL}(p^\star, P)=\underset{p}{\max}\mathbb{E}_{p^\star}[\log p(x)]$$

上述公式其实和最大似然学习是等价的。这一步是通过选择出滤波器$\{H_j(\cdot)\}$的优化集合实现的，从而在选择的滤波器的编码方案下最小化期望编码长度。除此之外，本文增加一组全局指示变量${z_j}$到优化中，最大化分布的$\log$似然，公式如下所示：

$$\max_{\lambda, z}\mathbb{E}_{x_i}[\log p(x_i)]=\mathbb{E}_{x_i}\left [-\sum_j \lambda_j z_j H_j(x_i)-\log Z \right ]$$

尽管如此，极大极小熵学习结构仍然有一点问题：传统的固定滤波器的结构并不能适应于不同的场景和案例。对此，本文设计了双层优化问题：内部优化计算出能恰好表述隐含概念的优化参数，外部就是刚刚的极大极小熵学习。因此在实例学习中，本文在一定限制下最大化$log$似然，如下面的完整公式所示：

$$\begin{aligned}
\max_{\lambda, z}\ \ \  \mathbb{E}_{x_i}[\log p(x_i)] &= \mathbb{E}_{x_i}\left [-\sum_j \lambda_j z_j H_j(x_i;\theta_j^\star)- \log Z\right ] \\
\text{subject to}\ \ \ \ \ \ \ \ \ \ \ \  \theta_j^\star &= \arg \min \ell_j(\{x_i\}, \theta_j), \forall j
\end{aligned}$$

其中，$\theta_j^\star$代表在过滤器组$H_j(\cdot; \theta_j)$中最能捕捉隐含概念的优化参数。

## 任务测试

本文使用了如下的测试任务：RPM、MNS和$\text{O}^3$，分别为形状归纳、数字归纳和数量归纳。

相比于目前的Transformer模型和其他归纳推理和抽象推理模型，本文的极大极小熵模型表现非常好，在归纳方面有很大的提升，超过了大部分人类。同时通过消融实验验证了各个模块的有效性。

## 个人感想

- 自己确实第一次接触这种学习方法，没有能力提出创新和改进。
- 目前确实基于神经网络的模型更像是在“记忆”，而不是推理。这并不是学习。
- 不过我认为学习还是不仅仅需要推理，还需要纠偏，对于错误的答案要分析问题，从而可以学习到新的知识。
