# Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation

## 背景

基于transformer的大语言模型，本质上是自然语言生成模型。过去一段时间，大语言模型将他们的能力拓展到生成更多模态的内容，包括音频、讲话、代码、医疗应用和机器人应用。

大语言模型有能力生成图像和视频。图像像素将通过一个视觉tokenizer映射到离散的token序列中。之后这些token就被喂入大语言模型transformer，被当成文本进行生成式建模。尽管大语言模型在视觉生成任务上取得了突破，但是相比扩散模型还是表现欠佳。

为什么语言模型会在视觉生成上落后于扩散模型呢？本文认为主要原因是缺乏好的视觉表示，与自然语言系统太过相似，很难有效建模视觉信息。为了支撑这一假设，本文展现了在相同条件下，带有改进的视觉tokenizer的掩码语言模型在图像和视频生成保帧率、效率上都超过了SOTA扩散模型。

主要注意的是，本文并没有强调大语言模型的优越性，而是想为大语言模型的视觉tokenize方法进行进一步探索。大语言模型和其他模型最基本的不同是，大语言模型使用的是离散的潜在格式：从视觉tokenizer得到的token。本文认为这些离散的视觉token不应该被忽视。相反，它们有一些独特的优势：

1. 与大语言模型的相容性：由于token表示和语言token一样，因此能直接享受一直以来对LLM的优化，包括更快的训练推理速度、模型架构的提升、模型规模的选择和GPU/TPU优化以及其他创新。通过将视觉、语言共享同一个token空间，可以真正帮助多模态LLM理解、生成、推理视觉信息。

2. 压缩的表示。离散的token可以在视频压缩中提供一种新颖的角度。视觉token可以作为一种新的视频压缩格式来减小磁盘储存和互联网传输时的带宽。与压缩的RGB像素不同，这些token可以直接喂入生成式模型中，绕过传统的解压缩和潜在编码阶段。这可以在生成式视频应用中提供更快的处理速度。

3. 视觉理解红利。先前的研究展示出离散的token在自监督表示学习中作为一个预训练对象很有用。另外，有研究发现使用token作为模型输入能提升鲁棒性和泛化性。

**视觉生成任务中的大语言模型**：视觉tokenizer $f$首先将视觉输入映射到一串离散token序列中。一段视频$\mathbf{V} \in \mathbb{R}^{T\times H \times W \times 3}$（$T=1$时为图片）被映射到离散表示$\mathbf{X} = f(\mathbf{V}) \in \{1, 2, \cdots, K\}^{T'\times H' \times W'}$。其中$K$为视觉token词典的大小。$\mathbf{X}$被按照光栅扫描顺序压扁为一个一维token序列，之后送入一个大语言模型中进行生成式建模。

视觉生成的大语言模型一共有两种类型。*自回归大语言模型*（AR-LM）包括ImageGPT、DALL-E、Parti等。AR-LM会根据先前的token和附加的条件信息$\mathbf{c}$通过无条件分布$p_\theta(\mathbf{x}_i | \mathbf{x}_{<i}; \mathbf{c})$ 预测下一个token。在推理阶段，AR-LM在token上使用标准的自回归解码。最后通过decider和视觉tokenizer将token转化为像素。

*掩码大语言模型*（MLM）是另一种用于视觉生成的大语言模型，包括MaskGIT、MAGVIT、Phenaki、MUSE等。MLM通过使用一个掩码的token目标进行训练。在目标中，某些token被随机盖住，需要根据可观察的token去预测被盖住的token。令$\mathbf{m} \in \{0, 1\}^n$为一个随机二进制序列，其中$\mathbf{m}^\top \mathbf{1} \in [0, n - 1]$。对所有$\mathbf{m}_i = 0$的$i$，MLM学习 $p_\theta(\mathbf{i}_i | \{\mathbf{x}_j: m_j = 1, \forall j\};\mathbf{c})$。为了在推理阶段生成视频或图片，MLM会使用无回归解码算法。首先会开始于一个完全掩码的序列，它会重复两个步骤实现递归式的填充：

1. 给定上一步未掩码的token，从$p_\theta$中采样整个序列$\hat{\mathbf{x}}^{(t)}$。
2. 重新对$\hat{\mathbf{x}}^{(t)}$中$\left \lfloor \lambda(t) \cdot n \right \rfloor$个token以最低概率掩码。$\lambda(t)$随时间$t$呈下降趋势。

**视觉tokenize**：VQ-VAE是图像tokenize的基石工作，其中包括了一个CNN编码器，一个视觉转化器（VQ）瓶颈和一个CNN解码器。给定一个视频$\mathbf{V} \in \mathbb{R}^{T \times H \times W \times 3}$，VQ-VAE的编码器$E$将输出潜在嵌入$\mathbf{Z} = E(\mathbf{V}) \in \mathbb{R}^{T' \times H' \times W' \times d}$。每个$\mathbf{Z}$中的嵌入向量$\mathbf{z} \in \mathbb{R}^d$被传入向量转化器$q$中，将嵌入向量分配到学习过的词典嵌入$\mathbf{C} \in \mathbb{R}^{K \times d}$中的最近的条目$\mathbf{c} \in \mathbb{R}^d$。

$$q(\mathbf{z}) = \mathbf{c}_i, \text{ where } i = \underset{j \in \{1,2,\cdots, K\}}{\arg \min}\left \| \mathbf{z} - \mathbf{c}_j\right \|_2$$

为了得到离散的token，本文丢掉了嵌入的维度，并用$\mathbf{Z}$的编号$\mathbf{X} \in \{1, 2, \cdots, K\}^{T' \times H' \times W'}$。对于解码，所有照片token的嵌入作为解码器$D$的输入来重新构建输入$\hat{\mathbf{V}} = D(\mathbf{Z})$。

## MAGVIT-2

本文提出了MAGVIT-2，一个将视频和图像映射到紧凑的离散token的tokenizer。本模型基于一个名为MAGVIT的SOTA视频tokenizer。本文进行了两个方面的创新。首先，本文提出一个新颖的无查找量化方法，使得能学习大容量的词典，从而提升语言模型的生成质量。其次，通过大量经验分析，本文发现对tokenizer的改动不仅能提升生成质量，也能让图像和视频的token共用同一个词典。

尽管VQ-VAE已经取得了巨大的进展，但在重构质量上的提升和之后的生成质量之间的关系还没有很好地裂解。一个错误的概念置换就是提升重构就等于提升大语言模型的生成能力。比如加大字典的容量能提升重构质量，但是当字典容量已经很大的时候生成能力并不会增加多少，反而可能会损害性能。

![Fig1](./fig/reconstruction%20generation%20quality%20curve.png)

为了训练一个更大的词典，一个简单的方法就是在提升词典大小的同时减小编码嵌入的维度。因为按照直觉，通过限制单个token的表示容量，就能加快在大词典分布上学习。

**无查找转换**（LFQ）：根据上述观察，本文减小了VQ-VAE的嵌入维度至0。具体来说，词典$\mathbf{C} \in \mathbb{R}^{K \times d}$被替换为整型集合$\mathbb{C}$，$\left | \mathbb{C} \right | = K$。在VQ-VAE模型中，转换器必须要在词典中查找所有$K$和$d$维度的嵌入，其中$d$为256，将最近的词典记录作为编码器的输入。但本文设计的方法并不需要全部查找，因此被称为LFQ。本文发现LFQ可以扩展词典大小，从而提升语言模型的生成质量。

![Fig1](./fig/reconstruction%20generation%20quality%20curve.png)

尽管各种LFQ方法都是可行的，但是本文提出一种直截了当的变体，假设词典维度和二进制潜在是独立的。

本文经验性地展示出本文的模型在三个方面胜过先前最好的视频tokenizer MAGVIT。第一，本文极大提升了MAGVIT的生成质量，在常见图像和视频benchmark中获得SOTA效果。第二，实验证明本模型的压缩质量好于MAGVIT和目前视频研所标准HEVC。另外，它和下一代视频编码解码器VV不相上下。最后，本文展示出相比MAGVIT，本文新提出的token更能帮助理解不同设置和数据集中的视频生成任务。
