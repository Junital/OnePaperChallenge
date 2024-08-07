# Distilling the Knowledge in a Neural Network

## 背景

类比昆虫，它们幼虫时期从环境中汲取能量和营养；而成虫阶段，因为要游行和繁殖，昆虫往往呈现出完全不一样的形态。

在机器学习阶段也是这样，我们在训练和部署阶段使用的模型都是十分相似的，但是实际上两个阶段的要求不同：对于音频和目标检测，训练阶段必须从庞大冗余的数据集中提取出模式和结构，但对于实时性要求不高。可是，在部署阶段，模型对于延迟和计算资源的要求是很高的。

因此，本文认为在训练阶段可以训练庞大的模型，但是应该在部署阶段进行“知识提取”，将其转化成一个小模型，从而能适应部署的要求。

一个挑战是，本文需要在学习好参数的模型中识别出知识，并且要改变模型的同时保证知识模式和结构是不变的。一种更加抽象的知识表示，其实就是从向量到向量的映射。

对于笨重的模型，通常训练目标是尽可能最大化正确答案的概率，但是副作用是会将概率分配给不正确的答案。即使这些概率可能很小，概率之间还是会有数量级的差距。这也是比较容易理解的，因为汽车和卡车之间被识别错误的概率，一定远远比汽车和萝卜之间识别错误的概率要高。

在训练的时候，目标方程应该尽可能反映用户真实的目标。尽管如此，模型通常在训练集上优化结果，但真实的目标是让其在新数据上泛化。当然，训练模型使其能够泛化再好不过，不过这需要正确的方式进行泛化，而方法并不容易获取。

但是，我们在将知识从大模型提取到小模型的过程中，我们可以按照训练大模型同样的方法训练小模型使其泛化。如果大模型因为用联邦学习泛化很好，那么小模型应该也通过联邦进行训练，这样会比正常方式训练泛化能力更强。

一种明显的知识提取方式是：将大模型预测得到的类别概率结果作为软标签训练小模型。在这个迁移阶段，可以使用相同的训练集，也可以使用一个单独的迁移集（可以是无标签的）。如果大模型是一个联邦学习模型，可以将所有模型得到的结果进行均值化，得到软标签。如果软标签有很高的熵，就会相比硬标签提供更多的信息，并且在训练样本之间**梯度相关更小**。因此小模型可以使用更小的数据和和更大学习率。

需要注意的是，小模型不能完全匹配软标签，因此退而求其次去追求正确答案是必要的。

## Distill

本文将使用带有Temperature的Softmax方法，公式为$\frac{e^{Z_i/T}}{\sum_de^{Z_i/T}}$。通过调高Temperature，可以提高信息熵，从而让小模型能获得更多泛化信息。

![Temperature Softmax](./fig/Temperature%20Softmax.gif)

本文采用两个目标函数权重相加，第一个目标函数是与软标签进行交叉熵计算、第二个目标函数是与硬标签计算交叉熵。第二个目标函数的权重低一些会比较好。

另外，当大模型的参数服从均值为0，相当于就是将小模型的输出参数与大模型参数趋于相同。

## 实验

实验表明，知识提取的模型保证了99.5%的准确率，并且模型的规模大幅度减小了。并且，软标签在数据量很小的情况下也能泛化。

## 个人感想

引用昆虫的变形作为因子，提出要将大模型变成小模型。分析出软标签内部会包含很多信息。然后制作出Temperature Softmax提取出更多信息，用大模型的结果训练小模型。思路非常清晰，值得借鉴。
