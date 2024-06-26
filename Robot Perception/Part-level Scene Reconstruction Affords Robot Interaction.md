# Part-level Scene Reconstruction Affords Robot Interaction

## 背景

重构场景对机器人尝试理解和接触环境特别重要。然而，传统的场景重构方法主要侧重于生成静态的场景，比如用系数的路标、网格、surfel（点云+RGB-D）、三维形状或者语义物体来表示。这样会缺乏机器人操作的动态特性，并限制了除去抓取的交互任务的复杂性。

Han等人通过将重构场景增加了交互的概念，允许机器人预感动作影响并不同执行就可以分析机器人的规划。他们提出了一种新颖的重构交互场景的任务，并且可以将这个任务导入到基于ROS的模拟器上。这对长程任务和动作规划（TAMP）有非常重要的作用。

这种方法先用3D全景映射的方法从RGB-D数据中重构场景，然后分割物体，最后将其表示为3D物体。现存的重构交互场景的方法集中于将需要被重构的物体替换为素材库中的CAD模型，导致重构场景和观测场景有很大差别。

## 组件级重构（Part-level Reconstruciton）

本文提出组件级的重构，使用原始的形状来重新组装物体。这样可以精确地复制物理场景，并使机器人能与复杂固定的物体进行交互。

通过切分需要重建的物体至语义组件，并将原始的形状与每个组件进行对齐，就可以通过评估动力学关系的过程中将组件组装成CAD模型了。组件之间的关系包括父子连接关系、衔接关系和参数关系。

具体而言，本文通过使用一个语义点云补全网络来分解并将每个分割到的物体组合成组件。之后，本文使用一个组件级的CAD替代，包括将原始形状于个体组件进行对齐和估计组件之间的动力学联系。

![Fig1](./Fig/Part-level%20interactive%20secne%20reconstruction.png)

### 动力学场景表示

通过将场景实体组件和其动力学信息进行合并，本文将Han等人提出的联系图谱（$cg$）延伸至表示3D室内场景中，表示为$cg=(pt, E)$。其中，$pt$是一棵分析树、$E$是一组邻近关系。通过支撑关系（$S$），分析树将场景实体结点（$V$）分级地组织起来。而邻近关系会捕捉物体之间的关系。$V$中每个物体结点包括描述语义的属性和几何信息。

为了提升$cg$，本文对每个物体结点$v\in V$增加了一种额外的属性$pt^p$。这个属性表示着单一物体内的组件分析树（$pt^p$），其由组件结点（$V^p$）和动力学联系（$\mathcal{J}$）组成：

**组件结点**集合$V^p = v^p$表示一个物体内的所有组件实体。对于每个实体，$v^p = \left \langle l,c,M,\Pi \right \rangle$，包括一个编号（$l$）、一个组件语义标签（$c$）比如桌腿、一个形式为三角网格或点云的几何模型（$M$）、和一组表面（$\Pi$）。表面由$\Pi = (\pi^k, U^k)$进行表示，其中$U^k$是一个3D顶点的列表，确定了一个多边形勾勒出平面$\pi^k$的轮廓。平面$\pi^k$表示为映射空间中的一个同质向量$[\mathcal{n}^{k^T}, d^k]^T \in \mathbb{R}^4$。单元平面向量由$\mathcal{n}_i^k$表示。方程$\mathcal{n}^{k^T}\cdot \mathcal{u} + d^k = 0$描述了平面上任何点$\mathcal{u} \in \mathbb{R}^3$都要满足的限制。

**动力学关联**集合，即$\mathcal{J} = J_{p,c}$，表示了一个物体内组件实体之间的参数连接。一个连接由$J_{p,c} = \left \langle t_{p,c}, T_{p, c}, \mathcal{F}_{p,c} \right \rangle$表示，存在于父组件（$v_p$）和子组件（$v_c$）之间。每个连接包括了连接类型（$t_{p,c}$）、父子转变（$T_{p,c}$）和连接轴（$\mathcal{F}_{p,c} \in \mathbb{R}^3$）。

本文只考虑三种连接类型：固定连接、棱柱连接（允许滑动）、关节连接（允许旋转）。

$v_p$和$v_c$如果要建立一个动力学的关联，需要满足如下限制：

$$\begin{aligned}
\exists (\pi^i_p, U^i_p) &\in \Pi_p, (\pi^j_c, U^j_c) \in \Pi_c, \\
\text{s.t. } \text{Align}\left (\pi^i_p, \pi^i_c\right )&\overset{\text{def}}{=}\text{abs}\left (\mathcal{n}^{i^T}_p, \mathcal{n}_c^j\right )\ge \theta_a,\\
\text{Dist}\left (\pi^i_p, \pi^i_c\right )&\overset{\text{def}}{=}\frac{1}{|U^j_c |}\sum_{\mathcal{u}\in U_c^j}\mathcal{n}_p^{i^T} \mathcal{u} + d^i_p \le \theta_d,\\
\text{Cont}\left (U^i_p, U^i_c\right )&\overset{\text{def}}{=}\text{A}\left(U^i_p\cap\text{proj}_{p,i}(U_c^j)\right) / \text{A}\left(U^j_c\right ) \ge \theta_c,
\end{aligned}$$

其中：

- $\text{Align}\left (\pi^i_p, \pi^i_c\right )$定义为两个表面之间的对齐，$\text{abs}\left (\cdot\right )$用来计算绝对值，$\theta_a$是用来判断对齐好不好的（$\theta_a = 1$就是完美的对齐，也就是平面是平行的）。
- $\text{Dist}\left (\pi^i_p, \pi^i_c\right )$定义为通过多边形$U_c^j$的顶点到平面$\pi_p^i$的平均距离，描述两个表面之间的距离。$|U|$是顶点的数量，$\theta_d$是最大允许的距离。
- $\text{Cont}\left (U^i_p, U^i_c\right )$定义为联系比率，$\text{A}\left(\cdot \right)$用来计算多边形的面积，$\theta_c$是最小可满足的联系比率，$\cap$用来计算两个多边形的交集，$\text{proj}_{p,i}(U_c^j)$通过将多边形$U_c^j$每个顶点映射到$\pi_p^i$，从而实现多边形到平面的映射。映射公式如下：

$$\hat{\mathcal{u}}^j_c = \mathcal{u}^j_c - \mathcal{n}_p^{i^T}(\mathcal{u}^j_c-\mathcal{u}^i_p)\mathcal{n}_p^{i}, \forall\mathcal{u}_c^j\in U^j_c$$

其中，$\hat{\mathcal{u}}^j_c$是$\mathcal{u}^j_c$映射到$\pi_p^i$的对应点。$\mathcal{u}_p^i$是$\pi_p^i$上的任意一点。

### 组件级CAD替换

**单组件对齐**：对于每个单独的组件，本文选择选择最相似的原始形状并将形状对齐到组件的6D转变。给定一个带有点云集合$P$的组件实体，本文从有限原始形状候选者集合$\mathcal{M}$中找到一个最优原始形状$M^*$，同时计算出一个优化6D转变$T^*_{ind} \in SE(3)$使$M^*$向$P$对齐。优化问题对应的公式如下所示：

$$M^*, T^*_{ind} = \min_{M_i \in \mathcal{M}^c, T\in SE(3)} \frac{1}{|h(M_i)|} \sum_{\mathcal{u} \in h(M_i)}d_P(T_i \circ \mathcal{u}),$$

其中，$h(M_i)$使一组从CAD模型$M_i$上均匀采样的点，$d_P(\mathcal{u})$是从采样点$\mathcal{u}$到$P$中的最近点的距离，$T_i\circ \mathcal{u}$是点$\mathcal{u}$应用转变$T_i$后的位置。通过这个优化公式得到最优的$M^*$和对应的$T^*_i$。

**动力学关联估计**：通过将对齐好的组件进行关联估计，就可以得到分析树$pt^p$了。

![Fig2](./Fig/Kinamatic%20relation%20estimation.png)

首先初始化一个组件结点$v^p$：

1. 获取到语义标签$c$、实例标签$l$和点云$P$。
2. 通过对齐操作将点云$P$替换为原始形状$M$。
3. 通过迭代使用RANSAC，提取表面$\Pi$。

对于一组对应一个物体的组件实体结点$V^p$，本文估计了$pt^p$的结构，比如一个物体中的优化父子关联$S^{p*} = \{s_{p,c}\}$。本文制作了一个优化问题：在满足动力学关联限制的情况下，最大化整体联系分数$\text{Cont}(\cdot, \cdot)$。

$$\begin{aligned}
S^{p*} &= \argmax_{S^p} \sum_{s_{p,c} \in S^p} \max_{i,j}(\text{Cont}(U^i_p, U^j_c))\\
\text{s.t. } &\text{Align}\left (\pi^i_p, \pi^i_c\right )\ge \theta_a,\\
&\text{Dist}\left (\pi^i_p, \pi^i_c\right )\le \theta_d
\end{aligned}$$

本文通过如下两个步骤来解决这个优化问题：

1. 本文用$V^p$里的结点来构建一个有向图：通过遍历所有结点对，如果两个结点满足上述限制，那么就从结点$v_p$到$v_c$增加一条边$s_{p,c}$，权重设置为$\max_{i, j}(\text{Cont}(U^i_p, U^j_c))$。
2. 本文开始寻找优化父子联系$S^{p*}$：尽管构建的图已经涵盖了所有可能的实体联系，但是这个有向图不一定就是树，因为一个结点的入度可能大于1。本文需要在有向图中找到一个最大权重的树，就使用了埃德蒙算法来实现。

之后，本文通过将原始组件与一个连接模板库进行配对，对在$S^{p*}$所有的父子关系估计出参数化的连接$\mathcal{J}$。这涉及到要决定连接类型、轴、基于语义的连接位置。比如说，一个微波炉的门应该与主体呈现关节连接，并通常固定在微波炉的边缘。

### 组件间空间改良

通过进一步实行改良过程以调整连接$\mathcal{J}$中的转变，使得组件形成的父子配对可以更好地对齐。本步骤是要减少组件之间的穿透、穿模情况。

算法自上而下对整个组件分析树进行改良。首先会拿到一组父子的相关转变$T_{p,c}$。其次，成对比较$v_c$和$v_p$的表面，选择粗略对齐的标准平面向量$X_c,X_p$以备下游的转变改良。接下来，利用从$X_c$向$X_p$对齐的改良转变$T^r_c$来调整$T_{pc}$，最后重新更新这个转变。

## 实验结果

通过实验，组件级的重构相比物体级的重构表现更好，能精准捕捉更细微的细节并提升准确率。

同时，这些重构的组件级交互场景会为不同的机器人应用提供有价值的动力学信息。本文通过在执行真实世界任务前的交互场景移动操控规划，证明了场景的可行性。

本文的方法取得了很好的结果，并且在仿真实验中机器人可以对物体进行交互，非常合适。

## 局限性

1. 目前输入的数据质量不高，会有噪声。需要更加高级的扫描补全方法来提升输入数据的局限性。
2. 仅仅通过静态的观察来判断运动学连接也存在模棱两可的感觉，这需要用一些方法来解决。
3. 目前的系统认为结构内部是实心的（比如衣柜内部），这和人类存在着差距。
