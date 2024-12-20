# Robot Parkour Learning

## 背景

人类和动物都有非常惊人的运动天赋。跑酷中包含了敏捷且灵活的动作，这需要实时的视觉感知和对周围环境的记忆、感知和动作的紧密配合与强有力的肢体来通过障碍。

跑酷对机器人腿部运动非常有挑战，需要在复杂环境中快速克服不同的障碍。现存的方法可以生成多样但是无视觉的技巧和基于视觉但是只能专项得技巧。但是，自动跑酷需要机器人学习可泛化的技巧，同时满足视觉性和多样性，以此来感知并对不同场景进行反应。

看到基于学习的方法在走路、爬楼梯、模仿动物和肢体灵活控制上有非常稳定的效果，本文想在只适用低成本硬件的情况下为机器人跑酷使用基于学习的方法。

下面是一些学习跑酷的挑战：

1. 学习不同的跑酷技巧很难：现有的强化学习需要设计复杂的、很多术语的奖励方程才能引出比较满意的结果。通常每个行为都需要认为对奖励术语和超参进行调整。因此这些方法没有办法拓展为像跑酷这样多样的任务。相比之下，直接向动物的行为进行模仿可以避免繁琐的奖励设计和优化。但是这种方法缺乏第一人称视觉数据和不同的动物动作捕捉技能，导致机器人不能学习到多样敏捷的技能和自动根据环境选择的技能。
2. 障碍物对低成本小型的机器人很有挑战性。
3. 除去学习多样性技巧的挑战，在高速运动中视觉感知会是动态且延迟的。比如，当机器人以1m/s的速度移动，那么就会有0.2秒的信号通信延迟，导致0.2米的感知延迟。并且目前的基于学习的方法并没有展示出在高速敏捷移动中的有效性。
4. 跑酷对机器人的电力马达有要求，需要达到最大容量才行。因此必须要使用能减轻马达潜在损坏的先行方法。

![Fig1](./fig/challenges%20for%20small%20robot%20to%20parkour.png)

## 跑酷学习系统

本文的目标是构建一个端到端跑酷系统，直接使用原主板深度传感和主体来控制一个低成本机器人的每个关节，从而表现不同的敏捷跑酷技能。

本文提出一个学习单一端到端基于视觉的多样跑酷技巧原则的系统，仅仅使用一个没有任何参考动作数据的奖励。本文使用直接配点的强化学习来生成跑酷技巧，包括攀爬高的障碍、跳过大间隙、匍匐过低栅栏、挤过窄裂缝和跑步。本文提取这些技巧到一个单一基于视觉的跑酷原则，并通过第一人称的深感相机将其迁移到一个四足机器人上。

本文的强化学习包含了两个模拟训练阶段：用软动力限制来预训练和用硬动力限制来微调。在预训练阶段，本文通过使用实施软动力限制的课程，允许机器人穿透障碍物。这会让机器人逐渐在减小穿透的情况下克服这些障碍。在微调阶段，本文施加了所有动力限制并用真实的动力微调预训练中学到的行为。这两个阶段，本文只使用了简单的奖励方程，促使机器人在保留机械能量的同时向前移动。每个独立的跑酷技能学到之后，本文使用DAgger来将这些技能提取到一个单一基于视觉的跑酷原则上，让机器人可以部署。为了实现在一个低成本机器人上的稳定虚转实部署，本文对深度图片使用了一些预处理的方法，校准主板视觉延迟并加入了保障马达安全先行方法。

![Fig2](./fig/parkour%20two-stage%20RL%20training.png)

### 双阶段强化学习

由于深度照片渲染成本很高，并且在视觉数据上直接训练强化学习不是很稳定，本文使用环境的特权信息来帮助强化学习在模拟中生成具体的跑酷技巧。特权视觉信息包括：从机器人目前位置到前方障碍物的距离、障碍物的高度、障碍物的宽度、四维one-hot向量表示上述四种障碍。

本文将每个具体技巧原则设计为一个GRU（门循环网络）。其输入除了循环潜在状态还有本体感觉$s_t^\text{proprio} \in \mathbb{R}^{29}$（包括横摇、俯仰、基础角速度、关节位置、关节速度）、上一个动作$a_{t - 1} \in \mathbb{R}^12$、特权视觉信息$e_t^{\text{vis}}$和特权物理信息$e_t^{\text{phy}}$。原则的输出为目标关节的位置$a_t \in \mathbb{R}^{12}$。

本文分别训练所有的技巧原则$\pi_\text{climb}, \pi_\text{leap}, \pi_\text{crawl}, \pi_\text{tilt}, \pi_\text{run}$。本文使用一个一般的技巧奖励$r_\text{skill}$，这对所有自然移动的技巧都有用。这个奖励分为三个部分：一个前向奖励$r_\text{forward}$、一个能量奖励$r_\text{energy}$和一个存活红利$r_\text{alive}$。

$$r_\text{skill} = r_\text{forward} + r_\text{energy} + r_\text{alive}$$

其中，三个部分的计算公式如下：

$$\begin{aligned}
r_\text{forward} &= -\alpha_1 * |v_x - v_x^\text{target}| - \alpha_2 * |v_y|^2 + \alpha_3 * e^{-|\omega_\text{yaw}|},\\
r_\text{energy} &= -\alpha_4 * \sum_{j \in \text{joints}} |\tau_j \dot{q}_j|^2, \ \ \ \ \ r_\text{alive} = 2
\end{aligned}$$

其中，$v_x$是前向基础线性速率，$v_x^\text{target}$是基础速度，$v_y$是横向基础线性速率，$\omega_\text{yaw}$是基础角速度，$\tau_j$是关节$j$的扭矩。$\dot{q}_j$是关节$j$的速度，$\alpha$是超参。本文将目标速度设为1m/s左右。本文对每个关节设为二档的马力。

**预训练**：为了让机器人减少穿模现象，本文设置了奖励$r_\text{penetrate}$。本文通过对机器人碰撞区域采样碰撞点，衡量出穿模的体积和深度。由于臀部和肩部包含了所有马达，本文会对这些区域采样更多的碰撞点以加强动力限制。

具体来说，对于碰撞体上的一个碰撞点$p$，判断$p$是否违反了软动力学限制的指示函数为$\mathbb{1}\left [p\right ]$，$p$与碰撞的障碍物表面的深度为$d(p)$。穿透的体积可以估算为所有碰撞点$\mathbb{1}\left [p\right ]$之和，穿透的平均深度可以估算为$d(p)$之和。

$$r_\text{penetrate} = - \sum_p (\alpha_5 * \mathbb{1}[p] + \alpha_6 * d(p)) * v_x$$

本文将前向基础速度$v_x$同时与穿透体积和穿透深度相乘，从而避免机器人为了获得奖励而选择快速冲刺。

![Fig3](./fig/Robot%20collisions%20points.png)

此外，在基于个体机器人模拟表现进行重置后，本文使用一个可以适应性调整障碍物难度的课程。本文先在重置前基于平均穿透奖励计算机器人的表现。如果穿透奖励超过了某个阈值，本文将会提升障碍物的难度分数$s$，如果低于阈值会降低难度分数。所有机器人都从难度分数0开始，最大难度分数为1。

**微调**：不允许穿透，只使用$r_\text{skill}$进行微调。

### 通过蒸馏学习单跑酷原则

本文随机采样障碍物类型和属性，构建了40条赛道，每条赛道有20个障碍物。由于本文已经完全了解了每个状态$s_t$对应的障碍物类型，本文可以分配对应的具体技巧原则$\pi_{s_t}^\text{specialized}$来教跑酷原则如何针对一个状态进行动作。比如，本文将分配攀爬原则$\pi_\text{climb}$到监督跑酷原则上。除去循环潜在状态，输入包括了本体感受$s_t^\text{proprio}$，上一个动作$a_{t-1}$和一个潜在深度图像经过小CNN处理得到的嵌入$I^\text{depth}_t$。

提取目标为：

$$\argmin_{\theta_\text{parkour}} \mathbb{E}_{s_t, a_t\sim \pi_\text{parkour},sim}\left [D\left(\pi_\text{parkour}\left(s_t^\text{proprio}, a_{t-1}, I^\text{depth}_t\right), \pi_{s_t}^\text{specialized}\left(s_t^\text{proprio}, a_{t-1}, e_t^\text{vis}, e_t^\text{phy}\right)\right)\right ]$$

其中，$\theta_\text{parkour}$是跑酷原则网络参数，$sim$是使用硬动力学限制的模拟器，$D$是损失计算函数。

### 虚到实迁移部署

本文还需要解决仿真和现实的视觉差异。本文对原始渲染深度图片使用深度裁剪、像素级高斯噪声和随机人工处理；对原始真实世界深度图像进行深度裁剪、修补破洞、空间顺滑、时间顺滑的预处理操作。

![Fig4](./fig/visual%20bridge%20-%20simulate%20and%20real%20world.png)

图像的分辨率为48*64。由于主板算力的限制，机器人捕捉图片的刷新率为10Hz。本文的跑酷原则在模拟和实际中都使用50Hz确保敏捷移动技巧，同时采用异步接收CNN处理好的嵌入。

## 测试

本文通过测试展示出系统可以让两个不同的低成本机器人自动选择并执行合适的跑酷技巧来穿梭于充满挑战的真实环境。

本文在模拟中准确率很高，同时学的很快。真实世界里也表现得很不错。

## 个人感想

本文的核心就是“拆分”，从跑酷技巧的角度，本文先将每个技巧都分别进行学习，之后再通过“教授”的方式合并到一个policy上；从强化学习的角度，本文设计了两个强化学习阶段，在预训练中先放宽限制，通过学习之后在严格限制进行微调。另外，环境随着个体学习能力而改变的思想特别有创意，并且我认为可以将其应用到其他学习任务中。
