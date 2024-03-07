# Cross-Modality Time-Variant Relation Learning for Generating Dynamic Scene Graphs

## 背景

从视频片段中生成动态场景图片可以在一系列有挑战的任务如环境感知、自动导航或者自动驾驶车辆和移动机器人的路径规划中提升语义视觉理解。但是，我们很难从一系列帧中通过动态场景找到时间变化的关系。

本文提出一种$\text{TR}^2$的方法，在动态场景图片中对时序变化进行建模。具体而言，就是将一些表示关系的embedding词嵌入进去。用带特征提取模块的Transformer和附加的消息token来描述相邻帧之间的关系。

最后，$\text{TR}^2$的表现超过了目前最好的方法。
