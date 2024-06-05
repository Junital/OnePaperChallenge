# mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video

## 背景

## mPLUG-2

本文提出mPLUG-2，一种新的多模态预训练范式，它可以有效地避免模态的牵连。与之前的工作用seq2seq生成或者endcoder实例判别不同，本文使用一个多模块组合网络，通过共享整体模块来进行模态联合、通过分理处不同模态模块来处理模态牵连。模态的选择很灵活，能为所有模态（包括文本、图像和视频）提供理解和生成任务。

## 实验结果

研究显示，mPLUG-2在30多个下游任务中获得了SOTA或强势的成绩，包括图像-文本理解、视频-文本理解和生成、单模态理解。

mPLUG-2在视频QA和视频描述任务中，通过很小的模型大小和数据规模就取得了SOTA准确率。
