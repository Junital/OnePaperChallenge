# Finetune Like You Pretrain: Improved Finetuning of Zero-Shot Vision Models

## 摘要

微调像CLIP这样的图像-文本模型在各种benchmark中都能获得SOTA准确率。但是，最近的工作展示了在微调过程中的微笑差异会导致最终表现能力的很大差距（ID分布相同的数据集、OOD分布不同的数据集都是这样）。

本文展示了一个自然而简单的模仿对抗预训练的方法，表现优于其他微调方法。具体来说，本文将下游类别标签作为文本提示词，并继续优化图像嵌入和类别可区分提示词嵌入之间的对抗损失。
