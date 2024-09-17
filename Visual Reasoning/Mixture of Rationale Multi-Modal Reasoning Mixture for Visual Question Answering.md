# Mixture of Rationale: Multi-Modal Reasoning Mixture for Visual Question Answering

## 背景

零样本视觉问答是一个需要在模态之间推理的挑战性任务。虽然一些方法依赖于思维链单一推理框架，但并不能捕捉VQA问题的复杂性。另一方面，一些用多种推理的其他的方法仍然出现低多样性、模态难以对齐、提取和融合效果差的现象。为了应对这些挑战，本文提出了推理融合（MoR），一个为VQA融合多种推理的新颖多模态推理方法。MoR使用一个冻结的视觉语言预训练模型（VLPM）来动态生成、提取并融合多模态思维。本文在NLVR2和OKVQA数据记上进行了评估，得到了准确率的提升。
