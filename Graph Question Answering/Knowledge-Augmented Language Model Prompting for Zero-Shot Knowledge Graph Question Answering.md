# Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering

## 背景

LLM能够借助预训练得到的参数进行零样本、无额外信息辅助的问答任务。但是，这样隐含的知识很可能不充分、不正确，导致LLM生成与事实相异的答案。另外，微调大模型更新知识比较昂贵。因此，本文想直接通过LLM的输入增强知识。具体而言，本文首先根据题目与图谱的语义相关性提取相关的事实。之后，本文将事实作为prompt丢入LLM中生成答案。本文的模型不需要模型训练，因此是完全零样本学习。最后的效果超过了baseline，比大多数LLM都更加有效。
