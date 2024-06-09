# LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention

## 摘要

本文提出LLaMA-Adapter，一个轻量的有效将LLaMA微调为跟随指令的模型的适应方法。通过使用5.2万自指导示范，LLaMA-Adapter只需要在7B固定模型的基础上使用1.2M可学习的参数，在8个A100上进行小于一小时的微调即可完成。具体来说，本文采用了一组可学习的适应提示词，并在更高的transformer层上预先考虑它们加入单词token中。之后，一个零初始化的无阀门的注意力机制被用来将新的指示性的提示加入到LLaMA中，同时有效保留其预训练的知识。
