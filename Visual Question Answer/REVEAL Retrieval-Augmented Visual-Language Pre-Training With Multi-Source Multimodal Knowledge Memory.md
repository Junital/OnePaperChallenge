# REVEAL: Retrieval-Augmented Visual-Language Pre-Training With Multi-Source Multimodal Knowledge Memory

## 背景

本文提出了一个端到端提取加强视觉语言模型（ReVeal），用于将世界上的知识编码进大容量的内存中，通过从内存中进行提取开源回答与知识相关的请求。ReVeal包括了四个重要组成部分：内存、编码器、提取器和生成器。提取器从内存中找到最相关的知识条目、生成器将输入请求和获取到的知识进行融合，生成输出。并且，所有的这些模块都是预训练、端到端的。
