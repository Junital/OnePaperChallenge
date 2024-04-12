# From Images to Textual Prompts: Zero-shot Visual Question Answering with Frozen Large Language Models

## 简介

LLM在无样本的情况下就能进行泛化新的任务。但是对于VQA，还是存在着挑战，因为图像和LLM之间的失联。然后本文就提出一个框架，先做一个注意力地图，然后捕捉对应的图片区域，用一个图片描述模型描述图片信息，然后再加上一些示例帮助LLM回答。
