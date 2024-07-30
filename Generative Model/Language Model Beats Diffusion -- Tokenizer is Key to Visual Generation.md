# Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation

## 背景

虽然大预言模型是语言生成任务中顶流，但是他们在图像和视频生成中表现没有扩散模型那么好。为了有效对视觉生成任务使用LLM，一个重要的部分就是能将像素空间输入映射到适合LLM学习的离散token的视觉tokenizer。本文提出了MAGVIT-v2，一个用于视频、图像生成的用一个常见token字典的能生成准确且出色的token的视频tokenizer。通过配备此tokenizer，本文展示出LLM在标准图像生成和视频生成benchmark（ImageNet、Kinetics）中超过了扩散模型。另外，本文展示了此tokenizer在额外两个任务中超过了之前顶尖的视频tokenizer：1. 根据人类评估的和下一代视频编码解码器（VCC）同类的视频压缩；2. 为动作识别任务学习有效的特征表示。
