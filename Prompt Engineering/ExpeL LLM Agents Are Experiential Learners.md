# ExpeL: LLM Agents Are Experiential Learners

## 背景

将LLM进行微调从而适应特定的决策任务非常需要资源并且有可能损失模型的泛化能力。另外，一些SOTA模型比如GPT-4和Claude是不开源的，不能更改其参数。因此，我们需要一种新的方法论，让大模型可以从智能体经验中学习，并且不需要更新权重。为了解决这些问题，本文引入经验学习（ExpeL）智能体。本文的智能体会利用一组训练任务中的自然语言自动收集经验和提取知识。在推理阶段，智能体会回忆收集到的知识和经验用来决策。本文的经验结果强调了ExpeL智能体的稳定学习效率。
