# Chain of Thought Empowers Transformers to Solve Inherently Serial Problems

## 背景

思维链，即引导模型生成中间步骤序列，是提升LLM在数学、符号推理任务准确率的有效方法。然而，CoT背后的机制仍然不清晰。本文从表述能力的角度，为decoder-only transformer提供了CoT强大的理论理解。根据概念，CoT让模型具备执行本质上为串行计算的能力，而transformer却缺少这种能力（尤其层数很低的时候）。给定输入长度$n$，先前的工作已经证明了带有有限精度$\text{poly}(n)$嵌入大小的固定深度transformer只能解决$\text{TC}^0$问题。本文会给出更加严格的表示能力上界，带有固定bit精度的常量深度transformer，只能解决$\text{AC}^0 \subset \text{TC}^0$问题。
