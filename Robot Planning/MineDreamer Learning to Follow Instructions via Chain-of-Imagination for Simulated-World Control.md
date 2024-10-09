# MineDreamer: Learning to Follow Instructions via Chain-of-Imagination for Simulated-World Control

## 背景

设计一个能像人类一样服从各种指令的通用具身智能体是一个长期持续目标。但是目前方法由于在理解抽象、连续的自然语言指令上存在困难，经常不能稳定服从指令。因此，本文引入MineDreamer，一个在Minecraft模拟器上的自由具身智能体，利用底层控制信号生成上设计新范式，提升指令服从能力。具体来说，MineDreamer基于多模态大语言模型和diffusion模型，使用想象链（CoI）机制来预想执行指令的过程并将想象转化为更加具体的视觉提示引导当前状态。之后，智能体生成键盘-鼠标动作来有效实现这些想象，在每一步稳定服从指令。
