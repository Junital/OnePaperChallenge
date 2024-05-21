# Large Language Models as Commonsense Knowledge for Large-Scale Task Planning

## 背景

本文展示了LLM可以提供一个世界常识模型，也能作为一个原则来依靠此模型来行动。世界常识模型和原则可以被合并在一个搜索算法中，比如蒙特卡洛树搜索，来提升任务规划。在本文新研究的算法LLM-MCTS中，基于LLM的世界模型会为蒙特卡洛树搜索提供一个常识先导，以此获得有效的推理；基于LLM的原则会作为启发式信息来帮助进行搜索，提升搜索的效率。
