# RelTR: Relation Transformer for Scene Graph Generation

## 背景

在一个相同的场景里，不同的物体或多或少会相互关联，但是只有一小部分的联系是值得注意的。受到在物体检测领域表现很好的Detection Transformer的启发，本文将场景图谱生成（SSG）看作一组预测问题。本文提出一个端到端SSG模型：Relation Transformer (RelTR)，是一个encoder-decoder结构。encoder用于推理视觉特征上下文；decoder用不同类型的注意力机制和成对的主体客体请求推断出主体、客体和关系三元组。本文为端到端训练设计了一份预测损失，表示真实关系和预测关系的匹配程度。
