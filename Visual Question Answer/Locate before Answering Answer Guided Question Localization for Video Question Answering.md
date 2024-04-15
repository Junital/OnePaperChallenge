# Locate before Answering: Answer Guided Question Localization for Video Question Answering

## 简介

本文侧重于视频问答（VedioQA）。不过目前相关的方法只侧重于很短时间内的视频，而无法处理长视频中场景变换和多动作带来的噪声和冗余。考虑到问题常常只集中在一个很小的时间段中，本文提出先通过问题锁定对应的视频片段，然后再根据这一片段进行答案推理。本文采用分离可选训练策略来更新问题定位模块和回答预测模块。
