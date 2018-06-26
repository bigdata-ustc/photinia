# Photinia

We believe that building an AI system has no difference to creating life. The tedious underlying construction will impede the pace of implementation.
The project is dedicated to the rapid implementation of inspiration, more convenient and flexible. You can be more focused on the process of building life.


我们相信，构建智能体无异于创造生命，繁琐的底层构建将阻碍实现的步伐。
石楠花项目致力于灵感的极速实现，更方便，更灵活，让你更加专注于构建生命的过程


## 项目构成

1. 数据输入与预处理
    1. 数据源（DataSource）接口
    1. 内存数据源
    1. 文件数据源
    1. 数据库数据源
    1. 数据预处理
    1. 异步加载
1. 模型构造
    1. 模型组件
    1. DNN组件
    1. CNN组件
    1. RNN组件
    1. 其他组件
1. 参数初始化
    1. 全连接层初始化
    1. 卷积层初始化
1. 模型训练
    1. 优化/训练操作
    1. 定义训练方法（Slot）
    1. 流程控制
1. 参数正则化
1. 模型保存与载入
    1. 保存模型到文件
    1. 保存模型到数据库
    1. 模型载入
    1. 部分组件载入与保存
1. 其他工具
1. 常用模型
    1. Alexnet
    1. VGG
    1. 卷积自编码DCAE
    1. 序列自编码SeqAE
1. 自然语言处理
    1. Word2Vec
    1. Doc2Vec
1. 生成对抗网络
    1. 卷积生成对抗网络DCGAN
    1. 序列生成对抗网络SeqGAN
1. 强化学习
    1. DQN
    1. DDPG
