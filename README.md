### paper:APFL: 一种隐私保护的抗投毒攻击联邦学习方案

### 描述
现有联邦学习改进方案大多仅从隐私保护或抗投毒攻击方面进行改进，不能兼顾两种攻击，为了同时解决联邦学习中的推断攻击和投毒攻击，提出了一个隐私保护的抗投毒攻击联邦学习方案 (APFL)，使用差分隐私技术和根据模型间余弦相似度赋予各客户端相应聚合权重的方式设计了模型检测算法MAD,使用同态加密技术将本地模型加权聚合。

### environment

1.python3.8

2.pytorch1.7.1

3.pip install -r requirements.txt

### usage

Run the code

```asp
python server.py -nc 21 -cf 1 -E 1 -B 16 -dataset cifar10 -Algorithm APFL -mn cifar10_cnn  -ncomm 100 -iid 1 -lr 0.0002 -g 0
```
### 文件和目录结构
| 文件/目录         | 描述                  |
| ----------------- | ---------            |
| `data/`           | 数据集目录            |
| `log/`            | 记录目录              |
| `model/`          | 模型配置文件          |
| `clients_ckks.py` | 客户端                |
| `getData.py`      | 配置数据集            |
| `Models.py`       | 模型配置              |
| `requirements.txt`| 项目依赖              |
| `server.py`       | 服务器端/执行         |
