<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2022-07-03 15:09:10
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-07-03 19:14:58
-->
# 基于XGBoost的流量分析识别系统

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/XGBoost-Flow-Analysis)
[![](https://img.shields.io/github/license/orion-orion/XGBoost-Flow-Analysis)](https://github.com/orion-orion/XGBoost-Flow-Analysis/blob/master/LICENSE)
[![](https://img.shields.io/github/stars/orion-orion/XGBoost-Flow-Analysis?style=social)](https://github.com/orion-orion/XGBoost-Flow-Analysis)
[![](https://img.shields.io/github/issues/orion-orion/TipDMCup21)](https://github.com/orion-orion/XGBoost-Flow-Analysis/issues)

### 关于本项目

本项目为2020年中国高校计算机大赛(C4)－网络技术挑战赛EP2初赛赛项，题目为构建一个在线流量识分析与识别系统，能够实时识别出网络上的正常业务流量、恶意软件流量和网络攻击流量，并对各种流量随时序的变化进行进行可视化展示，我们在XGboost模型的基础上使用Stacking集成学习技术，将思博伦官方给出的流量pcap包解析为流量的URL进行训练, 最终在官方给出的测试流量包上达到 82.18% 的准确率。（该项目的[决赛版本](https://github.com/orion-orion/CNN-LSTM-Flow-Analysis)将模型换为了时空神经网络）。

### 环境依赖
运行以下命令安装环境依赖：
```
pip install -r requirements.txt
```

### 数据集
我们的数据读取和预处理(TF-IDF编码)逻辑由在`process.py`模块完成
训练所用的数据集采用赛方提供的pcap包数据集，已经经过Scapy的解析将流量的URL提取出来，放在项目目录中的`data`文件夹下，三种类型的流量分别存放为`业务流量.csv`、`恶意软件.csv`、`网络攻击.csv`。
  
因为我们采用的是线上实时分析系统，线上实时测试数据需要从MySQL数据库中读取，经过模型的推断后再在前端可视化呈现。我们这里为了方便已经将MySQL中的已经经过Scapy解析的URL流量数据提取出来存放在`data`目录下，将流量内容和时间戳分别保存为`时间戳.csv`和`测试流量.csv`。

### 特征工程
特征工程逻辑由`feature.py`模块完成，详情可阅读项目根目录下的项目介绍PDF文档。

### 模型
我本地训练好的特征选择模型和主模型已经分别保存在项目的`features_model`目录和`model`目录下，可直接进行测试。

### 项目目录说明
-data  -------------------  存放数据  

-features_model  -------------------  存放用于特征选择的相关模型

-model  -------------------  存放最终训练的相关主模型 

-prediction  -------------------  存放对第6年数据的预测结果

-feature.py  -------------------  完成特征选择的操作（包含尝试依据不同的阈值选取特征并训练模型）  

-main.py  -------------------  主文件，用于从构建特征工程到模型的训练、评估与推断的pipline  

-model.py ------------------- 在训练集和验证集上进行关于主模型的训练与验证的实现  


-process.py  ------------------- 完成数据的读取与预处理操作

### 使用方法
运行:

```
python main.py \
    --features_model load \
    --main_model load 
```

`features_model`参数表示选择是否重新开始训练特征选择模型，若需重新训练特征选择模型可将 `feature_model `参数设置为 `retrain `，否则设置为 `load`直接加载已经训练好的特征选择模型(（但前提是特征选择模型已经放置于 `features_model`目录下）)。

`main_model`参数表示是否重新开始训练主模型（即Stacking+XGBoost模型），若需重新训练主模型可将 `main_model `参数设置为 `retrain `，否则设置为 `load`直接加载已经训练并保存好的模型（但前提是主模型已经放置于 `model`目录下）。

上线后的代码是从MySQL中读取测试数据集，这里为了方便我们已经将经过Scapy解析的测试数据集放在`data`目录下，在模型训练与验证完成之后，会直接进行推断操作。


