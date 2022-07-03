'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-07-03 15:09:10
LastEditors: ZhangHongYu
LastEditTime: 2022-07-03 16:50:38
'''
from process import readFile
from feature import feature_eng
from model import train
import pandas as pd
import numpy as np
import argparse
import os

def parse_args():
    """parse the command line args

    Returns:
        args: a namespace object including args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--features_model',
        help="whether use the features selected or retrain the model to select the features"
        " possible are `load`,`ratrain`",
        type=str,
        default='load'
    )
    parser.add_argument(
        '--main_model',
        help="whether use the main model trained or retrain the main model"
        " possible are `load`,`ratrain`",
        type=str,
        default='retrain'
    )

    args = parser.parse_args()
    return args

prediction_direct = "prediction"

if __name__ == '__main__':
    args = parse_args()

    # 读取训数据并构建训练集特征矩阵，训练集标签，测试集特征矩阵
    X_train_sparse, Y_train, X_valid_sparse, Y_valid, X_test_sparse, time = readFile()

    # 特征提取
    feature_eng(X_train_sparse, Y_train,
                X_valid_sparse, Y_valid, X_test_sparse, mod=args.features_model)

    # 模型训练与交叉验证
    cls, X_test = train(X_train_sparse, Y_train,
                        X_valid_sparse, Y_valid, X_test_sparse, mod=args.main_model)

    # 模型推断并写入结果
    Y_test_pred = cls.predict(X_test)
    predictions = [round(value) for value in Y_test_pred]
    print("成功完成推断!")

    id = np.arange(0, len(predictions), 1)
    StackingSubmission = pd.DataFrame(
        {'id': id, 'label': predictions, 'time': time})
    StackingSubmission.to_csv(os.path.join(prediction_direct, "testy.csv"), index=False, sep=',')
