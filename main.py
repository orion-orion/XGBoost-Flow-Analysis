from process import readFile
from feature import feature_eng
from model import train
import pandas as pd
import numpy as np
import pymysql
import xgboost as xgb
from sklearn.metrics import accuracy_score
if __name__ =='__main__':
    # db=pymysql.connect(
    #   host='47.93.50.246',
    #   user='root',
    #   passwd='123',
    #   db='EP2',
    #   charset='utf8',
    #   port=3306,
    #   autocommit=True
    # )
    db="test"
    #读取训数据并构建训练集特征矩阵，训练集标签，测试集特征矩阵
    X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse=readFile(db)
    #获取top 30 的特征对应的下标
    feature_eng(X_train_sparse, Y_train, X_valid_sparse,Y_valid,X_test_sparse)
    #模型训练与交叉验证
    cls,X_test=train(X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse)
    
    #模型推断并写入结果
    Y_test_pred = cls.predict(X_test)
    predictions = [round(value) for value in Y_test_pred]
    Y_test=len(Y_test_pred)*[0]
    accuracy = accuracy_score(Y_test, predictions)
    print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

    id=np.arange(0,len(predictions),1)
    StackingSubmission = pd.DataFrame({'ID':id,'label': predictions}) 
    StackingSubmission.to_csv('data/testy.csv',index=False,sep=',') 
    #cursor=db.cursor()
    #cursor.execute('use EP2')
    #cursor.execute(""" set sql_mode=""; """)
    #cursor.execute(r""" load data infile '/var/lib/mysql-files/testy.csv' into table flow ; """) 
    #按照原本的训练集进行推断查看精度
    #进行推断

    #db.close()
    
