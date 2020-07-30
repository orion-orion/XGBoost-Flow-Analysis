from process import readFile
from feature import feature_eng
from model import train
import pandas as pd
import numpy as np
import pymysql
import xgboost as xgb
from sklearn.metrics import accuracy_score
if __name__ =='__main__':
    print("******推断程序开始******")
    db=pymysql.connect(
       host='47.93.50.246',
       user='root',
       passwd='123',
       db='EP2',
       charset='utf8',
       port=3306,
       autocommit=True
    )
    #读取训数据并构建训练集特征矩阵，训练集标签，测试集特征矩阵
    X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse,time=readFile(db)
    #特征提取
    feature_eng(X_train_sparse, Y_train, X_valid_sparse,Y_valid,X_test_sparse)
    #模型训练与交叉验证
    cls,X_test=train(X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse)
    
    #模型推断并写入结果
    Y_test_pred = cls.predict(X_test)
    predictions = [round(value) for value in Y_test_pred]
    print("成功完成推断!")

    id=np.arange(0,len(predictions),1)
    StackingSubmission = pd.DataFrame({'id':id,'label': predictions,'time':time}) 
    StackingSubmission.to_csv('/var/lib/mysql-files/testy.csv',index=False,sep=',') 
    cursor=db.cursor()
    cursor.execute('use EP2')
    cursor.execute(""" set sql_mode=""; """)
    cursor.execute(r""" load data infile '/var/lib/mysql-files/testy.csv' into table flow ; """) 
    print("成功将推断结果写入数据库!")
    db.close()
    
