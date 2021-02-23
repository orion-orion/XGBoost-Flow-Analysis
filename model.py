from __future__ import division
import numpy as np
import joblib
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
def train(X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse):
    #准备Stacking第一层的模型
    #我们利用XGBoost，使用Stacking第一层中所有基分类器验证集的预测结果Y_valid_pred作为特征对最终的结果进行预测
    Y_valid_pred1,Y_valid1,Y_test_pred1=xgb_base(1,X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse)
    Y_valid_pred2,Y_valid2,Y_test_pred2=xgb_base(2,X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse)
    Y_valid_pred3,Y_valid3,Y_test_pred3=xgb_base(3,X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse)
    Y_valid_pred4,Y_valid4,Y_test_pred4=xgb_base(4,X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse)
    X_train = np.concatenate((Y_valid_pred1,Y_valid_pred2,Y_valid_pred3,Y_valid_pred4), axis=1)
    Y_train = np.concatenate((Y_valid1,Y_valid2,Y_valid3,Y_valid4), axis=1)
    X_test = np.concatenate((Y_test_pred1,Y_test_pred2,Y_test_pred3,Y_test_pred4), axis=1)
   
    #Stacking第二层，使用XGBoost分类器
    cls = XGBClassifier( n_estimators= 10000, gamma=0.9, subsample=1,learning_rate=0.05, 
                    colsample_bytree=0.6, objective= 'multi:softmaix class=3', nthread= -1).fit(X_train, Y_valid.ravel()) 
    joblib.dump(cls,"model/xgb.json")
    cls=joblib.load("model/xgb.json")
 
    return  cls,X_test
def xgb_base(id,X_train_sparse,Y_train,X_valid_sparse,Y_valid,X_test_sparse):
    params={
    'booster':'gbtree',
    #这里分类数字是0-3，是一个多类的问题，因此采用了multisoft多分类器，
    'objective': 'multi:softmax',
    'num_class':3, # 类数，与 multisoftmax 并用
    'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    #'max_depth':12, # 构建树的深度 [1:]
    #'lambda':450,  # L2 正则项权重
    'subsample':0.4, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
    #'min_child_weight':12, # 节点的最少特征数
    'eta': 0.005, # 如同学习率
    'seed':710,
    'nthread':4,# cpu 线程数,根据自己U的个数适当调整
    }
    plst=list(params.items())
    num_rounds = 500 # 迭代你次数

    #划分训练集与验证集
    xgtrain = xgb.DMatrix(X_train_sparse, Y_train,missing=0)
    xgval = xgb.DMatrix(X_valid_sparse,Y_valid,missing=0)

    #在训练中动态显示训练和验证的错误率
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]

    #开始训练
    cls = xgb.train(plst, xgtrain, num_rounds,watchlist,early_stopping_rounds=100)


    joblib.dump(cls,'model/xgb_base_'+str(id)+'.json')
    cls=joblib.load('model/xgb_base_'+str(id)+'.json')


    #用验证集验证最后结果
    Y_valid_pred = cls.predict(xgb.DMatrix(X_valid_sparse))
    Y_test_pred = cls.predict(xgb.DMatrix(X_test_sparse))
    predictions = [round(value) for value in Y_valid_pred]
    accuracy = accuracy_score(Y_valid, predictions)
    print("%d base_xgb: Accuracy: %.2f%%" % (id,accuracy * 100.0))
    return Y_valid_pred.reshape(-1,1),Y_valid.reshape(-1,1),Y_test_pred.reshape(-1,1)
'''
#十折交叉验证
def get_out_fold(clf_name,clf, x_train, y_train, x_test,ntrain,ntest,NFOLDS,kf):
    oof_trainy_hat=np.zeros((ntrain,1))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, valid_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_valid = x_train[valid_index] #验证集，用于评估模型性能
        y_valid = y_train[valid_index]
        #用train_index对应的数据数据训练模型
        #clf.fit(x_tr, y_tr.ravel())

        filename="model/"+clf_name+str(i)+".json"
        #joblib.dump(clf,filename)
        clf=joblib.load(filename)

        #用valid_index对应的数据数据进行验证
        y_valid_hat=clf.predict(x_valid)
        #将预测结果写回对应的地方
        oof_trainy_hat[valid_index] = y_valid_hat.reshape(-1,1)
             
        #测试集的推断结果用于第二层模型的训练,每次迭代都推断一次        
        oof_test_skf[i, :] = clf.predict(x_test)
    dts=len([1 for y1,y2 in zip(oof_trainy_hat,y_train) if y1==y2])/len(y_train)
    dts_test=len([1 for y in oof_test_skf[i,:] if y==0])/len(oof_test_skf[i,:])
    print("{} k折验证集精度:{:.5f}".format(clf_name,dts*100))
    #print("{} 测试集精度:{:.5f}".format(clf_name,dts_test*100))
    oof_test[:] = oof_test_skf.mean(axis=0)#取所有次迭代的平均值
    return oof_trainy_hat.reshape(-1, 1), oof_test.reshape(-1, 1)
'''
