import pandas as pd
import joblib
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
import xgboost as xgb
import numpy as np
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from scipy import sparse
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
def feature_eng(X_train_sparse, Y_train, X_valid_sparse,Y_valid,X_test_sparse):

    '''#Z-score标准化，统一量纲
    ss=StandardScaler()
    train_X=ss.fit_transform(train_X) 
    test_X=ss.transform(test_X)
    
    #数据归一化，归一到0-1区间，提高模型收敛速度      
    ms=MinMaxScaler()
    train_X=ms.fit_transform(train_X)
    test_X=ms.transform(test_X) 

    #Normalizer 基于矩阵的行将样本向量转换为单位向量
    norm=Normalizer()
    train_X=norm.fit_transform(train_X)
    test_X=norm.transform(test_X)

    #移除低方差特征i
    sel = VarianceThreshold(threshold=(0.01))
    train_X=sel.fit_transform(train_X)
    test_X=sel.transform(test_X)
    print(train_X.shape)
   
    #卡方特征选择
    sk=SelectKBest(chi2,k=1000)
    train_X = sk.fit_transform(train_X,train_Y)
    test_X = sk.transform(test_X)
    print(train_X.shape)
    
    #PCA降维
    pca=PCA(n_components=100)
    train_X=pca.fit_transform(train_X)
    test_X=pca.fit_transform(test_X)
    
    #LDA判别分析
    lda = LinearDiscriminantAnalysis(n_components=10)
    train_X=lda.fit_transform(train_X,train_Y)
    test_X =lda.transform(test_X)

    '''
    # 初步训练模型，准备特征选择
    xgb_est =XGBClassifier(random_state=0)
    xgb_param_grid = {'n_estimators': [100],'gamma':[0.9],'subsample':[1],'learning_rate':[0.05],\
    'colsample_bytree':[0.6],'objective':['multi:softmaix class=3']}
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    xgb_grid = model_selection.GridSearchCV(xgb_est, xgb_param_grid, cv=kfold,n_jobs=-1,verbose=1)
    #xgb_grid.fit(X_train_sparse.todense()[:300], Y_train[:300])
    #joblib.dump(xgb_grid,"model/feature_xgb_grid.json")
    xgb_grid=joblib.load("model/feature_xgb_grid.json")
    Y_valid_pred = xgb_grid.predict(X_valid_sparse.todense())
    predictions = [round(value) for value in Y_valid_pred]
    accuracy = accuracy_score(Y_valid, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


    
    # Fit model using each importance as a threshold
    thresholds = sort(xgb_grid.best_estimator_.feature_importances_)
    thresholds=list(set(thresholds))
    max_accuracy=-1
    best_thresh=0
    best_num_feature=0
    for i,thresh in enumerate(thresholds):
	# select features using threshold
        selection = SelectFromModel(xgb_grid.best_estimator_, threshold=thresh, prefit=True)
        X_train_selected = selection.transform(X_train_sparse.todense())
        X_valid_selected = selection.transform(X_valid_sparse.todense())
        s_xgb_est =XGBClassifier(random_state=0)
        s_xgb_param_grid = {'n_estimators': [10],'gamma':[0.9],'subsample':[1],'learning_rate':[0.05],\
        'colsample_bytree':[0.6],'objective':['multi:softmaix class=3']}
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        s_xgb_grid = model_selection.GridSearchCV(s_xgb_est, s_xgb_param_grid, cv=kfold,n_jobs=-1,verbose=1)
        #s_xgb_grid.fit(X_train_selected[100:], Y_train[100:])
        #joblib.dump(s_xgb_grid,"model/s_xgb"+str(i)+"_grid.json")
        s_xgb_grid=joblib.load("model/s_xgb"+str(i)+"_grid.json")
        Y_valid_pred = s_xgb_grid.predict(X_valid_selected)
        predictions = [round(value) for value in Y_valid_pred]
        accuracy = accuracy_score(Y_valid, predictions)
        print("iter=%d,Thresh=%.3f, num_feature=%d, Accuracy: %.2f%%" % (i,thresh, X_train_selected.shape[1], accuracy*100.0))
        if accuracy>max_accuracy:
           max_accuracy=accuracy
           best_thresh=thresh
           best_num_feature=X_train_selected.shape[1]
    print("best Thresh=%.3f, best_num_feature=%d, best_accuracy: %.2f%%" % (best_thresh,best_num_feature, max_accuracy*100.0))
    selection = SelectFromModel(xgb_grid.best_estimator_, threshold=best_thresh, prefit=True)
    X_train_sparse= sparse.csc_matrix(selection.transform(X_train_sparse.todense()))
    X_valid_sparse= sparse.csc_matrix(selection.transform(X_valid_sparse.todense()))
    X_test_sparse= sparse.csc_matrix(selection.transform(X_test_sparse.todense()))
