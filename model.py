from __future__ import division
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
def train(X_train,Y_train,X_test):
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    SEED = 10 # 
    NFOLDS = 10 # 采用十折交叉验证
    kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=True)

    #准备第一层的模型
    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)
    dt = DecisionTreeClassifier(max_depth=8)
    knn = KNeighborsClassifier(n_neighbors = 10)
    svm = SVC(kernel='linear', C=0.025)

    # numpy 转arrays:
    x_train = np.array(X_train)
    x_test = np.array(X_test)
    y_train =Y_train

   # 采用七折交叉验证的模型第一层，除了train之外，我们还需要将test传入，并将其预测结果在第二层做进一步预测
   
    rf_oof_trainy_hat, rf_oof_testy_hat = get_out_fold("rf",rf, x_train, y_train, x_test,ntrain,ntest,NFOLDS,kf) # Random Forest
    ada_oof_trainy_hat,ada_oof_testy_hat = get_out_fold("ada",ada, x_train, y_train, x_test,ntrain,ntest,NFOLDS,kf) # AdaBoost 
    et_oof_trainy_hat,et_oof_testy_hat = get_out_fold("et",et, x_train, y_train, x_test,ntrain,ntest,NFOLDS,kf) # Extra Trees
    gb_oof_trainy_hat,gb_oof_testy_hat = get_out_fold("gb",gb, x_train, y_train, x_test,ntrain,ntest,NFOLDS,kf) # Gradient Boost
    dt_oof_trainy_hat,dt_oof_testy_hat = get_out_fold("dt",dt, x_train, y_train, x_test,ntrain,ntest,NFOLDS,kf) # Decision Tree
    knn_oof_trainy_hat,knn_oof_testy_hat = get_out_fold("knn",knn, x_train, y_train, x_test,ntrain,ntest,NFOLDS,kf) # KNeighbors
    svm_oof_trainy_hat,svm_oof_testy_hat = get_out_fold("svm",svm, x_train, y_train, x_test,ntrain,ntest,NFOLDS,kf) # Support Vector

    #我们利用XGBoost，使用第一层预测七折交叉验证中的验证集预测结果rf_oof_train等作为特征对最终的结果进行预测
    x_train = np.concatenate((rf_oof_trainy_hat,ada_oof_trainy_hat,et_oof_trainy_hat,gb_oof_trainy_hat,dt_oof_trainy_hat,knn_oof_trainy_hat,svm_oof_trainy_hat), axis=1)
    x_test = np.concatenate((rf_oof_testy_hat,ada_oof_testy_hat,et_oof_testy_hat,gb_oof_testy_hat,dt_oof_testy_hat,knn_oof_testy_hat,svm_oof_testy_hat), axis=1)
    gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8,learning_rate=0.01, 
                     colsample_bytree=0.8, objective= 'multi:softmax class=3', nthread= -1).fit(x_train, y_train.ravel())
    joblib.dump(gbm,"model/gbm.json")
    gbm=joblib.load("model/gbm.json")
    
    '''
    #观察不同的学习曲线
    # RandomForest
    rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2, 
                'max_features' : 'sqrt','verbose': 0}
    # AdaBoost
    ada_parameters = {'n_estimators':500, 'learning_rate':0.1}
    # ExtraTrees
    et_parameters = {'n_jobs': -1, 'n_estimators':500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}
    # GradientBoosting
    gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}
    # DecisionTree
    dt_parameters = {'max_depth':8}
    # KNeighbors
    knn_parameters = {'n_neighbors':2}
    # SVM
    svm_parameters = {'kernel':'linear', 'C':0.025}
    # XGB
    gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma':0.9, 'subsample':0.8, 
                'colsample_bytree':0.8, 'objective': 'binary:logistic', 'nthread':-1, 'scale_pos_weight':1}
i    title = "Learning Curves"
    plot_learning_curve(RandomForestClassifier(**rf_parameters), title, x_train, y_train, cv=None,  n_jobs=4, train_sizes=[50,200])
    plt.show()
    '''
    return gbm,x_train,x_test  #x_train后续验证用
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
        clf.fit(x_tr, y_tr.ravel())

        filename="model/"+clf_name+str(i)+".json"
        joblib.dump(clf,filename)
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
    print("{} 测试集精度:{:.5f}".format(clf_name,dts_test*100))
    oof_test[:] = oof_test_skf.mean(axis=0)#取所有次迭代的平均值
    return oof_trainy_hat.reshape(-1, 1), oof_test.reshape(-1, 1)

#学习曲线绘制
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0): #绘制学习曲线 
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
