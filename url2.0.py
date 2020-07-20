import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from html.parser import HTMLParser
import re
from urllib import parse
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
from sklearn import model_selection
import matplotlib.pyplot as plt
def DecodeQuery(fileName):
    data = [x.strip() for x in open(fileName, "r").readlines()]
    query_list = []
    for item in data:
        item = item.lower()
        # if len(item) > 50 or len(item) < 5:
        #     continue        
        h = HTMLParser()
        item = h.unescape(item)
        item = parse.unquote(item)
        item, number = re.subn(r'\d+', "8", item)
        item, number = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?:]+', "http://u", item)
        query_list.append(item)
    return list(set(query_list))
    
def get_top_n_features(train_X, train_Y, top_n_features):
    # # random forest
    # rf_est = RandomForestClassifier(random_state=0)
    # rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    # rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    # rf_grid.fit(train_X, train_Y)
    # joblib.dump(rf_grid,'model/rf_grid.json')

    # # AdaBoost
    # ada_est =AdaBoostClassifier(random_state=0)
    # ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    # ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    # ada_grid.fit(train_X, train_Y)
    # joblib.dump(ada_grid,'model/ada_grid.json')

    # # ExtraTree
    # et_est = ExtraTreesClassifier(random_state=0)
    # et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    # et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    # et_grid.fit(train_X, train_Y)
    # joblib.dump(et_grid,'model/et_grid.json')

    # # GradientBoosting
    # gb_est =GradientBoostingClassifier(random_state=0)
    # gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    # gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    # gb_grid.fit(train_X, train_Y)
    # joblib.dump(gb_grid,'model/gb_grid.json')

    # # DecisionTree
    # dt_est = DecisionTreeClassifier(random_state=0)
    # dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    # dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    # dt_grid.fit(train_X, train_Y)
    # joblib.dump(dt_grid,'model/dt_grid.json')

    rf_grid=joblib.load('model/rf_grid.json')
    ada_grid=joblib.load('model/ada_grid.json')
    et_grid=joblib.load('model/et_grid.json')
    gb_grid=joblib.load('model/gb_grid.json')
    dt_grid=joblib.load('model/dt_grid.json')

    feature_imp_sorted_rf = pd.DataFrame({'feature': list(train_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']

    feature_imp_sorted_ada = pd.DataFrame({'feature': list(train_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']

    feature_imp_sorted_et = pd.DataFrame({'feature': list(train_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']

    feature_imp_sorted_gb = pd.DataFrame({'feature': list(train_X),
                                           'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']

    feature_imp_sorted_gb = pd.DataFrame({'feature': list(train_X),
                                           'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']

    feature_imp_sorted_dt = pd.DataFrame({'feature': list(train_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']

    # 将3个模型融合
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt], 
                               ignore_index=True).drop_duplicates()
    
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, 
                                   feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)
    
    return features_top_n , features_importance

#十折交叉验证
def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

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

if __name__ =='__main__':
    #读取训练集数据
    vectorizer =TfidfVectorizer(ngram_range=(1,3))
    bX_d = DecodeQuery('./data/badx.txt')
    gX_d = DecodeQuery('./data/goodx.txt')
    gY = [1]*len(gX_d)   #正常请求标签为1
    bY = [0]*len(bX_d)   #恶意请求标签为0 
    X_train=pd.DataFrame(vectorizer.fit_transform(bX_d+gX_d).todense())
    Y_train=bY+gY
    #读取测试集数据
    X_test_d=DecodeQuery('./data/testx.txt')
    X_test =pd.DataFrame(vectorizer.transform(X_test_d).todense())
    print(X_train.shape,len(Y_train))
    #用模型进行特征选择
    feature_to_pick = 30
    feature_top_n, feature_importance = get_top_n_features(X_train, Y_train, feature_to_pick)

    #用筛选出的特征重新构建训练集和测试集
    X_train = pd.DataFrame(X_train[feature_top_n])
    X_test = pd.DataFrame(X_test[feature_top_n])

    #stack第一层

    # Some useful parameters which will come in handy later on
    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]
    SEED = 0 # for reproducibility
    NFOLDS = 7 # set folds for out-of-fold prediction
    kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)
    dt = DecisionTreeClassifier(max_depth=8)
    knn = KNeighborsClassifier(n_neighbors = 2)
    svm = SVC(kernel='linear', C=0.025)

    # numpy 转arrays:
    x_train = np.array(X_train)
    x_test = np.array(X_test)
    y_train =np.array(Y_train)

   # Create our OOF train and test predictions. These base results will be used as new features
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest
    ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost 
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees
    gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost
    dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree
    knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test) # KNeighbors
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector

    #我们利用XGBoost，使用第一层预测的结果作为特征对最终的结果进行预测。
    x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

    gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                     colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
    predictions = gbm.predict(x_test)

    StackingSubmission = pd.DataFrame({'流量url': X_test_d, '是否是业务流量': predictions}) 
    StackingSubmission.to_csv('./data/testy.csv',index=False,sep=',') 

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
    title = "Learning Curves"
    plot_learning_curve(RandomForestClassifier(**rf_parameters), title, x_train, y_train, cv=None,  n_jobs=4, train_sizes=[50,200])
    plt.show()

    # #进行推断
    # y_hat=gbm.predict(x_train)
    # res_list = []

    # #打印结果
    # tot=r=0
    # for url , y,y_hat in zip(x_train,y_train,y_hat):
    #     if y ==  y_hat:
    #         r=r+1
    #     tot=tot+1
    # print("准确度:%f"%(r/tot))
        
        



    