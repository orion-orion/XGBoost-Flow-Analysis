import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import numpy as np
def get_top_n_features(train_X, train_Y, top_n_features):
    
    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=-1, cv=10, verbose=1)
    rf_grid.fit(train_X, np.array(train_Y).ravel())
    joblib.dump(rf_grid,'model/rf_grid.json')
    # AdaBoost
    ada_est =AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=-1, cv=10, verbose=1)
    ada_grid.fit(train_X, np.array(train_Y).ravel())
    joblib.dump(ada_grid,'model/ada_grid.json')
    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)
    et_grid.fit(train_X, np.array(train_Y).ravel())
    joblib.dump(et_grid,'model/et_grid.json')
     
    # GradientBoosting
    gb_est =GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=-1, cv=10, verbose=1)
    gb_grid.fit(train_X, np.array(train_Y).ravel())
    joblib.dump(gb_grid,'model/gb_grid.json')
  
    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=-1, cv=10, verbose=1)
    dt_grid.fit(train_X, np.array(train_Y).ravel())
    joblib.dump(dt_grid,'model/dt_grid.json')
      
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
  
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(train_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    # 将多个模型选择出的的特征融合
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada,features_top_n_et,features_top_n_gb,features_top_n_dt], 
                               ignore_index=True).drop_duplicates()
    
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada,feature_imp_sorted_et,feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)
    
    return features_top_n , features_importance
