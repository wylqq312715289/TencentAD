#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor

from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('train_modified.csv')
target = 'Disbursed'
IDcol = 'ID'

from featureProject.features1 import make_train_set
from featureProject.my_import import split_libsvm_data
from featureProject.my_import import feature_importance2file

model_path = "./model/my_gbm.model" #模型保存的地址

def logloss(act, pred):
    # act和pred都是list类型
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def cv_model(train_samples_df,libsvm_dump_path):
    dtrain = lgb.Dataset(libsvm_dump_path)
    estimator = LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
    )
    params_grid = {
        'num_leaves':range(31,300,50)
    }
    gsearch = GridSearchCV( estimator , param_grid = params_grid, scoring='roc_auc', cv=5 )
    gsearch.fit(dtrain);
    print gsearch.best_params_
    #history = lgb.cv(params, dtrain, num_boost_round=20, nfold=5)
    #print pd.DataFrame(history);


def lightGBM_make_submission():
    if os.path.exists(model_path): os.remove(model_path); print "Remove file %s."%(model_path)
    libsvm_train_dump_path = './cache/libsvm_train_dump_path.txt'
    libsvm_eval_dump_path = './cache/libsvm_eval_dump_path.txt'
    #################### 训练数据 ##############################################
    train_samples_df, libsvm_dump_path = make_train_set(train_step=True)
    # train_data_df = train_data_df.iloc[:,:205]
    # 改这里要执行如下两行语句
    #if os.path.exists(libsvm_train_dump_path): os.remove(libsvm_train_dump_path)
    cv_model(train_samples_df,libsvm_dump_path); exit();

if __name__ == '__main__':
    lightGBM_make_submission()
