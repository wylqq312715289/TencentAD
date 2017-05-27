#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys,psutil
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem

from sklearn import  metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

from featureProject.features import make_train_set
from featureProject.my_import import split_data
from featureProject.features import TencentReport
from featureProject.my_import import feature_importance2file


def print_best_score(gsearch,param_test):
     # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def lightGBM_CV():
    print ('获取内存占用率： '+(str)(psutil.virtual_memory().percent)+'%')
    samples_df, data_df = make_train_set(train_step=True)
    labels =  samples_df['label'].values;samples_df=None
    values = data_df.values; data_df = None;
    param_test = {
        'max_depth': range(5,15,2),
    }
    estimator = LGBMRegressor(
        num_leaves = 50, # cv调节50是最优值
        max_depth = 13,
        learning_rate =0.1, 
        n_estimators = 140, 
        objective = 'regression', 
        min_child_weight = 1, 
        subsample = 0.8,
        colsample_bytree=0.8,
        nthread = 7,
    )
    gsearch = GridSearchCV( estimator , param_grid = param_test, scoring='roc_auc', cv=5 )
    gsearch.fit( values, labels )
    gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
    print_best_score(gsearch,param_test)


if __name__ == '__main__':
    lightGBM_CV()
