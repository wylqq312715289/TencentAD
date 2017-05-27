#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys
import xgboost as xgb
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem

from featureProject.features import make_train_set
from featureProject.features import make_test_set
from featureProject.my_import import split_data

def report( right_list, pre_list ):
    epsilon = 1e-15
    act = right_list
    pred = sp.maximum(epsilon, pre_list)
    pred = sp.minimum(1-epsilon, pre_list)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def xgboost_make_submission():
    #################### 训练数据 #####################
    train_samples_df, train_data_df = make_train_set()
    train_samples_df,train_data_df,test_samples_df,test_data_df = split_data(train_samples_df, train_data_df, 0.15)
    train_samples_df,train_data_df,eval_samples_df,eval_data_df = split_data(train_samples_df, train_data_df, 0.15)
    ################################## 评估数据 #########################################
    dtrain = xgb.DMatrix(train_data_df,label = train_samples_df['label'])
    deval = xgb.DMatrix(eval_data_df,label = eval_samples_df['label'])
    scale_pos_weight = 1.0*len(train_samples_df[train_samples_df['label']<=0.1].index)/len(train_samples_df[train_samples_df['label']>=0.9].index);
    print "scale_pos_weight=%f"%(scale_pos_weight);
    param = { 'learning_rate' : 0.1,    'n_estimators': 1000,             'max_depth': 4, 
              'min_child_weight': 5,    'gamma': 0,                       'subsample': 1.0, 
              'colsample_bytree': 0.8,  'eta': 0.05,                      'eval_metric':'auc',
              'silent': 1,       'scale_pos_weight':scale_pos_weight,     'objective': 'binary:logistic' }
    epochs = 283 #训练的轮数
    plst = param.items(); plst += [('eval_metric', 'logloss')]
    evallist = [ (dtrain, 'train'),(deval, 'eval')]
    bst = xgb.train(plst, dtrain, epochs, evallist)
    ############################## 查看正确率 #############################################
    y = bst.predict( xgb.DMatrix(test_data_df,label = test_samples_df['label']) )
    print report(test_samples_df['label'].values,y)
    ############################ 生成提交文件 ###################################
    sub_samples_df, sub_data_df = make_test_set()
    y = bst.predict( xgb.DMatrix(sub_data_df) )
    sub_samples_df['prob'] = pd.Series(y)
    pre_df = sub_samples_df[['instanceID','prob']].sort_values('instanceID',ascending=False).groupby('instanceID', as_index=False).first()
    pre_df.to_csv('./sub/submission.csv', index=False, index_label=False)

if __name__ == '__main__':
    xgboost_make_submission()
