#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem


from featureProject.features import make_train_set
from featureProject.my_import import split_data

model_path = "./model/my_gbm.model" #模型保存的地址


def lightGBM_make_submission():
    if os.path.exists(model_path): os.remove(model_path); print "Remove file %s."%(model_path)
    #################### 训练数据 #####################
    train_samples_df, train_data_df = make_train_set(train_step=True)
    # train_data_df = train_data_df.iloc[:,:205]
    train_samples_df,train_data_df,test_samples_df,test_data_df = split_data(train_samples_df, train_data_df, 0.15)
    train_samples_df,train_data_df,eval_samples_df,eval_data_df = split_data(train_samples_df, train_data_df, 0.15)


    
    if os.path.exists(model_path): 
        bst = lgb.Booster(model_file=model_path)
    else:
        ################################## 评估数据 #########################################
        dtrain = lgb.Dataset(train_data_df.values, train_samples_df['label'].values)
        deval = lgb.Dataset(eval_data_df.values, eval_samples_df['label'].values)
        dtest = lgb.Dataset(test_data_df.values, test_samples_df['label'].values)
        scale_pos_weight = 1.0*len(train_samples_df[train_samples_df['label']<=0.1].index)/len(train_samples_df[train_samples_df['label']>=0.9].index);
        print "scale_pos_weight=%f"%(scale_pos_weight);
        params = { 'boosting_type': 'gbdt',  'task': 'train',       'objective': 'regression',
                   'num_leaves': 31,         'learning_rate': 0.1,  'feature_fraction': 0.9,
                   'bagging_fraction': 0.8,  'bagging_freq': 5,     'verbose': 0, 'n_estimators':30,
                   'n_estimators': 40,
                   'metric': {'auc','binary_logloss'},  'scale_pos_weight':scale_pos_weight, 'device':'gpu',
        }
        bst = lgb.train(params, dtrain, num_boost_round=400, valid_sets=deval, early_stopping_rounds=5 )
        bst.save_model(model_path)
        ############################## 查看正确率 #############################################
    y = bst.predict( test_data_df.values )
    test_samples_df['prob'] = y
    TencentReport( test_samples_df[['label','prob']] )
    ############################ 生成提交文件 ###################################
    sub_samples_df, sub_data_df = make_train_set(train_step=False)
    y = bst.predict( sub_data_df.values )
    sub_samples_df['prob'] = pd.Series(y)
    # sub_samples_df = set0state(sub_samples_df)
    pre_df = sub_samples_df[['instanceID','prob']].sort_values('instanceID',ascending=False).groupby('instanceID', as_index=False).first()
    pre_df.to_csv('./sub/submission.csv', index=False, index_label=False)

if __name__ == '__main__':
    lightGBM_make_submission()
