#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys,psutil
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem


from featureProject.features import make_train_set
from featureProject.my_import import split_data
from featureProject.features import TencentReport
from featureProject.my_import import feature_importance2file


model_path = "./model/my_gbm.model" #模型保存的地址
feature_importance_path = "./logs/feature_importence.csv"


def train_gbm_model( samples_df, data_df, model_type="train" ):
    print "all_train_data size = %d * %d."%(len(samples_df.index),len(data_df.columns))
    ################################## 评估数据 #########################################
    all_samples_train = lgb.Dataset(data_df.values, samples_df['label'].values)
    scale_pos_weight = 1.0*len(samples_df[samples_df['label']<=0.1].index) / len(samples_df[samples_df['label']>=0.9].index);
    print "scale_pos_weight=%f"%(scale_pos_weight);
    params = { 
        'boosting_type': 'gbdt',  
        'task': 'train',
        'objective': 'regression',
        'num_leaves': 50,
        'max_depth':13,         
        'learning_rate': 0.1,
        'verbose': 0, 
        'n_estimators': 400, 
        'metric': {'auc','binary_logloss'},  
        'scale_pos_weight': scale_pos_weight, 
        'device':'gpu',
        'num_threads':7,
    }
    num_boost_round = 500
    print ('获取内存占用率： '+(str)(psutil.virtual_memory().percent)+'%')
    if model_type == "cv":
        print "start cv, please wait ........"
        cv_history = lgb.cv(params, all_samples_train, num_boost_round, nfold=5, metrics = {"binary_logloss"}, early_stopping_rounds= 10, callbacks=[lgb.callback.print_evaluation(show_stdv=True)]);
        history_df = pd.DataFrame(cv_history)
        num_boost_round = len(history_df.index)
    else: num_boost_round = 63
    bst = lgb.train(params, all_samples_train, num_boost_round, valid_sets=all_samples_train, early_stopping_rounds=1000 )
    bst.save_model(model_path)
    feature_importance2file(bst, feature_importance_path, model_name='gbm')
    return bst

def lightGBM_make_submission(model_type):
    if os.path.exists(model_path): os.remove(model_path); print "Remove file %s."%(model_path)
    ######################### 训练数据 #########################
    print ('获取内存占用率： '+(str)(psutil.virtual_memory().percent)+'%')
    samples_df, data_df = make_train_set(train_step=True)
    ###################### 随机打乱数据 #########################
    #idx = np.random.permutation(samples_df.index)
    #samples_df = samples_df.iloc[idx].reset_index(drop=True)
    #data_df = data_df.iloc[idx].reset_index(drop=True)
    #############################################################
    print ('获取内存占用率： '+(str)(psutil.virtual_memory().percent)+'%')
    if os.path.exists(model_path):  bst = lgb.Booster(model_file=model_path)
    else:                           bst = train_gbm_model(samples_df, data_df, model_type)
    ############################## 查看正确率 #############################################
    test_size = 0.15
    split_line = int((1.0-test_size)*len(data_df.index))
    y = bst.predict( data_df.iloc[:split_line,:].values )
    samples_df['prob'] = pd.Series(y)
    TencentReport( samples_df.iloc[:split_line,:][['label','prob']] )
    samples_df, data_df = None, None #优化内存
    ############################ 生成提交文件 ###################################
    sub_samples_df, sub_data_df = make_train_set(train_step=False)
    y = bst.predict( sub_data_df.values )
    sub_samples_df['prob'] = pd.Series(y)
    pre_df = sub_samples_df[['instanceID','prob']].sort_values('instanceID',ascending=False).groupby('instanceID', as_index=False).first()
    pre_df.to_csv('./sub/submission.csv', index=False, index_label=False)

if __name__ == '__main__':
    lightGBM_make_submission(model_type="cv")
