#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from lightgbm.sklearn import LGBMRegressor
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem


from featureProject.features2 import make_train_set
from featureProject.my_import import split_libsvm_data
from featureProject.my_import import feature_importance2file

model_path = "./model/my_gbm.model" #模型保存的地址
test_size = 0.01;

def logloss(act, pred):
    # act和pred都是list类型
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def TencentReport( sub_df ):
    print "len(sub_df)=%d,"%( len(sub_df.index) ),
    print "len(real=1)=%d,"%( np.sum(sub_df['label'].values) ),
    print "len(pre>0.6)=%d,"%( len(sub_df.index) - len(sub_df[sub_df['prob']>=0.6].values) ),
    print "logloss=%f"%logloss(sub_df['label'],sub_df['prob'])

def train_gbm_model(samples_df, libsvm_dump_path, libsvm_test_dump_path,mode="train"):
    ################################## 评估数据 #########################################
    dtrain = lgb.Dataset(libsvm_dump_path)
    dtest = lgb.Dataset(libsvm_test_dump_path)
    scale_pos_weight = 1.0*len(samples_df[samples_df['label']<=0.1].index)/len(samples_df[samples_df['label']>=0.9].index);
    print "scale_pos_weight=%f"%(scale_pos_weight);
    params = {  
        'boosting_type': 'gbdt',
        'task': 'train',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'metric': {'auc','binary_logloss'},
        'is_unbalance': True,
        'scale_pos_weight': scale_pos_weight,
        'device':'cpu',
        'num_threads':6,
    }
    if mode=="cv":
        result = pd.DataFrame(lgb.cv(params, dtrain, num_boost_round=2000, early_stopping_rounds=5, nfold=5 ));
        print result;
        result.to_csv('./logs/cv_result.csv',index=False, index_label=False ); exit();
    bst = lgb.train(params, dtrain, num_boost_round=165, valid_sets=dtest, early_stopping_rounds=5 )
    bst.save_model(model_path)
    return bst;

def lightGBM_make_submission():
    if os.path.exists(model_path): os.remove(model_path); print "Remove file %s."%(model_path)
    libsvm_train_dump_path = './cache/libsvm_train_dump_path.txt'
    libsvm_test_dump_path = './cache/libsvm_test_dump_path.txt'
    #################### 训练数据 ##############################################
    samples_df, libsvm_dump_path = make_train_set(train_step=True)
    # 改test_size这里要执行如下两行语句
    test_size = 0.01; 
    if os.path.exists(libsvm_train_dump_path): os.remove(libsvm_train_dump_path)
    train_samples_df, test_samples_df = split_libsvm_data(
                                                        samples_df,
                                                        libsvm_dump_path, 
                                                        libsvm_train_dump_path, 
                                                        libsvm_test_dump_path, 
                                                        test_size = test_size)
    if os.path.exists(model_path):
        bst = lgb.Booster(model_file=model_path)
    else: 
        #bst = train_gbm_model(train_samples_df, libsvm_train_dump_path, libsvm_test_dump_path, "train")
        bst = train_gbm_model(samples_df, libsvm_dump_path, libsvm_test_dump_path,"train")
    feature_importance2file(bst,'./sub/features_importance.csv','gbm')
    ############################## 查看正确率 #############################################
    y = bst.predict( libsvm_test_dump_path, num_iteration=bst.best_iteration)
    test_samples_df['prob'] = y
    TencentReport( test_samples_df[['label','prob']] )
    ############################ 生成提交文件 ###################################
    sub_samples_df, libsvm_sub_dump_path = make_train_set(train_step=False)
    y = bst.predict( libsvm_sub_dump_path , num_iteration=bst.best_iteration)
    sub_samples_df['prob'] = y
    # sub_samples_df = set0state(sub_samples_df)
    pre_df = sub_samples_df[['instanceID','prob']].sort_values('instanceID',ascending=False).groupby('instanceID', as_index=False).first()
    pre_df.to_csv('./sub/submission.csv', index=False, index_label=False)

if __name__ == '__main__':
    lightGBM_make_submission()
