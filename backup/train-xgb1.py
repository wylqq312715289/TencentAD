#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem


from featureProject.features1 import make_train_set
from featureProject.my_import import split_libsvm_data

model_path = "./model/my_gbm.model" #模型保存的地址

def feature_score(bst):
    feat_imp = pd.Series(bst.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

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


def CV_xgb_model(train_samples_df,libsvm_train_dump_path,libsvm_eval_dump_path):
    pass


def train_xgb_model(train_samples_df,libsvm_train_dump_path,libsvm_eval_dump_path,mode="train"):
    if mode=="cv": CV_xgb_model(train_samples_df,libsvm_train_dump_path,libsvm_eval_dump_path); exit();
    ################################## 评估数据 #########################################
    dtrain = xgb.DMatrix(libsvm_train_dump_path)
    deval = xgb.DMatrix(libsvm_eval_dump_path)
    scale_pos_weight = 1.0*len(train_samples_df[train_samples_df['label']<=0.1].index)/len(train_samples_df[train_samples_df['label']>=0.9].index);
    print "scale_pos_weight=%f"%(scale_pos_weight);
    param = { 
            'eta': 0.05, 
            'eval_metric':'logloss',
            'objective': 'binary:logistic',
            'scale_pos_weight':scale_pos_weight,
    }
    epochs = 30 #训练的轮数
    plst = param.items(); plst += [('eval_metric','auc')]
    evallist = [ (deval, 'eval')] 
    bst = xgb.train( plst, dtrain, epochs, evallist, early_stopping_rounds=5 )
    bst.save_model(model_path)
    return bst;

def lightGBM_make_submission():
    libsvm_train_dump_path = './cache/libsvm_train_dump_path.txt'
    libsvm_eval_dump_path = './cache/libsvm_eval_dump_path.txt'
    #################### 训练数据 ##############################################
    train_samples_df, libsvm_dump_path = make_train_set(train_step=True)
    # if os.path.exists(libsvm_train_dump_path): os.remove(libsvm_train_dump_path)
    train_samples_df,test_samples_df = split_libsvm_data(
                                                        train_samples_df, 
                                                        libsvm_dump_path, 
                                                        libsvm_train_dump_path, 
                                                        libsvm_eval_dump_path, 
                                                        0.01)
    if os.path.exists(model_path): os.remove(model_path); print "Remove file %s."%(model_path)
    if os.path.exists(model_path): bst = xgb.Booster(model_file=model_path)
    else: bst = train_xgb_model(train_samples_df,libsvm_train_dump_path,libsvm_eval_dump_path,"train")
    ############################## 查看正确率 #############################################
    y = bst.predict( xgb.DMatrix(libsvm_eval_dump_path))
    test_samples_df['prob'] = y
    TencentReport( test_samples_df[['label','prob']] )
    ############################ 生成提交文件 ###################################
    sub_samples_df, libsvm_sub_dump_path = make_train_set(train_step=False)
    y = bst.predict( xgb.DMatrix(libsvm_sub_dump_path))
    sub_samples_df['prob'] = pd.Series(y)
    # sub_samples_df = set0state(sub_samples_df)
    pre_df = sub_samples_df[['instanceID','prob']].sort_values('instanceID',ascending=False).groupby('instanceID', as_index=False).first()
    pre_df.to_csv('./sub/submission.csv', index=False, index_label=False)

if __name__ == '__main__':
    lightGBM_make_submission()
