#!/usr/bin/env python
# -*- coding: utf-8 -*-
#coding:utf-8
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os,copy,math,time
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem

def one_hot( pd_series,  prefix, max_data ):
    pd_df = pd.DataFrame(pd.Series(pd_series.values), columns=[prefix])
    max_value = int(max_data)+1
    seq = [ prefix+"_%d"%i for i in range(max_value) ]
    values = np.zeros((len(pd_df.index),max_value))
    for i in range(len(pd_df.index)):
        values[i,int(pd_df.values[i])] = 1.0
    
    return pd.DataFrame(values,columns = seq)

def re_onehot( list_x ):
    ans = []
    for i in range(len(list_x)):
        ans.append(list_x[i][1])
    return ans    

# 转换一个实数从10进制到2进制，并放入数组，2进制最大长度为max_value
def decimal2binary(x, max_value):
    ans = [0.0 for i in range(max_value)] 
    tmp = int(x)
    for i in range(max_value):
        ans[i] = 1.0 if tmp%2!=0 else 0.0;
        tmp >>= 1;
    return ans

# 将一列dataFrame数据转换成二进制形式的dataFrame
def pd_decimal2binary( pd_series, max_data, prefix ):
    pd_df = pd.DataFrame(pd.Series(pd_series.values), columns=[prefix])
    max_value = int( np.log2(max_data) ) + 1
    seq = [ prefix+"_%d"%i for i in range(max_value) ]
    values = []
    for i in range(len(pd_df.values)):
        values.append( decimal2binary(pd_df.values[i],max_value) )
    return pd.DataFrame(np.array(values),columns = seq)

# libsvm文件过采样
def over_sample( sample_df, dump_svmlight_file ):
    train_y, train_x = svm_read_problem(dump_svmlight_file)
    with open(dump_svmlight_file,"rb") as f: ans_lines = np.array(f.readlines());
    sample_df['label'] = pd.Series(train_y)
    positive = sample_df[ sample_df['label'] >= 0.99 ].index
    balence = ( len(train_y) - len(positive) ) / len(positive);
    for i in range(balence):
        ans_lines =  np.hstack((ans_lines, ans_lines[positive]))
        sample_df = pd.concat([sample_df, sample_df.iloc[positive,:] ], axis=0)
    new_idx = np.random.permutation(len(ans_lines))
    ans_lines = ans_lines[new_idx]
    sample_df = sample_df.reset_index(drop=True)
    sample_df = sample_df.iloc[new_idx,:].reset_index(drop=True)
    with open(dump_svmlight_file,"wb") as f: f.writelines(list(ans_lines))
    return sample_df[ np.setdiff1d(sample_df.columns,['label']) ], dump_svmlight_file

# 将int型列向量 添加至libsvm文件
def add_Series2libsvm(file_path, value_series, feat_name, max_value, feature_columns, end_feat_idx):
    print "in function add_Series2libsvm. Add feat %-10s.  "%(feat_name),
    feature_columns.extend( [ "%s_%.8d"%(feat_name,i) for i in range(max_value)] )
    with open(file_path,"rb") as f:
        ans_lines = f.readlines();
        values = value_series.values;
        for i in range(len(values)):
            ans_lines[i] = ans_lines[i][:-1]
            ans_lines[i] += " %d:1.0\n"%(end_feat_idx+values[i])
    with open(file_path,"wb") as f: f.writelines(ans_lines)
    print "end_feat_idx = %.10d. end libsvm_add"%(end_feat_idx + max_value)
    return feature_columns, end_feat_idx + max_value

# 向libsvm文件中添加理论特征
def libsvm_add(file_path, new_df, end_feat_idx):
    ans_lines = [];
    with open(file_path,"rb") as f:
        ans_lines = f.readlines();
        values = new_df.values;
        if len(values)!=len(ans_lines): print "libsvm_add error lines num"
        for i in range(len(values)):
            ans_lines[i] = ans_lines[i][:-1] 
            for j in range(len(values[i])):
                if abs( values[i][j] ) < 1e-6 or values[i][j]==None: continue;
                ans_lines[i] += " %d:%f"%(end_feat_idx+j,values[i][j])
            ans_lines[i] += '\n'
    with open(file_path,"wb") as f: f.writelines(ans_lines)
    print "end libsvm_add"
    return end_feat_idx + len(values[0])

#向训练样本中加入特征列
def add_feat2sample( sample_df, new_feat, main_feature, end_feat_idx, libsvm_dump_path, on_what ):
    new_feat = pd.merge(sample_df, new_feat, how='left', on=on_what)
    new_feat = new_feat.replace( np.nan, 0.0 )
    new_feat = new_feat[ np.setdiff1d(new_feat.columns,main_feature) ]
    end_feat_idx = svmlib_add(libsvm_dump_path,new_feat,end_feat_idx);
    print "end_feat_idx=%d"%(end_feat_idx)
    return end_feat_idx

def split_data( samples_df, data_df, test_size=0.2 ):
    samples_num = len(samples_df.index)
    split_line = int( (1.0-test_size) * samples_num )
    train_samples_df = samples_df.iloc[:split_line,:]
    train_data_df = data_df.iloc[:split_line,:]
    test_samples_df = samples_df.iloc[split_line:,:]
    test_data_df = data_df.iloc[split_line:,:]
    return train_samples_df,train_data_df,test_samples_df,test_data_df

def shuffleDF(new_df1,new_df2):
    idx = np.random.permutation(new_df1.index)
    return new_df1.iloc[idx,:],new_df2.iloc[idx,:]

# 删除包含关键字的文件
def files_remove(path="$", key_name="$"): 
    print "start remove files in %s(key name is %s)."%(path,key_name)
    for item in os.listdir(path):  
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) and (key_name in item_path): 
            os.remove(item_path)
            print "rmove file '%s.'"%(item_path)
    print "end remove files."

def full256(data_df, scale=16*16):
    print "start full256 data.shape=%d*%d"%(len(data_df.index),len(data_df.columns))
    for i in range( len(data_df.columns), scale):
        data_df['full256_%d'%(i)] = 0.0
    print "end full256 data.shape=%d*%d"%(len(data_df.index),len(data_df.columns))
    return data_df

def split_libsvm_data( samples_df, libsvm_dump_path, libsvm_train_dump_path, libsvm_eval_dump_path, test_size=0.2 ):
    sub_samples_df = copy.deepcopy(samples_df)
    samples_num = len(sub_samples_df.index)
    split_line = int( (1.0-test_size) * samples_num )
    train_samples_df = sub_samples_df.iloc[:split_line,:]
    test_samples_df = sub_samples_df.iloc[split_line:,:]
    if os.path.exists(libsvm_train_dump_path) and os.path.exists(libsvm_eval_dump_path):
        return train_samples_df,test_samples_df
    with open(libsvm_dump_path,"rb") as f: lines = f.readlines()
    with open(libsvm_train_dump_path,"wb") as f: f.writelines(lines[:split_line])
    with open(libsvm_eval_dump_path,"wb")  as f: f.writelines(lines[split_line:])
    return train_samples_df,test_samples_df


# 画特征重要程度图
def feature_importance2file(bst, file_path, model_name='xgb',feature_columns_sump_path = './logs/feature_name.csv'):
    if model_name=='gbm': 
        importance = bst.feature_importance(importance_type="gain"); # type = array
        df = pd.read_csv(feature_columns_sump_path)
        df['fscore'] = pd.Series(importance)
        df = df.sort_values('fscore',ascending=False).reset_index(drop=True)
        df.to_csv(file_path,index=False,index_label=False)
    else:
        importance = bst.get_fscore() #type = dict()
        importance_df = pd.DataFrame(importance.items(),columns=["feature","fscore"])
        ans_df = pd.read_csv(feature_columns_sump_path)
        ans_df["feature"] = [ "f%d"%i for i in ans_df.index ]
        ans_df = pd.merge(ans_df,importance_df,how="left",on="feature")
        ans_df = ans_df.replace(np.nan,0.0)
        ans_df['fscore'] = 1.0 * ans_df['fscore'] / np.sum( ans_df['fscore'].values )
        ans_df = ans_df[ np.setdiff1d(ans_df.columns,["feature"]) ]
        ans_df = ans_df.sort_values('fscore',ascending=False).reset_index(drop=True)
        ans_df.to_csv(file_path,index=False,index_label=False)

def plot_feature_importance(bst):
    import matplotlib.pylab as plt # python2.7 引入该包会报错
    importance = bst.get_fscore(fmap='xgb.fmap')  
    importance = sorted(importance.items(), key=operator.itemgetter(1))  
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
    df['fscore'] = df['fscore'] / df['fscore'].sum()  
    df.to_csv("../input/feat_sel/feat_importance.csv", index=False)
    plt.figure()  
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
    plt.title('XGBoost Feature Importance')  
    plt.xlabel('relative importance')  
    plt.show() 
