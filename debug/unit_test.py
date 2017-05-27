#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import os,time
from sklearn.datasets import dump_svmlight_file
#import tensorflow as tf
#import lightgbm as lgb
date_max = 312400
fibo_seq = [100,200,600,1200,1*2400,2*2400,4*2400,7*2400] # 点击时间序列

end_date = 270000

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def get_actions( train_step ):
    # train输出 columns = ['label','creativeID','userID','positionID','appID','clickTime']
    # test输出 columns = ['label','creativeID','userID','positionID','appID','clickTime','instanceID']
    if train_step:  dump_path = './cache/train_plus_%d.csv'%(end_date)
    else:           dump_path = './cache/test_plus.csv'
    if os.path.exists(dump_path): actions = pd.read_csv(dump_path)
    else:      
        ad_feat_df = pd.read_csv(ad_path)
        if train_step: # 训练集需要优化最后几天的标签
            actions = pd.read_csv(train_path)
            del_index = actions[ (actions['clickTime'] >= end_date) & (actions['label'] <= 1e-2) ]
            actions = actions.iloc[np.setdiff1d(actions.index,del_index),:].reset_index(drop=True)
        else:  
            actions = pd.read_csv(test_path)
        ad_feat_df = ad_feat_df[['creativeID','appID']]
        actions = pd.merge(actions,ad_feat_df,how='left',on='creativeID')
        actions.to_csv(dump_path,index=False, index_label=False)
    return actions

def get_time_bucket(actions_df):
    # time_bucket = [ [click_times,sum_labels], [],...] 
    time_bucket = np.zeros((date_max,3));
    print "bucket will loop %d times."%(len(actions_df.index));
    values = actions_df[['clickTime','label']].values
    for i in range(len(actions_df.index)):
        time_seed = int(values[i][0])
        label_val = 1.0*(values[i][1])
        time_bucket[time_seed,1] += 1.0
        time_bucket[time_seed,2] += label_val
    for i in range(1,len(time_bucket)):
        time_bucket[i,0] = i;
        time_bucket[i,1] += time_bucket[i-1,1];
        time_bucket[i,2] += time_bucket[i-1,2];
    print "max value in bucket is %f"%(np.max(time_bucket))
    return time_bucket;

# 滑动窗口获得以clickTime为切分之前的点击率滑动窗口
def slide_window( main_columns, key_name, train_step ):
    if train_step: dump_path = './cache/slide_train_%s.csv'%(key_name)
    else:          dump_path = './cache/slide_test_%s.csv'%(key_name)
    if os.path.exists(dump_path): actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(train_step)[ main_columns + ['clickTime','label'] ]
        print "slide_window start groupby"; a = time.time()
        actions = actions.groupby(main_columns + ['clickTime'], as_index=False).sum().reset_index(drop=True)
        print "slide_window end groupby. use time %dm."%((time.time()-a)/60)
        if train_step: time_bucket_array = get_time_bucket(actions); #时间桶装改时间内的click行为
        else:          time_bucket_array = get_time_bucket(get_actions(True));
        actions = actions[ main_columns+['clickTime'] ]
        for i in range(len(fibo_seq)): # fibo滑动窗口
            new_time_bucket_array = time_bucket_array.copy();
            for j in range(fibo_seq[i],len(time_bucket_array)):
                new_time_bucket_array[j][1] -= time_bucket_array[ j - fibo_seq[i] ][1]
                new_time_bucket_array[j][2] -= time_bucket_array[ j - fibo_seq[i] ][2]
            max_click_num = np.max(new_time_bucket_array[:,1])
            min_click_num = np.min(new_time_bucket_array[:,1])
            max_buy_num = np.max(new_time_bucket_array[:,2])
            min_buy_num = np.min(new_time_bucket_array[:,2])
            for j in range(fibo_seq[i],len(time_bucket_array)):
                new_time_bucket_array[j][1] = (new_time_bucket_array[j][1] - min_click_num)/(max_click_num - min_click_num)
                new_time_bucket_array[j][2] = (new_time_bucket_array[j][2] - min_buy_num)/(max_buy_num - min_buy_num)
            time_bucket_df = pd.DataFrame(new_time_bucket_array,columns=['clickTime','ratio_click_%s_%d'%(key_name,i),'ratio_buy_%s_%d'%(key_name,i) ])
            actions = pd.merge(actions, time_bucket_df, how='left', on='clickTime')
            print "end fibo %s %d"%(key_name,i)
        actions.to_csv(dump_path,index=False, index_label=False)
    actions = actions.replace(np.nan,0.0)
    return actions

def add_feature():
    main_columns =['label','creativeID','userID','positionID','appID','clickTime']
    ####################### 添加positionID安装率特征+滑动窗口 ######################
    sample_df = get_actions(train_step=True)[ main_columns + ['clickTime','label'] ]
    new_feat = slide_window( ['positionID'], 'position_slide')
    new_feat = pd.merge(sample_df, new_feat, how='left',on=['positionID','clickTime']); 
    new_feat = new_feat.replace( np.nan, 0.0 )
    new_feat = new_feat[ np.setdiff1d(new_feat.columns,main_columns) ]
    end_feat_idx = libsvm_add(libsvm_dump_path,new_feat,end_feat_idx);
    print "num(user-sku):%d,end_feat_idx:%d"%(len(sample_df.values),end_feat_idx)

a = slide_window(['appID'],'app_slide',train_step= False);a = None;
a = slide_window(['appID'],'app_slide',train_step= True)
a = slide_window(['userID'],'user_slide',train_step= False);a = None;
a = slide_window(['userID'],'user_slide',train_step= True)
a = slide_window(['positionID'],'position_slide',train_step= False);a = None;
a = slide_window(['positionID'],'position_slide',train_step= True)
a = slide_window(['creativeID'],'creative_slide',train_step= False);a = None;
a = slide_window(['creativeID'],'creative_slide',train_step= True)

#dump_svmlight_file(a,b,'./cache/a.txt', zero_based=True,multilabel=False)















