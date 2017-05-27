#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import os,copy,math,time,sys
import scipy as sp
import numpy as np

from my_import import pd_decimal2binary
from my_import import libsvm_add
sys.path.append("../featureProject")

ad_path = './data/ad.csv'
app_categories_path = './data/app_categories.csv'
position_path = './data/position.csv'
test_path = './data/test.csv'
train_path = './data/train.csv'
user_path = './data/user.csv'
user_app_actions_path = './data/user_app_actions.csv'
user_installedapps_path = './data/user_installedapps.csv'

max_age = 80
max_gender = 2
max_education = 7
max_marriageStatus = 3
max_haveBaby = 6
max_connectionType = 4
max_telecomsOperator = 3
max_sitesetID = 2
max_positionType = 6
max_appPlatform = 2
max_appCategory_level1 = 9
max_appCategory_level2 = 99
max_creativeID = 6582
max_adID = 3616
max_camgainID = 720
max_advertiserID = 91
max_appID=433269
max_appCategory = 503
max_positionID=7654
max_userID = 2805118
max_hometown = 3401
max_residence = 3401
max_installTime = 302359
max_clickTime = 302359
max_conversionTime = 302359
end_date = 270000

date_max = 312400
fibo_seq = [100,200,600,1200,1*2400,2*2400,4*2400,7*2400,12*2400,20*2400,25*2400] # 点击时间滑动窗口

# 读取整个用户基本的特征
def get_basic_user_feat():
    dump_path = './cache/basic_user_feature.csv'
    if os.path.exists(dump_path): feat_df = pd.read_csv(dump_path)
    else:
        feat_df = pd.read_csv(user_path, encoding='gbk')
        #age_df = pd.get_dummies(feat_df["age"], prefix="age")
        gender_df = pd.get_dummies(feat_df["gender"], prefix="gender")
        education_df =  pd.get_dummies(feat_df["education"], prefix="education")
        marriageStatus_df = pd.get_dummies(feat_df["marriageStatus"], prefix="marriageStatus")
        haveBaby_df = pd.get_dummies(feat_df["haveBaby"], prefix="haveBaby")
        hometown_df  = pd_decimal2binary( feat_df["hometown"], max_data = max_hometown, prefix="home_town" )
        residence_df = pd_decimal2binary( feat_df["residence"], max_data = max_residence, prefix="residence" )
        feat_df = pd.concat([feat_df[['userID','age']], gender_df, education_df, marriageStatus_df, haveBaby_df, hometown_df, residence_df], axis=1)
        feat_df = feat_df.groupby(["userID"], as_index=False).first().reset_index(drop=True)
        feat_df.to_csv( dump_path, index=False, index_label=False )
        print "User feature length is %d"%(len(feat_df.columns))
        for column in feat_df.columns: print column,
        print "\n--------------------------------------------------"
    return feat_df

# 读取整个广告的基本特征
def get_basic_ADcreative_feat():
    dump_path = './cache/basic_ADcreative_feature.csv'
    if os.path.exists(dump_path): feat_df = pd.read_csv(dump_path)
    else:
        feat_df = pd.read_csv(ad_path)
        adID_df = pd_decimal2binary( feat_df["adID"], max_data = max_adID, prefix="adID" )
        camgaignID_df = pd_decimal2binary( feat_df["camgaignID"], max_data = max_camgainID, prefix="camgaignID" )
        advertiserID_df = pd_decimal2binary( feat_df["advertiserID"], max_data = max_advertiserID, prefix="advertiserID" )
        appID_df = pd_decimal2binary( feat_df["appID"], max_data = max_appID, prefix="appID" ) 
        appPlatform_df = pd.get_dummies(feat_df["appPlatform"], prefix="appPlatform")
        feat_df = pd.concat([feat_df['creativeID'], adID_df, camgaignID_df, advertiserID_df, appID_df, appPlatform_df], axis=1)
        feat_df = feat_df.groupby('creativeID', as_index=False).first().reset_index(drop=True)
        feat_df.to_csv(dump_path,index=False,index_label=False)
        print "ADcreative feature length is %d"%(len(feat_df.columns))
        for column in feat_df.columns: print column,
        print "\n--------------------------------------------------"
    return feat_df

# 读取整个广告曝光位置的基本特征
def get_basic_position_feat():
    dump_path = './cache/basic_position_feature.csv'
    if os.path.exists(dump_path): feat_df = pd.read_csv(dump_path)
    else:
        feat_df = pd.read_csv(position_path)
        sitesetID_df = pd.get_dummies(feat_df["sitesetID"], prefix="sitesetID")
        positionType_df = pd.get_dummies(feat_df["positionType"], prefix="positionType")
        feat_df = pd.concat([feat_df['positionID'], sitesetID_df, positionType_df], axis=1)
        feat_df = feat_df.groupby('positionID', as_index=False).first().reset_index(drop=True)
        feat_df.to_csv(dump_path,index=False,index_label=False)
        print "position feature length is %d"%(len(feat_df.columns))
        for column in feat_df.columns: print column,
        print "\n--------------------------------------------------"
    return feat_df

# 读取APP类别基本特征
def get_basic_APPcategories_feat():
    dump_path = './cache/basic_APPcategories_feature.csv'
    if os.path.exists(dump_path): feat_df = pd.read_csv(dump_path)
    else:
        feat_df = pd.read_csv(app_categories_path)
        fun1 = lambda x: int(1.0*float(x)/100);
        fun2 = lambda x: int(x) - int(1.0*float(x)/100)*100;
        feat_df['appCategory_lv1'] = feat_df['appCategory'].map(fun1)
        feat_df['appCategory_lv2'] = feat_df['appCategory'].map(fun2)
        appCategory1_df = pd.get_dummies(feat_df['appCategory_lv1'], prefix='appCategory_lv1')
        appCategory2_df = pd.get_dummies(feat_df['appCategory_lv2'], prefix='appCategory_lv2')
        feat_df = pd.concat([feat_df['appID'], appCategory1_df, appCategory2_df ], axis=1)
        feat_df = feat_df.groupby('appID', as_index=False).first().reset_index(drop=True)
        feat_df.to_csv(dump_path,index=False,index_label=False)
        print "APPcategories feature length is %d"%(len(feat_df.columns))
        for column in feat_df.columns: print column,
        print "\n--------------------------------------------------"
    return feat_df

# 获取用户点击行为
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

# 获得时间段内的操作向量
def get_action_feat(train_step):
    # 返回值 columns = ['label',creativeID','userID','positionID','appID','clickTime']
    train_need_del_columns = ['connectionType','telecomsOperator','conversionTime']
    test_need_del_columns = ['connectionType','telecomsOperator']
    if train_step:  dump_path = './cache/train_action_feat.csv'
    else:           dump_path = './cache/test_action_feat.csv'
    if os.path.exists(dump_path): actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(train_step)
        connectionType_df = pd.get_dummies(actions["connectionType"], prefix="connectionType")
        telecomsOperator_df = pd.get_dummies(actions["telecomsOperator"], prefix="telecomsOperator")
        actions = pd.concat([actions,connectionType_df,telecomsOperator_df],axis=1)
        actions = pd.merge(actions, get_basic_user_feat(), how='left', on=['userID'])
        actions = pd.merge(actions, get_basic_ADcreative_feat(), how='left', on=['creativeID'])
        actions = pd.merge(actions, get_basic_position_feat(), how='left', on=['positionID'])
        actions = pd.merge(actions, get_basic_APPcategories_feat(), how='left', on=['appID'])
        if train_step:  actions = actions[ np.setdiff1d(actions.columns,train_need_del_columns) ]
        else:           actions = actions[ np.setdiff1d(actions.columns,test_need_del_columns) ]
        actions.to_csv(dump_path,index=False,index_label=False)
    return actions

# 获得(用户是否安装过该次点击的app)的特征
def get_install_feat(actions_df, train_step):
    if train_step:  dump_path = './cache/train_install.csv'
    else:           dump_path = './cache/test_install.csv'
    if os.path.exists(dump_path): actions_df = pd.read_csv(dump_path)
    else:
        installed_df = pd.read_csv(user_installedapps_path)
        installed_df['installed'] = 1.0
        actions_df = pd.merge(actions_df,installed_df,how='left',on=['userID','appID']); installed_df = None
        appAction_df = pd.read_csv(user_app_actions_path)
        appAction_df =  appAction_df.sort_values('installTime',ascending=True).reset_index(drop=True)
        appAction_df = appAction_df.groupby(['userID','appID'],as_index=False).first().reset_index(drop=True)
        actions_df = pd.merge(actions_df,appAction_df,how='left',on=['userID','appID']); appAction_df = None
        actions_df['installed'] = actions_df['installed'].replace(np.nan, 0.0)
        actions_df['installed'] = (actions_df['installed']==1.0) | (actions_df['clickTime'] >= actions_df['installTime'] )
        fun = lambda x: 1.0 if x else 0.0;
        actions_df['installed'] = actions_df['installed'].map(fun)
        actions_df = actions_df[['userID','appID','installed']]
        actions_df.to_csv(dump_path,index=False, index_label=False)
    return actions_df

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

# 滑动窗口获得以clickTime为切分点之前的user/sku点击率滑动窗口
def slide_window( main_columns, key_name, train_step ):
    dump_path = './cache/slide_%s.csv'%(key_name)
    if os.path.exists(dump_path): actions = pd.read_csv(dump_path)
    else:
        actions = get_actions(train_step=True)[ main_columns + ['clickTime','label'] ]
        print "slide_window start groupby"; a = time.time()
        actions = actions.groupby(main_columns + ['clickTime'], as_index=False).sum().reset_index(drop=True)
        print "slide_window end groupby. use time %dm."%((time.time()-a)/60)
        time_bucket_array = get_time_bucket(actions); #时间桶装改时间内的click行为
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

def add_feat(): pass

def make_train_set( train_step, feature_columns_sump_path = './logs/feature_name.csv' ):
    main_train_columns =['label','creativeID','userID','positionID','appID','clickTime']
    main_test_columns = ['label','creativeID','userID','positionID','appID','clickTime','instanceID']
    if train_step:
        sample_df_path = './cache/train_sample.csv'
        data_df_path = './cache/train_data_set.csv'
        main_columns = main_train_columns
    else: 
        sample_df_path = './cache/test_sample.csv'
        data_df_path = './cache/test_data_set.csv'
        main_columns = main_test_columns
    if os.path.exists(sample_df_path) and os.path.exists(data_df_path): 
        return pd.read_csv(sample_df_path), pd.read_csv(data_df_path)
    actions = get_action_feat(train_step)
    sample_df = actions[main_columns].copy() #样本dataFrame
    end_feat_idx = len(actions.columns) - len(main_columns)
    print "num(samples):%d,end_feat_idx:%d"%(len(actions.index),end_feat_idx)
    ####################### 添加app是否安装特征 ######################
    install_df = get_install_feat( actions[['userID','appID','clickTime']], train_step)
    actions =  pd.concat([ actions, install_df['installed'] ],axis = 1); install_df = []
    actions['installed'] = actions['installed'].replace(np.nan, 0.0)
    end_feat_idx = len(actions.columns) - len(main_columns)
    print "num(samples):%d,end_feat_idx:%d"%(len(actions.index),end_feat_idx)
    ####################### 添加user安装率特征+滑动窗口 ######################
    slide_df = slide_window( ['userID'], 'user_slide', train_step)
    actions =  pd.merge(actions, slide_df, how='left',on=['userID','clickTime']); slide_df = []
    actions = actions.replace(np.nan, 0.0)
    end_feat_idx = len(actions.columns) - len(main_columns)
    print "num(samples):%d,end_feat_idx:%d"%(len(actions.index),end_feat_idx)
    """
    ####################### 添加app安装率特征+滑动窗口 ######################
    slide_df = slide_window( ['appID'], 'app_slide', train_step)
    actions =  pd.merge(actions, slide_df, how='left',on=['appID','clickTime']); slide_df = []
    actions = actions.replace(np.nan, 0.0)
    end_feat_idx = len(actions.columns) - len(main_columns)
    print "num(samples):%d,end_feat_idx:%d"%(len(actions.index),end_feat_idx)
    """
    #############################   写入备份 #############################
    end_feat_idx = len( np.setdiff1d(actions.columns,main_columns) )
    actions = actions[ np.setdiff1d(actions.columns,main_columns) ]
    #actions = actions.replace(np.nan, 0.0)
    print "all sample is %d, positive samples is %d."%( len(sample_df.index), np.sum(sample_df['label']) )
    print "num(samples):%d,end_feat_idx:%d"%(len(sample_df.index),end_feat_idx)
    # if if_over_sample: sample_df, libsvm_dump_path = over_sample(sample_df, libsvm_dump_path)
    sample_df.to_csv(sample_df_path,index=False,index_label=False)
    pd.DataFrame(actions.columns,columns=["feature_name"]).to_csv(feature_columns_sump_path,index=False,index_label=False)
    actions.to_csv(data_df_path,index=False,index_label=False)
    return sample_df, actions

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
    print "len(pre=1)=%d,"%( len(sub_df.index) - len(sub_df[sub_df['prob']==0].values) ),
    print "logloss=%f"%logloss(sub_df['label'],sub_df['prob'])

def change_feature():
    print "Start change feature."
    print "Start remove files."
    sample_df_path = './cache/train_sample.csv'
    if os.path.exists(sample_df_path): os.remove(sample_df_path) 
    data_df_path = './cache/train_data_set.csv' 
    if os.path.exists(data_df_path): os.remove(data_df_path)
    sample_df_path = './cache/test_sample.csv'
    if os.path.exists(sample_df_path): os.remove(sample_df_path)
    data_df_path = './cache/test_data_set.csv'
    if os.path.exists(data_df_path): os.remove(data_df_path)
    print "End remove files. Start make_train_set()."
    make_train_set(train_step=True)
    make_train_set(train_step=False)

if __name__ == '__main__':
    change_feature()
    print "import feature.py"
    

