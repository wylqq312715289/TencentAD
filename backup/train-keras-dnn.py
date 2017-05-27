#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import copy,os,sys
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import dump_svmlight_file
from svmutil import svm_read_problem
np.random.seed( 123 )

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding, LSTM
from keras.utils import np_utils
from keras.datasets import mnist
from keras import metrics
from keras.models import load_model

from tensorflow.contrib import learn
import tensorflow as tf
import keras.backend as K

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from featureProject.features import make_train_set
from featureProject.my_import import split_data
from featureProject.my_import import one_hot
from featureProject.my_import import re_onehot

model_path = "./model/my_keras.model" #模型保存的地址

def logloss(act, pred):
    # act和pred都是list类型
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def TencentReport( sub_df ):
    print "logloss=%f"%logloss(sub_df['label'],sub_df['prob'])
    # sub_df = set0state(sub_df)
    print "len(sub_df)=%d,"%( len(sub_df.index) ),
    print "len(real=1)=%d,"%( np.sum(sub_df['label'].values) ),
    print "len(pre=1)=%d,"%( len(sub_df.index) - len(sub_df[sub_df['prob']==0].values) ),
    print "logloss=%f"%logloss(sub_df['label'],sub_df['prob'])

def model_dnn( input_shape ):
	model = Sequential()
	model.add( Dense(64, activation='sigmoid', input_shape=(input_shape[1],)) )
	model.add( Dropout(0.25) )
	model.add( Dense(8, activation='sigmoid')  )
	model.add( Dropout(0.25) )
	model.add( Dense(1, activation='sigmoid') )
	return model

if __name__ == '__main__':
	#################### 训练数据 #####################
	train_samples_df, train_data_df = make_train_set(train_step=True)
	train_samples_df,train_data_df,test_samples_df,test_data_df = split_data(train_samples_df, train_data_df, 0.2)
	os.remove(model_path)
	if os.path.exists(model_path): model = load_model(model_path)
	else:
		#################### 训练数据 #####################
		print "train max value: %f"%( np.max(train_data_df.values) )
		print "Train data scale %d*%d"%(len(train_data_df.values),len(train_data_df.columns))
		target = train_samples_df['label'].values
		model = model_dnn( input_shape = (len(train_data_df.index),len(train_data_df.columns)) )
		model.compile( loss='mean_squared_error', optimizer='adam', metrics=['accuracy'] )
		model.fit( train_data_df.values, target, batch_size=512, epochs=10, verbose=1, validation_split = 0.15) 
		model.save(model_path)
	############################## 查看正确率 #############################################
	y = model.predict( test_data_df.values )
	test_samples_df['prob'] = y
	TencentReport( test_samples_df[['label','prob']] )
	############################ 生成提交文件 ###################################
	sub_samples_df, sub_data_df = make_train_set(train_step=False)
	y = model.predict( sub_data_df.values )
	print type(y)
	sub_samples_df['prob'] = y
	# sub_samples_df = set0state(sub_samples_df)
	pre_df = sub_samples_df[['instanceID','prob']].sort_values('instanceID',ascending=False).groupby('instanceID', as_index=False).first()
	pre_df.to_csv('./sub/submission.csv', index=False, index_label=False)


