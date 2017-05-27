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

a = pd.DataFrame(np.random.randint(3,size=(6,2)),columns=["one",'two'])
b = pd.Series(np.random.randint(3,size=(6)));
print a
print b
dump_svmlight_file(a,b,'../cache/b.txt', zero_based=True,multilabel=False)