#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import os,time
#import tensorflow as tf
#import lightgbm as lgb
fibo_seq = [0,1,2,4,7,12]
end_date = 270000
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll
"""
a = pd.DataFrame(np.random.randint(3,size=(6,1)),columns=["one"])
a = a.groupby('one',as_index=False).first().reset_index(drop=True)
b = pd.DataFrame(np.random.randint(3,size=(6,3)),columns=["one","two","three"])
print a
print b
a = pd.merge(a,b, how='left',on=['one'])
print a
print a[a.one>a.two]
a = [1,2,3,4,5]
b = [1,2,3,4,5]
print a + b
x = 8400
#print np.random.randint(2,size=(5,))
act = list(np.ones(x)) +  list(np.zeros(338489-x) )
#act = list(np.random.randint(2,size=(338489,)))
pre = list(np.zeros(338489))
print logloss(act, pre)
"""
a = np.zeros((5,2))
b = np.ones((5,2))
a[4][1] = 10
print a[4]-b[4]
print np.tanh(5*2.366375e-02)
print np.tanh(5*4.819277e-02)
print np.tanh(0)


import shutil
#shutil.copy('./cache/a.txt','./cache/b.txt')
print np.log2(2805118)
feature_columns = np.array(['xxxxx','yyyyyyyy','44444444444444'])
pd.DataFrame(np.array(feature_columns),columns=['feature_columns']).to_csv('../cache/a.csv',index=False, index_label=False)

print int(500.0/100)
print "%.10d"%(200)

samples_df = pd.DataFrame(np.random.randint(3,size=(6,2)),columns=["one",'two'])
idx = np.random.permutation(samples_df.index)
print samples_df
samples_df = samples_df.iloc[idx].reset_index(drop=True)
print samples_df







