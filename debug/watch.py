#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import copy

ad_path = '../data/ad.csv'
app_categories_path = '../data/app_categories.csv'
position_path = '../data/position.csv'
test_path = '../data/test.csv'
train_path = '../data/train.csv'
user_path = '../data/user.csv'
user_app_actions_path = '../data/user_app_actions.csv'
user_installedapps_path = '../data/user_installedapps.csv'

def watch_distribute(user_path,main_column):
	df = pd.read_csv( user_path )
	df['count'] = 1.0
	df = df[main_column+["count"]]
	df = df.groupby(main_column,as_index=False).sum().reset_index(drop=True)
	df = df.sort_values('count',ascending=False).reset_index(drop=True)
	print df

#x = pd.read_csv( user_path )
#print x.describe()
watch_distribute(user_path,['hometown'])

