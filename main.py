# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:33:57 2019

@author: VE00YM015
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
data_train = pd.read_csv("train_LZdllcl.csv")
data_test = pd.read_csv("test_2umaH9m.csv")

data_merge = pd.concat([data_train[data_train.columns[0:13]],data_test], sort = False)
data_merge['awards_won?'].value_counts().plot(kind='bar')
#data_merge['department'].value_counts().plot(kind='bar')
#data_merge['region'].value_counts().plot(kind='bar')
#data_merge['gender'].value_counts().plot(kind='bar')
#data_merge['recruitment_channel'].value_counts().plot(kind='bar')

#data_merge['education'].isnull()

#data_merge[data_merge['education'] == 'Bachelor\'s']['department'].value_counts().plot(kind = 'bar')

data_merge['education'] = data_merge['education'].replace(np.NaN,'Bachelor\'s')
#data_merge['department'].isna().sum()
#data_merge['region'].isna().sum()
#data_merge.isna().sum()
#data_merge.dtypes
#df = data_merge[data_merge['previous_year_rating'].isna()]
#data_merge['previous_year_rating'].mean()

data_merge['previous_year_rating'] = data_merge['previous_year_rating'].replace(np.NaN,data_merge['previous_year_rating'].mean())

from sklearn.preprocessing import StandardScaler
normalizer = StandardScaler()
dataset_norm1 = normalizer.fit_transform(data_merge[data_merge.columns[6:10]])
dataset_norm2 = normalizer.fit_transform(data_merge[data_merge.columns[12:13]])
data_merge['department'] = data_merge['department'].astype('category')
data_merge['region'] = data_merge['region'].astype('category')
data_merge['education'] = data_merge['education'].astype('category')
data_merge['gender'] = data_merge['gender'].astype('category')
data_merge['recruitment_channel'] = data_merge['recruitment_channel'].astype('category')
data_merge['KPIs_met >80%'] = data_merge['KPIs_met >80%'].astype('category')
data_merge['awards_won?'] = data_merge['awards_won?'].astype('category')
dataset_norm1 = pd.DataFrame(dataset_norm1)
dataset_norm2 = pd.DataFrame(dataset_norm2)

dataset_norm = np.concatenate((dataset_norm1,dataset_norm2,data_merge[['employee_id','department','region','education','gender','recruitment_channel','KPIs_met >80%','awards_won?']]),axis = 1)
dataset_norm = pd.DataFrame(dataset_norm)

train = data_merge.iloc[:54808,:].values
test = dataset_norm.iloc[54808:,:].values
y = data_train['is_promoted']
train = pd.DataFrame(train)
#train = train.assign(is_promoted = data_train['is_promoted']) 
#train = train.iloc[:,:].values

#Model Fitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
RF.fit(train, y)
RF.predict(x_test)