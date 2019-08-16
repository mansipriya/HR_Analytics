# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:34:06 2019

@author: Mansi Priya
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
data_train = pd.read_csv("train_LZdllcl.csv")
data_test = pd.read_csv("test_2umaH9m.csv")

data_merge = pd.concat([data_train[data_train.columns[0:13]],data_test], sort = False)
#data_merge['awards_won?'].value_counts().plot(kind='bar')

data_merge['education'] = data_merge['education'].replace(np.NaN,'Bachelor\'s')
data_merge['previous_year_rating'] = data_merge['previous_year_rating'].replace(np.NaN,data_merge['previous_year_rating'].mean())

data_on_hot = pd.get_dummies(data_merge)

train = data_on_hot.iloc[:54808,:].values
test = data_on_hot.iloc[54808:,:].values
y = data_train['is_promoted']

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
n_estimators = [30,50,70,90,100]
max_depth = [25,45,65]
min_samples_split = [4,5,6]
min_samples_leaf = [1, 2] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 10, 
                      n_jobs = -1)
bestF = gridF.fit(train, y)
gridF.best_params_

forestOpt = RandomForestClassifier(random_state = 1, max_depth = 45,n_estimators = 30, min_samples_split = 5, min_samples_leaf = 1)                                 
modelOpt = forestOpt.fit(train, y)
y_pred = modelOpt.predict(test)
p = pd.DataFrame(y_pred)
p.to_csv("pred.csv")