# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:07:31 2015

@author: nightfox
"""
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model 
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import svm
from sklearn.metrics import mean_squared_error 

train = pd.read_csv('data/train.csv', encoding='utf-8')
test = pd.read_csv('data/test.csv', encoding = 'utf-8')

#pre-processing

#transforming date

open_date = train['Open Date'].tolist()
today = datetime.strptime(datetime.strftime(datetime.now(), '%m/%d/%Y'), '%m/%d/%Y')
delta = [(today - datetime.strptime(i, '%m/%d/%Y')).days for i in open_date]
train = train.drop('Open Date', axis=1)
train['Open Date'] = pd.Series(delta)

open_date1 = test["Open Date"].tolist()
today1 = datetime.strptime(datetime.strftime(datetime.now(), '%m/%d/%Y'), '%m/%d/%Y')
delta1 = [(today - datetime.strptime(i, '%m/%d/%Y')).days for i in open_date1]
test = test.drop('Open Date', axis=1)
test['Open Date'] = pd.Series(delta1)

#Train & Test data label encoding City, City Group & Type

train_cat_dict={'City':train['City'].tolist(), 'City Group':train['City Group'].tolist(), 'Type':train['Type'].tolist()}
test_cat_dict={'City':test['City'].tolist(), 'City Group':test['City Group'].tolist(), 'Type':test['Type'].tolist()}
le=LabelEncoder()

for key in train_cat_dict:
    train_cat_dict[key] = le.fit_transform(train_cat_dict[key])
    train = train.drop(key, axis=1)
    train[key] = pd.Series(train_cat_dict[key])

for key in test_cat_dict:
    test_cat_dict[key] = le.fit_transform(test_cat_dict[key])
    test = test.drop(key, axis=1)
    test[key] = pd.Series(test_cat_dict[key])

#Splitting Data into Training and Validation

y = train['revenue'].tolist()
colNames = train.columns.tolist()
colNames.remove('revenue')
X = np.array(train[colNames]).tolist()
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.1, train_size = 0.9, random_state = 42)

#Feature selection

sk = SelectKBest(f_regression, k = 25)
sk.fit(trainX, trainy)
trainX = sk.transform(trainX)
testX = sk.transform(testX)

'''rfecv = RFECV(linear_model.LinearRegression(), step = 1, cv = 4)
rfecv.fit(trainX,trainy)
trainX = rfecv.transform(trainX)
testX = rfecv.transform(testX)
print ('Best parameters: ', rfecv.ranking_)
print ('Best score: ', rfecv.grid_scores_)'''

#Parameter tuning

'''tuning_para = {'n_estimators': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], 'max_depth':[3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5]}
gscv = GridSearchCV(RandomForestRegressor(), tuning_para, scoring = 'mean_squared_error', cv = 10)
gscv.fit(trainX, trainy)

print ('Best parameters: ', gscv.best_estimator_)
print ('Best score: ', gscv.best_score_)

rf = RandomForestRegressor(n_estimators=28,max_depth=9)
rf.fit(trainX,trainy)
pred = np.array(rf.predict(testX))
score = rf.score(testX, testy)
rmse = np.sqrt(mean_squared_error(testy, np.array(pred)))
print (rmse, score)'''

'''
#Linear Regression
lr = linear_model.LinearRegression()
lr.fit(trainX,trainy)
pred = np.array(lr.predict(testX))
score = lr.score(testX, testy)

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=28,max_depth=9)
rf.fit(x_train,y_train)
pred = np.array(rf.predict(x_test))
score = rf.score(x_test, y_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, np.array(pred)))
print (score)'''

#Classification / Prediction

y = np.array(train['revenue'])
X = np.array(train.drop('revenue', axis=1))

#Feature selection

sk = SelectKBest(f_regression, k = 35)
sk.fit(X, y)
X = sk.transform(X)
test = sk.transform(test)

'''
rfecv = RFECV(linear_model.LinearRegression(), step = 1, cv = 4)
rfecv.fit(X,y)
X = rfecv.transform(X)
test = rfecv.transform(test)

#Linear Regressor
lr = linear_model.LinearRegression()
lr.fit(X,y)
pred = np.array(lr.predict(test))'''

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=2000,max_depth=9)
rf.fit(X,y)
pred = np.array(rf.predict(test))

rmse = np.sqrt(mean_squared_error(y, np.array(pred[:137])))
print(rmse)

#Save file
dfpred = pd.Series(pred)
Id = np.arange(dfpred.shape[0])

result = pd.DataFrame({"Id":Id, "Prediction":dfpred})
result.to_csv('Sample_Submission2.csv')

