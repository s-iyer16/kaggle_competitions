# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:07:31 2015

@author: nightfox
"""
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model 
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import svm
from sklearn import metrics

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

#Validation

y_train = np.array(train['revenue'])[:67]
x_train = np.array(train.drop('revenue', axis=1))[:67]
y_test = np.array(train['revenue'])[67:]
x_test = np.array(train.drop('revenue', axis=1))[67:]



'''#Linear Regression
lr = linear_model.LinearRegression(normalize=True)
lr.fit(x_train,y_train)
pred = np.array(lr.predict(x_test))
score = lr.score(x_test, y_test)'''

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=28,max_depth=9)
rf.fit(x_train,y_train)
pred = np.array(rf.predict(x_test))
score = rf.score(x_test, y_test)

rmse = np.sqrt(metrics.mean_squared_error(y_test, np.array(pred)))
print (score)

#Classification / Prediction

y = np.array(train['revenue'])
x = np.array(train.drop('revenue', axis=1))



#Linear Regressor
'''lr = linear_model.LinearRegression(normalize=True)
lr.fit(x,y)
pred = np.array(lr.predict(test))'''

#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=28,max_depth=9)
rf.fit(x,y)
pred = np.array(rf.predict(test))

rmse = np.sqrt(metrics.mean_squared_error(y, np.array(pred[:137])))
print(rmse)

#Save file
dfpred = pd.Series(pred)
Id = np.arange(dfpred.shape[0])

result = pd.DataFrame({"Id":Id, "Prediction":dfpred})
result.to_csv('pseudoSampleSubmission2.csv')
     

