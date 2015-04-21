# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:03:23 2015

@author: nightfox
"""

from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.svm.libsvm import predict_proba
import numpy as np
import pandas as pd

train = pd.read_csv('data/train.csv', encoding='utf-8')

#Label Encoding

target = train['target'].tolist()
le = LabelEncoder()
label = le.fit_transform(target)
train = train.drop('target', axis=1)
train['target'] = pd.Series(label)

#Train/Test data split

y = np.array(train['target'])
colNames = train.columns.tolist()
colNames.remove('target')
X = np.array(train[colNames])
X, y = shuffle(X,y)
trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.7, train_size = 0.3, random_state = 42)

#Training & Validation

'''sk = SelectKBest(f_regression, k=60)
sk.fit(trainX,trainy)
trainX = sk.transform(trainX)
testX = sk.transform(testX)'''

'''rfe = RFECV(LinearSVC(), step = 1)
rfe.fit(trainX, trainy)
trainX = rfe.transform(trainX)
testX = rfe.transform(testX) 

tuning = {'n_neighbors': [50, 60, 70, 80, 90, 100], 'leaf_size': [10, 20]}
gscv = GridSearchCV(KNeighborsClassifier(), param_grid = tuning, scoring = 'log_loss')
gscv.fit(trainX, trainy)

print("best parameters: ", gscv.best_estimator_)
print("best score: ", gscv.best_score_)

knn = KNeighborsClassifier(n_neighbors=70, leaf_size=20, p=2)
knn.fit(trainX, trainy)
pred1 = np.array(knn.predict(testX))
proba1 = [i for index, i in enumerate(knn.predict_proba(testX))]
print (pd.Series(proba1))'''


#Testing

test = pd.read_csv('data/test.csv', encoding='utf-8')

'''sk = SelectKBest(f_regression, k=60)
sk.fit(X,y)
X = sk.transform(X)
test = sk.transform(test)'''

rfe = RFECV(LinearSVC(), step = 1)
rfe.fit(X, y)
X = rfe.transform(X)
test = rfe.transform(test)  

knn = KNeighborsClassifier(n_neighbors=90, leaf_size=10, p=2)
knn.fit(X, y)
pred = np.array(knn.predict(test))
proba = [i for index, i in enumerate(knn.predict_proba(test))]
print (pd.Series(proba))
probadf = pd.DataFrame(proba, columns=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
nrow = probadf.shape[0]+1
ids = pd.Series(np.arange(nrow))
ids = ids.drop(0)
result = pd.concat([ids, probadf], axis=1)
result.to_csv('Submission.csv', header=True, index=None)

import scipy as sp
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll





