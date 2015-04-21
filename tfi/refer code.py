import pandas as pd
import numpy as np
import pickle
import glob
import time
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# <codecell>

train = pd.read_csv("data/train.csv",header=0, encoding="utf-8")
# data preprocessing 

open_date = train["Open Date"].tolist()
diff_date = [(datetime.strptime(time.strftime("%m/%d/%Y"),'%m/%d/%Y') - datetime.strptime(i,'%m/%d/%Y')).days for i in open_date]
train = train.drop("Open Date",axis=1)

train["Diff Date"] = pd.Series(np.array(diff_date))
train = train.drop("Id",axis=1)
# <codecell>
# Label Encoding
cityGroup = train["City Group"].tolist()
train = train.drop("City Group", axis=1)
le = LabelEncoder()
cityGroupEncoded = le.fit_transform(cityGroup)
train["City Group"] = pd.Series(np.array(cityGroupEncoded))

city = train["City"].tolist()
train = train.drop("City", axis=1)
le = LabelEncoder()
cityEncoded = le.fit_transform(city)
train["City"] = pd.Series(np.array(cityEncoded))

cityType = train["Type"].tolist()
train = train.drop("Type", axis=1)
le = LabelEncoder()
cityTypeEncoded = le.fit_transform(cityType)
train["Type"] = pd.Series(np.array(cityTypeEncoded))
# <codecell>
# training and testing data split 
y = np.array(train["revenue"])
colNames = train.columns.tolist()
colNames.remove("revenue")
X = np.array(train[colNames])
X, y = shuffle(X, y)
trainX, testX, trainy, testy = train_test_split(X,y,test_size=0.1, random_state=42)
# <codecell>
# variable selection KBest (univariate)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
sk = SelectKBest(f_regression, k=20)
sk.fit(trainX, trainy)
trainX = sk.transform(trainX)
testX = sk.transform(testX)
# <codecell>
tuned_parameters = {"n_estimators": [10,20,30,40,50,60,70,80,90,100],"max_depth": [1,2,3,4,5]}
clf = GridSearchCV(RandomForestRegressor(),tuned_parameters, scoring="mean_squared_error", cv=10)
clf.fit(trainX, trainy)
print ("the best parameters are ", clf.best_estimator_)
print ("the best score is ", clf.best_score_)
# <codecell>
rf = RandomForestRegressor(n_estimators=30,max_depth=5)
rf.fit(trainX, trainy)
pred = rf.predict(testX)
score = rf.score(testX,testy)
print (score)
print (np.sqrt(mean_squared_error(testy, pred)))
# <codecell>
# import test 
test = pd.read_csv("data/test.csv",header=0)
# data preprocessing 

open_date = test["Open Date"].tolist()
diff_date = [(datetime.strptime(time.strftime("%m/%d/%Y"),'%m/%d/%Y') - datetime.strptime(i,'%m/%d/%Y')).days for i in open_date]
test = test.drop("Open Date",axis=1)

test["Diff Date"] = pd.Series(np.array(diff_date))
test = test.drop("Id",axis=1)

# Label Encoding
cityGroup = test["City Group"].tolist()
test = test.drop("City Group", axis=1)
le = LabelEncoder()
cityGroupEncoded = le.fit_transform(cityGroup)
test["City Group"] = pd.Series(np.array(cityGroupEncoded))

city = test["City"].tolist()
test = test.drop("City", axis=1)
le = LabelEncoder()
cityEncoded = le.fit_transform(city)
test["City"] = pd.Series(np.array(cityEncoded))

cityType = test["Type"].tolist()
test = test.drop("Type", axis=1)
le = LabelEncoder()
cityTypeEncoded = le.fit_transform(cityType)
test["Type"] = pd.Series(np.array(cityTypeEncoded))

# <codecell>

sk = SelectKBest(f_regression, k=20)
sk.fit(X,y)
X = sk.transform(X)
test = sk.transform(test)
rf = RandomForestRegressor(n_estimators=30,max_depth=5)
rf.fit(X,y)
pred = rf.predict(test)
dfPred = pd.Series(np.array(pred))
nrow = dfPred.shape[0]
rmse = np.sqrt(mean_squared_error(y, np.array(pred[:137])))
print(rmse)
Id = pd.Series(np.arange(nrow))
result = pd.DataFrame({"Id":Id, "Prediction":dfPred})
result.to_csv("reference.csv",header=True, index=None)
# <codecell>
# take a peek at the submission file format

# <codecell>
