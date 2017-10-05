import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

bcycle = pd.read_csv('/Users/marthamorrissey/Desktop/bcycle_modified_full.csv')
#contains feature engineered data

bcycle_model = bcycle[['encoded_cstation', 'encoded_rstation', 'am_commute_hrs', 'pm_commute_hrs', 'weekend']]


#Do a train/test split
X_train, X_test, y_train, y_test = train_test_split(bcycle_model, bcycle['ot'], test_size=0.4, random_state=0)

seed = 7
test_size = 0.33

model = XGBClassifier()
model.fit(X_train, y_train)

predcs = model.predict(X_test)
pd.crosstab(predcs,y_test)

predcs = model.predict(X_test)
accuracy = accuracy_score(y_test, predcs)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
