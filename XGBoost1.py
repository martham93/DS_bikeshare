import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('sheet_checkouts_by_station_map_data.csv')

#Get all trips longer than 30 mins and the type is not equal to annual
df2 = pd.DataFrame((df['Trip Duration (Minutes)'] > 30) & (df['Entry Pass Type'] !='Annual')\
|(df['Trip Duration (Minutes)'] > 30) & (df['Entry Pass Type'] != 'Student')\
|(df['Trip Duration (Minutes)'] > 60) & (df['Entry Pass Type'] == 'Annual')\
|(df['Trip Duration (Minutes)'] > 60) & (df['Entry Pass Type'] == 'Student'))

df['Overcharge'] = df2
df['Overcharge'].head(5)

# Convert more variables to numeric
label_encoder = preprocessing.LabelEncoder()
encoded_day = label_encoder.fit_transform(df["Checkout Day of Week"])
encoded_cstation = label_encoder.fit_transform(df["Checkout Station (Station Information)"])
encoded_rstation = label_encoder.fit_transform(df["Return Station (Station Information)"])

train_features = pd.DataFrame([encoded_cstation,
                              encoded_rstation,
                              encoded_day]).T
#Do a train/test split
X_train, X_test, y_train, y_test = train_test_split(train_features, df['Overcharge'], test_size=0.4, random_state=0)

seed = 7
test_size = 0.33

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

predcs = model.predict(X_test)
pd.crosstab(predcs,y_test)

predcs = model.predict(X_test)
accuracy = accuracy_score(y_test, predcs)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
