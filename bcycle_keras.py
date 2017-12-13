import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn import utils
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from scipy.stats.stats import pearsonr
import itertools
import logging

from keras.optimizers import SGD


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline





df = pd.read_csv('/Users/marthamorrissey/Desktop/DS_practice/bcycle_modified_full.csv')
bcycle_model = df[['encoded_cstation', 'encoded_rstation', 'am_commute_hrs', 'pm_commute_hrs', 'weekend']]
df_fixed = df[(df['Trip Duration (Minutes)'] > 3) & (df['Trip Duration (Minutes)'] < 300)]

#Up-sampling data (to correct for low number of trips in the data set that are overtime)
df_majority = df_fixed[df_fixed.ot==0]
df_minority = df_fixed[df_fixed.ot==1]

len(df_majority)
len(df_minority)

# Upsample minority class
df_minority_upsampled = utils.resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=300000,    # to match majority class
                                 random_state=123) # reproducible results


# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])


# Convert more variables to numeric
label_encoder = preprocessing.LabelEncoder()
encoded_day = label_encoder.fit_transform(df_upsampled["Checkout Day of Week"])
encoded_cstation = label_encoder.fit_transform(df_upsampled["Checkout Station (Station Information)"])
encoded_rstation = label_encoder.fit_transform(df_upsampled["Return Station (Station Information)"])
comhrs = df_upsampled['pm_commute_hrs']
acomhrs = df_upsampled['am_commute_hrs']
chour = df_upsampled['Checkout hour']
wknd = df_upsampled['weekend']

wknd

train_features = pd.DataFrame([encoded_cstation,
                              encoded_rstation,
                              encoded_day,
                              comhrs,
                              chour,
                              wknd]).T


#Do a train/test split, updated with upsampled data
X_train, X_test, y_train, y_test = train_test_split(train_features, df_upsampled['ot'], test_size=0.4, random_state=0)
X_train.shape
X_test.shape

y_train.shape
y_test.shape
model = Sequential()

bcycle = pd.read_csv('/Users/marthamorrissey/Desktop/DS_practice/bcycle_modified_full.csv')
#contains feature engineered data

bcycle_model = bcycle[['encoded_cstation', 'encoded_rstation', 'am_commute_hrs', 'pm_commute_hrs', 'weekend']]






#Do a train/test split
X_train2, X_test2, y_train2, y_test2 = train_test_split(bcycle_model, bcycle['ot'], test_size=0.4, random_state=0)

X_train2.shape
X_test2.shape
y_train2.shape
y_test2.shape




score = model.evaluate(x_test, y_test, batch_size = 32)


########
seed = 7
np.random.seed(seed)

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


model.fit(np.array(X_train), np.array(y_train))
score = model.evaluate(np.array(X_test), np.array(y_test), batch_size = 32)
score



#########
# create model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=100)

np.array(X_train).shape
np.array(y_train).shape

scores = model.evaluate(np.array(X_train), np.array(y_train))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

scores_test = model.evaluate(np.array(X_test), np.array(y_test))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100))

import pydot_ng as pydot
import graphviz
from keras.utils import plot_model
plot_model(model, to_file='/Users/marthamorrissey/Desktop/model.png')


model.summary()


from sklearn.metrics import roc_auc_score

 y_pred = model.predict_proba(np.array(X_test))
roc_auc_score(np.array(y_test), y_pred)
