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

from keras.layers.normalization import BatchNormalization

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/marthamorrissey/Desktop/DS_practice/df_elev_dist.csv')

df.columns


#Get all trips longer than 30 mins and the type is not equal to annual
df2 = pd.DataFrame((df['Trip Duration (Minutes)'] > 30) & (df['Entry Pass Type'] !='Annual')\
|(df['Trip Duration (Minutes)'] > 30) & (df['Entry Pass Type'] != 'Student')\
|(df['Trip Duration (Minutes)'] > 60) & (df['Entry Pass Type'] == 'Annual')\
|(df['Trip Duration (Minutes)'] > 60) & (df['Entry Pass Type'] == 'Student'))

df['Overcharge'] = df2
df['Overcharge'].head(5)
df.columns





df_majority = df[df.Overcharge==0]
df_minority = df[df.Overcharge==1]
len(df_majority)
len(df_minority)

df_minority_upsampled = utils.resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=300000,    # to match majority class
                                 random_state=123) # reproducible results


df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.Overcharge.value_counts()


# Convert more variables to numeric
label_encoder = preprocessing.LabelEncoder()
encoded_day = label_encoder.fit_transform(df_upsampled["Checkout Day of Week"])
encoded_cstation = label_encoder.fit_transform(df_upsampled["Checkout Station"])
encoded_rstation = label_encoder.fit_transform(df_upsampled["Return Station (Station Information)"])
comhrs = df_upsampled['pm_commute_hrs']
acomhrs = df_upsampled['am_commute_hrs']
chour = df_upsampled['Checkout hour']
wknd = df_upsampled['weekend']
gains = df['net_gain']
dist = df['o_d_dist']

train_features = pd.DataFrame([encoded_cstation,
                              encoded_rstation,
                              encoded_day,
                              comhrs,
                              chour,
                              wknd,
                              gains,
                              dist]).T


#Do a train/test split, updated with upsampled data

bcycle_model2 = df[['encoded_cstation', 'encoded_rstation', 'am_commute_hrs', 'pm_commute_hrs', 'weekend', 'net_gain', 'o_d_dist']]
bcycle_model2.columns

X_train2, X_test2, y_train2, y_test2 = train_test_split(bcycle_model2, df['ot'], test_size=0.4, random_state=0)


######
model = Sequential()
model.add(Dense(16, input_dim=7, activation='relu')) #input dim X_trai shape[1]
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

m = model.fit(np.array(X_train2), np.array(y_train2), epochs=10, batch_size=100, verbose = 1)

X_train2.shape
y_train2.shape


scores = model.evaluate(np.array(X_train2), np.array(y_train2))


print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)) #86.16% without batchnorm, 80 with

scores_test = model.evaluate(np.array(X_test2), np.array(y_test2))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores_test[1]*100)) #80.15% without batchnorm

print(scores_test[0])

m2 = model.fit(np.array(X_train2), np.array(y_train2), batch_size= 100,epochs= 10,verbose=1,validation_data=(np.array(X_test2), np.array(y_test2)))


m2.history

accuracy = m2.history['acc']
val_accuracy = m2.history['val_acc']
loss = m2.history['loss']
val_loss = m2.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

m.history





import pydot_ng as pydot
import graphviz
from keras.utils import plot_model
plot_model(model, to_file='/Users/marthamorrissey/Desktop/model.png')


model.summary()


from sklearn.metrics import roc_auc_score

y_pred = model.predict_proba(np.array(X_test2))
roc_auc_score(np.array(y_test2), y_pred)
#.67 RUC score

model = Sequential()
model.add(Dense(16, input_dim=7, activation='relu')) #input dim X_trai shape[1]
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[''])

model.fit(np.array(X_train2), np.array(y_train2), epochs=10, batch_size=100)
