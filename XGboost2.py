import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
from sklearn.utils import resample
%matplotlib inline
os.chdir('/Users/danielfeeney/Documents/DataScience/Bicycle')
#df = pd.read_csv('bcycle_modified_full.csv')
df = pd.read_csv('df_elev_dist.csv')
#Gridsearch, sklearn api xgboost, drop in gridsearch cv class sets the score by parameter.


#Get all trips longer than 30 mins and the type is not equal to annual
df2 = pd.DataFrame((df['Trip Duration (Minutes)'] > 30) & (df['Entry Pass Type'] !='Annual')\
|(df['Trip Duration (Minutes)'] > 30) & (df['Entry Pass Type'] != 'Student')\
|(df['Trip Duration (Minutes)'] > 60) & (df['Entry Pass Type'] == 'Annual')\
|(df['Trip Duration (Minutes)'] > 60) & (df['Entry Pass Type'] == 'Student'))

df['Overcharge'] = df2
df['Overcharge'].head(5)
df.columns



# Separate majority and minority classes
df_majority = df[df.Overcharge==0]
df_minority = df[df.Overcharge==1]
len(df_majority)
len(df_minority)

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=300000,    # to match majority class
                                 random_state=123) # reproducible results


# Combine majority class with upsampled minority class
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
#Do a train/test split
X_train, X_test, y_train, y_test = train_test_split(train_features, df_upsampled['Overcharge'], test_size=0.4, random_state=0)

seed = 7
test_size = 0.33


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
model
predcs = model.predict(X_test)
pd.crosstab(predcs,y_test)

from xgboost import plot_importance
from matplotlib import pyplot
print(model.feature_importances_)
plot_importance(model)


from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, predcs)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
