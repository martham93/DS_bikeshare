import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



bcycle = pd.read_csv('/Users/marthamorrissey/Desktop/sheet_checkouts_by_station_map_data.csv', encoding = 'utf-8')


bcycle.head()

bcycle['Entry Pass Type'].unique()


#Figure out over-time

#split big data frame into (2), 1 with the 60min time limit catagories and the other with the 30 min catagories then recombine
bcycle_60 = bcycle[(bcycle['Entry Pass Type'] == 'Annual') | (bcycle['Entry Pass Type']== 'Semester (150-day)')]

bcycle_60['Entry Pass Type'].unique()
bcycle_60.columns

bcycle_60['ot'] = np.where((bcycle_60['Trip Duration (Minutes)'] > 60), 1, 0)


bcycle_30 = bcycle[(bcycle['Entry Pass Type'] != 'Annual') & (bcycle['Entry Pass Type'] != 'Semester (150-day)')]
bcycle_30['Entry Pass Type'].unique()
bcycle_30['ot'] = np.where((bcycle_30['Trip Duration (Minutes)'] > 30), 1, 0)

bcycle_30.head()

frames = [bcycle_30, bcycle_60]

bcycle_new = pd.concat(frames)

bcycle_new.columns


bcycle_new.columns

bcycle_new['ot'].unique()

bcycle_new['weekend'] = np.where((bcycle_new['Checkout Day of Week'] == 'Sunday') | (bcycle_new['Checkout Day of Week'] == 'Saturday'), 1, 0)
bcycle_new['Return Time'][0]

bcycle_new['Return dt'] = pd.to_datetime(bcycle_new['Return Time'])



bcycle_new['Return hour'] = bcycle_new['Return dt'].dt.hour.astype(int)
bcycle_new['Return hour'].unique()
bcycle_new['Checkout dt'] = pd.to_datetime(bcycle_new['Checkout Time'])
bcycle_new['Checkout hour'] = bcycle_new['Checkout dt'].dt.hour.astype(int)
bcycle_new.head()
bcycle_new.to_csv('/Users/marthamorrissey/Desktop/bcycle_modified.csv')

bcycle_new['Checkout hour']


bcycle_new['am_commute_hrs'] = np.where((bcycle_new['Checkout hour'] >= 7 ) & (bcycle_new['Checkout hour'] <= 9), 1, 0)


bcycle_new['pm_commute_hrs'] = np.where((bcycle_new['Checkout hour'] >= 17 ) & (bcycle_new['Checkout hour'] <= 19),1, 0)

bcycle_new['encoded_cstation'] = label_encoder.fit_transform(bcycle_new["Checkout Station (Station Information)"])
bcycle_new['encoded_rstation'] = label_encoder.fit_transform(bcycle_new["Return Station (Station Information)"])


bcycle_model = bcycle_new[['encoded_cstation', 'encoded_rstation', 'am_commute_hrs', 'pm_commute_hrs', 'weekend']]



### Adding More Variables
label_encoder = preprocessing.LabelEncoder()

# Convert day of week variable to numeric
encoded_day = label_encoder.fit_transform(bcycle_new["Checkout Day of Week"])

# Initialize logistic regression model
log_model = linear_model.LogisticRegression()

#Try with more variables

# Convert more variables to numeric
encoded_day = label_encoder.fit_transform(bcycle_new["Checkout Day of Week"])
encoded_cstation = label_encoder.fit_transform(bcycle_new["Checkout Station (Station Information)"])
encoded_rstation = label_encoder.fit_transform(bcycle_new["Return Station (Station Information)"])

train_features = pd.DataFrame([encoded_cstation,
                              encoded_rstation,
                              encoded_day]).T

train_features

#Do a train/test split
X_train, X_test, y_train, y_test = train_test_split(train_features, bcycle_new['ot'], test_size=0.4, random_state=0)
# Initialize logistic regression model
log_model = linear_model.LogisticRegression()

# Train the model
log_model.fit(X = X_train ,
              y = y_train)


# Check trained model intercept
print(log_model.intercept_)

# Check trained model coefficients
print(log_model.coef_)
# Make predictions
preds = log_model.predict(X= X_test)

# Generate table of predictions vs actual
pd.crosstab(preds,y_test)
log_model.score(X = X_test,
                y = y_test)


#### Logistic Regression w/ more variables
X_train, X_test, y_train, y_test = train_test_split(bcycle_model, bcycle_new['ot'], test_size=0.4, random_state=0)


log_model2 = linear_model.LogisticRegression()

# Train the model
log_model2.fit(X = X_train ,
              y = y_train)

# Check trained model intercept
print(log_model2.intercept_)

# Check trained model coefficients
print(log_model2.coef_)
# Make predictions
preds = log_model2.predict(X= X_test)

# Generate table of predictions vs actual
pd.crosstab(preds,y_test)
log_model2.score(X = X_test,y = y_test)
