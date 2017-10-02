import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Read in the dataframe
df = pd.read_csv('sheet_checkouts_by_station_map_data.csv')

#Get all trips longer than 30 mins and the type is not equal to annual. Call it 'Overcharge'
#Get all trips longer than 30 mins and the type is not equal to annual
df2 = pd.DataFrame((df['Trip Duration (Minutes)'] > 30) & (df['Entry Pass Type'] !='Annual')\
|(df['Trip Duration (Minutes)'] > 30) & (df['Entry Pass Type'] != 'Student')\
|(df['Trip Duration (Minutes)'] > 60) & (df['Entry Pass Type'] == 'Annual')\
|(df['Trip Duration (Minutes)'] > 60) & (df['Entry Pass Type'] == 'Student'))

df['Overcharge'] = df2
# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()

# Convert day of week variable to numeric
encoded_day = label_encoder.fit_transform(df["Checkout Day of Week"])

# Initialize logistic regression model
log_model = linear_model.LogisticRegression()

#Try with more variables

# Convert more variables to numeric
encoded_day = label_encoder.fit_transform(df["Checkout Day of Week"])
encoded_cstation = label_encoder.fit_transform(df["Checkout Station (Station Information)"])
encoded_rstation = label_encoder.fit_transform(df["Return Station (Station Information)"])

train_features = pd.DataFrame([encoded_cstation,
                              encoded_rstation,
                              encoded_day]).T
#Do a train/test split
X_train, X_test, y_train, y_test = train_test_split(train_features, df['Overcharge'], test_size=0.4, random_state=0)
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
