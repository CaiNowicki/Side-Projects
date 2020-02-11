import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = 'SalePrice'
X_train = train.drop(target, axis=1)
y_train = train[target]
X_test = test

for column in X_train.columns:
    try:
        X_train[column] = X_train[column].fillna(X_train[column].mean())
    except TypeError:
        X_train[column] = X_train[column].fillna('unknown')
for column in X_test.columns:
    try:
        X_test[column] = X_test[column].fillna(X_test[column].mean())
    except TypeError:
        X_test[column] = X_test[column].fillna('unknown')

X_train_encoded = OrdinalEncoder().fit(X_train).transform(X_train)
X_train_scaled = StandardScaler().fit(X_train_encoded).transform(X_train_encoded)

X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)

X_test_encoded = OrdinalEncoder().fit(X_test).transform(X_test)
X_test_scaled = StandardScaler().fit(X_test_encoded).transform(X_test_encoded)

X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)
# model = LinearRegression(n_jobs=-1)
# model.fit(X_train_scaled, y_train)
# print("Linear", model.score(X_train_scaled, y_train))
#
# model = LogisticRegression(n_jobs=-1, max_iter=10000)
# model.fit(X_train_scaled, y_train)
# print("Logistic", model.score(X_train_scaled, y_train))

model = RandomForestRegressor().fit(X_train_scaled, y_train)
print('Random Forest', model.score(X_train_scaled, y_train))

preds = pd.DataFrame(data=model.predict(X_test_scaled), columns = ['SalePrice'])
preds['Id'] = X_test['Id']

preds.to_csv('predictions.csv')