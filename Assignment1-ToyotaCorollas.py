# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 2023

@author: Ko-Jen Wang
"""
# Import data
import pandas as pd
import numpy as np

df = pd.read_csv("~/PycharmProjects/data-mining-ml/ToyotaCorolla.csv")
df.columns
df.info()

### Task 0: Define the predictors as all variables except ID, Model, and Price. Define the target variable as Price
X = df.iloc[:, 3:]
y = df['Price']
X.columns  # Make sure that predictors are set up correctly


### Task 1: Check multicollinearity for each predictor.
# Create VIF dataframe
from statsmodels.tools.tools import add_constant
X1 = add_constant(X)
vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns

# Calculating VIF for each feature
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(len(X1.columns)):
    vif_data.loc[vif_data.index[i], "VIF"] = variance_inflation_factor(X1.values, i)

print(vif_data)
# The VIF of "Cylinders" is > 5, then we will remove that variable.

# Remove variable "Cylinders"
X.drop(columns=['Cylinders'], inplace=True)
X.columns


### Task 2: Standardize the predictors
from sklearn.preprocessing import StandardScaler
X.describe()
scaler=StandardScaler() # scales the features to have mean=0 and variance=1
scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, columns=X.columns)


### Task 3: Split the dataset into trat and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.35, random_state = 662)
X_train.columns


### Task 4: Build a Linear Regression
from sklearn.linear_model import LinearRegression
lm1 = LinearRegression()
model1 = lm1.fit(X_train, y_train)
y_test_pred = model1.predict(X_test)

# Calculate the MSE for the test set
from sklearn.metrics import mean_squared_error
lm1_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE using validation set approach = "+str(lm1_mse))


### Task 5: Develop a Ridge Regression (with alpha = 1)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)
ridge_model = ridge.fit(X_train,y_train)

ridge_model.coef_  # Print the coefficients

# Generate the prediction value from the test data
y_test_pred2 = ridge_model.predict(X_test)

# Calculate the MSE
ridge_mse = mean_squared_error(y_test, y_test_pred2)
print("Test MSE using ridge with penalty of 1 = "+str(ridge_mse))


### Task 6: Develop a LASSO Regression (with alpha = 1)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1)
lasso_model = lasso.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred3 = lasso_model.predict(X_test)

# Calculate the MSE
lasso_mse = mean_squared_error(y_test, y_test_pred3)
print("Test MSE using lasso with penalty of 1 = "+str(lasso_mse))

# Print the coefficients
lasso_model.coef_


### Task 7: Develop a Ridge and LASSO Regression (with alpha = 10, 100, 1000, 10000)
## Ridge Regression
for i in [10, 100, 1000, 10000]:
    ridge2 = Ridge(alpha=i)
    ridge_model2 = ridge2.fit(X_train,y_train)
    y_test_pred4 = ridge_model2.predict(X_test)
    print(f'Alpha = {i} / MSE = {mean_squared_error(y_test, y_test_pred4)}')

    if i == 10000:
    print(f'When Alpha = {i} / Coef = {ridge_model2.coef_}')
X_test.columns

## LASSO Regression
for i in [10, 100, 1000, 10000]:
    lasso2 = Lasso(alpha=i)
    lasso_model2 = lasso2.fit(X_train,y_train)
    y_test_pred5 = lasso_model2.predict(X_test)
    print(f'Alpha = {i} / MSE = {mean_squared_error(y_test, y_test_pred5)}')

    if i == 10000:
    print(f'When Alpha = {i} / Coef = {lasso_model2.coef_}')
