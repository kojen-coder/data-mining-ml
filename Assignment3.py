#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 2023

Author: Ko-Jen Wang

"""

import pandas as pd
import numpy as np

df = pd.read_csv("~/PycharmProjects/data-mining-ml/Sheet4.csv")
df
### Task 1: Pre-process the data
# 1. Drop observations with missing values
df = df.dropna()

# 2. Drop irrelevenat predictors if one exists. 'Name' is irrelevant as it is an identifier and will not have predictive power.
df = df.drop(columns=['Name'], axis=1)

# 3. Dummify all categorical predictors
df = pd.get_dummies(df, columns=['Manuf', 'Type'])
df.columns


### Task 2: Construct predictor and target variables
X = df.drop(columns=['Rating_Binary'])
y = df['Rating_Binary']
X.columns


### Task 3: Build a random forest model
# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

# Setting up parameter grid
param_grid ={
    'n_estimators': [50, 100, 150, 200],
    'max_features': [3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3, 4]
}

# Using GridSearchCV with Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random_state=0)
gs_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Train the model
gs_rf.fit(X_train, y_train)

# Make prediction using the best model
y_pred = gs_rf.best_estimator_.predict(X_test)

# Get the accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Best combination of hyperparameters: {gs_rf.best_params_}")  # Result: 'max_features': 3, 'min_samples_leaf': 2, 'n_estimators': 50
print(f"Best model performance (accuraacy score): {accuracy:.4f}")  # Result: 0.9600


### Task 4: Build a gradient boosting algorithm
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(random_state=0)
gs_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Train the model
gs_gbc.fit(X_train, y_train)

# Make prediction using the best model
y_pred_gbc = gs_gbc.best_estimator_.predict(X_test)

# Get the accuracy score
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
print(f"Best combination of hyperparameters: {gs_gbc.best_params_}")  # Result: 'max_features': 3, 'min_samples_leaf': 1, 'n_estimators': 50
print(f"Best model performance (accuraacy score): {accuracy_gbc:.4f}")  # Result: 0.9200




