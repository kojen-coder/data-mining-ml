#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:47:40 2023

@author: ehan1
"""

## Import data and construct variables (use all predictors)
import pandas as pd
import numpy as np

bank_df = pd.read_csv("~/PycharmProjects/data-mining-ml/UniversalBank.csv")
bank_df=pd.get_dummies(data=bank_df, columns=["Education"])
# context: to predict whether a person accepts a personal loan
X = bank_df.iloc[:,2:16]
y = bank_df["Personal Loan"]


##### Build ANN model without feature selection #####
## Split the data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=5)

## Build ANN with one hidden layer and 11 nodes
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13), max_iter=1000, random_state=0, activation="logistic") # We set a bunch of hyperparameters. We use hidden layer sizes arguments to set up one hidden layer and 11 nodes in that hidden layer. We will learn how to set up multiple hidden layers later.
model = mlp.fit(X_train,y_train)
# hidden_layer_sizes: start with the number of predictors as nodes (whcih is 13 predictors in this case)
# max_iter: number of iterations. Once all the observatiosn passed the neural network -> count as one iteration. You can reduce the number to reduce the number of passes of iterations.
# random state: set up the initial weights and bias (theta) to be consistent across multiple runs. And then the algo will upadte the weights later.
# activation: the default is reLu for regression problem as it is more computationally efficient one. Now, we set it as "logistic", which is the same as "sigmoid" because it is a classification problem.

## Make prediction and evaluate accuracy
y_test_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)

## Varying the number of hidden layers
mlp2 = MLPClassifier(hidden_layer_sizes=(13, 13), max_iter=1000, random_state=0, activation="logistic") # the number of hidden layers in the neureal network.
model2 = mlp2.fit(X_train,y_train)
y_test_pred_2 = model2.predict(X_test)
accuracy_score(y_test, y_test_pred_2)
# hidden_layer_sizes=(13, 13): we create 2 hidden layers, and it means -> (the number of nodes in the first hidden layer, and the number of notes in the second hidden layer)
# Adding the hidden layers does not necesarily improve the prediction performance. The accurcy drops from 0.98 to 0.978
# Since two hidden layers' performance is not so good. We stick to the single hidden layer and start looking for the optmial size of the hidden layer, as below

## Cross-validate with different size of the hidden layer
from sklearn.model_selection import cross_val_score
for i in range (7,16):
    model3 = MLPClassifier(hidden_layer_sizes=(i),max_iter=1000, random_state=0, activation="logistic")
    scores = cross_val_score(estimator=model3, X=X, y=y, cv=5)
    print(i,':',np.average(scores))

# Result: Since the 13 has the highest accruacy score. We will set the hidden layer sizes as 13.
# You can be more efficent to use GridSearchCV to find the best combincations of hyperparatmeters. But since we haven't submitted our assignment yet, we will not cover in this class

## ANN with optimal size of hidden layer from above
mlp = MLPClassifier(hidden_layer_sizes=(13), max_iter=1000, random_state=0, activation="logistic")
model = mlp.fit(X_train,y_train)
y_test_pred = model.predict(X_test)
accuracy_ann = accuracy_score(y_test, y_test_pred)


##### Apply feature selection ##### 
## 1. LASSO
# Standardize predictors
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Run LASSO with alpha=0.01
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.01) # you can control the number of predictors through alpha. Increase alpha to impose more penalty, forcing more coefficeints to be 0. This helps control the number of features we are getting.
model = ls.fit(X_std,y)

pd.DataFrame(list(zip(X.columns,model.coef_)), columns = ['predictor','coefficient']) #print out the coefficients here

# Fit the model on new dataset and evaluate
X_lasso_train = X_train.drop(['Age','Experience','Education_2','Education_3'], axis=1)
X_lasso_test = X_test.drop(['Age','Experience','Education_2','Education_3'], axis=1)

model = mlp.fit(X_lasso_train,y_train)  # Using mlp, we want to use Lasso to see if the model performance improves.
y_test_pred = model.predict(X_lasso_test)
accuracy_ann_lasso = accuracy_score(y_test, y_test_pred)


## 2. Random forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0) #There are a lot of parameters, and hyperparameters tuning, now I set it as default.  Setting random state is trying to make the result reproducible because otherwise it wil be random every time.
model = randomforest.fit(X, y)

pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns = ['predictor','feature importance'])
# Can specify a threshold for the feature importance; common is 0.05 but it's always subjective. If it is above 0.05 we will include them. This is always subjective depending on your dataset and context.

# Fit the model on new dataset and evaluate
X_rf_train = X_train[['Securities Account','Online','CreditCard','Mortgage'], axis=1]
X_rf_test = X_test[['Securities Account','Online','CreditCard','Mortgage'], axis=1]

model = mlp.fit(X_rf_train,y_train)
y_test_pred = model.predict(X_rf_test)
accuracy_ann_rf = accuracy_score(y_test, y_test_pred)
# if you include more predictors -> yield a higher rate of R2 and accuracy even though these features are not necessarily important.
# If I have to select the two, I would go for random forest selection feature.

# feature selection v.s. dimension reduction
# dimension reduction: lose a lot of interpretability. Transform a certain features.
# feature selection: elimiateing certain features, keeping the original value of certain features.
# If you want to go for interpretability -> you can go for feature selection.

# Why do we capture prepenticular information in the PCA? Because we want to capture the remaining and the information do not correlate and overlap at all
# If there are two original predictos, and there are two principle components. In this case, we did not reduce the dimention of the data.
## 3. Principal component analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=13) # always require the data to be standardized
pca.fit(X_std)

pca.explained_variance_ratio_ # For example, the first compoenet is 0.16, suggesting that it contains 16% of the variability of the data.
# It is not correpsonging to the original predictors, so it is not interpretable. The first number just means the first principle component.

pca.components_ # The array element corresponds to the predictors. This means to create the first PC they need to assign the weights to each predictor. It is really hard to use this as interpretation.

# Using elbow method to select number of components. We use to draw the elbow graph to determine the kink.
import matplotlib.pyplot as plt
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# Fit the model on new dataset and evaluate
pca = PCA(n_components=6)  # number of components that are selected
X_pca = pca.fit_transform(X_std)
X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.33, random_state = 5)

model = mlp.fit(X_pca_train, y_train)
y_test_pred = model.predict(X_pca_test)
accuracy_ann_pca = accuracy_score(y_test, y_test_pred) # the accruacy score is about 0.94, which is the worst compared to other models so far.


## 4. Recursive feature elimination (Doesn't work with MLPclassifier, thus using logistic regression for demonstration) 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# neural network does not provide feature importance score
lr = LogisticRegression(max_iter=5000)
rfe = RFE(lr, n_features_to_select=3) #need to specify how many features you want at the end. Ask python to test all combinations of the three predictors.
model = rfe.fit(X, y)
model.support_ #shows whether each feature is used or not as a boolean value

pd.DataFrame(list(zip(X.columns,model.support_)), columns = ['predictor','ranking']) #ranks features by their importance
# We yield three features that the support vaue is True. We can then use different training set to test on these predictors/features.
# For using "support", we can see "which" predictors contribute. But using "ranking", we can see "how" these predictors contribute.

model.ranking_
pd.DataFrame(list(zip(X.columns,model.ranking_)), columns = ['predictor','ranking']) #ranks features by their importance
# When we use ranking, it shows more specific information rather than just True / False. In this case, "1" as the most important predictor