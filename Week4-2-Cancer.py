# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:30:39 2023

"""

## 1. Illustrative example from the slides
# Load libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

# Create a dataframe
data = np.array([['drug_z',0.0467,0.2471],['drug_y',0.0533,0.1912],['drug_x',0.0917,0.2794]]) # already normalized
column_names = ['Drug', 'Age (MMN)', 'Na/K (MMN)']
row_names  = ['A', 'B', 'C']
df = pd.DataFrame(data, columns=column_names, index=row_names)

# Construct variables
X = df.iloc[:,1:3]
y = df['Drug']

# Build a model
knn = KNeighborsClassifier(n_neighbors=1)
model = knn.fit(X,y)

# Make prediction for a new observation (age = 0.05, Na/K = 0.25)
new_obs = [[0.05,0.25]]
model.predict(new_obs)


## 2. Build KNN model using cancer.csv
# Load data and construct variables
df = pd.read_csv("~/PycharmProjects/data-mining-ml/cancer.csv")
X = df.iloc[:,0:9]
y = df['class']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

# Standardize the dataset
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler() # Z-score standardization
scaled_X_train = standardizer.fit_transform(X_train)
scaled_X_test = standardizer.transform(X_test)

# Build a model with k = 3 and using euclidean distance function
knn = KNeighborsClassifier(n_neighbors=3,p=2) #Setting the distance function. We set up euclidean distance function. 1 is for Manhattan distance function.
model = knn.fit(scaled_X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model.predict(scaled_X_test)

# Get accuracy score
from sklearn.metrics import recall_score
recall_score(y_test, y_test_pred) # we do not want to skip any patients that are positive, so we change accuracy to recall


## 3. Choosing k
from sklearn.metrics import accuracy_score
for i in range (15,25):
    knn = KNeighborsClassifier(n_neighbors=i)
    model = knn.fit(scaled_X_train,y_train)
    y_test_pred = model.predict(scaled_X_test)
    print("Accuracy score using k-NN with ",i," neighbors = "+str(accuracy_score(y_test, y_test_pred)))
    

## 4. Make prediction for a new observation using optimal K
new_obs = [[4,2,1,1,1,8,3,1,1]]
scaled_new_obs = standardizer.transform(new_obs) # Standardize the observation

knn = KNeighborsClassifier(n_neighbors=19, p=2) #Using Euclidean distance function
model = knn.fit(scaled_X_train,y_train)
model.predict(scaled_new_obs) # We see that this new patient does not have cancer.
# If you do not use the standardization, you might get a different result (which is lable 0). This is not correct.