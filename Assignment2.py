# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 2023

@author: Ko-Jen Wang
"""
# Import data
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Task 2: create the dataset
data = {
    'y': ['black', 'blue', 'blue'],
    'x1': [1, 0, -1],
    'x2': [1, 0, -1]
}

df = pd.DataFrame(data)
df

### Task 3: Develop a KNN model, and specify k=2 without any other parameters
# construct variable
X = df[['x1', 'x2']]
y = df['y']

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a KNN model
knn=KNeighborsClassifier(n_neighbors=2)
model=knn.fit(X, y)

### Task 4: Make a prediction with a new observation where x1=0.1, and x2=0.1
# Using the model to predict the results based on the test dataset
new_observation=  pd.DataFrame(data = [[0.1,0.1]], columns=["x1", "x2"])
y_pred=model.predict(new_observation)
y_pred[0] # Black

### Task 5: Use predict_proba method instead of predict method, what are the probabiity that the target variable is Black & Blue?
y_pred_prob = knn.predict_proba(new_observation)
knn.classes_ # Black, Blue

print("The probability that the new observation is black:", y_pred_prob[0][0])
print("The probability that the new observation is blue:", y_pred_prob[0][1])

### Task 6: Make sure the K-NN algorithms uses distance in the classification task, we need to specify a parameter when we build the K-NN model.
# The command to initiate the KNN model should be?
# knn = KNeighborsClassifier(n_neighbors=2, weights='distance')



