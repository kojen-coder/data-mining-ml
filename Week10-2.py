# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:00:39 2023

@author: ehan1
"""
## Example 1: DBSCAN using class example
import pandas as pd
from matplotlib import pyplot

df = pd.read_csv("~/PycharmProjects/data-mining-ml/ClusterExample.csv")

# Creating a scatterplot
pyplot.scatter(df['x0'], df['x1'])
pyplot.xlabel("Feature 0")
pyplot.ylabel("Feature 1")
pyplot.show() # Result: From the graph, we can see that there are 5 clusters. So we can run K-means below with the K-5

# Building K-Means with K=5 and plotting clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, n_init='auto')
model = kmeans.fit(df)
labels = model.predict(df)

# Adding the color based on the label
pyplot.scatter(df['x0'], df['x1'], c=labels, cmap='rainbow')
pyplot.xlabel("Feature 0")
pyplot.ylabel("Feature 1")
pyplot.show()

# Running DBSCAN and plotting clusters
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=4) #defualt value is 0.5. Min-samples: we have two features. The rule of thumb is the twice of the num of features
labels = dbscan.fit_predict(df) # We see label column that represents which cluster. When it is -1, that means the outlier and noise.

pyplot.scatter(df['x0'], df['x1'], c=labels, cmap='rainbow') 
pyplot.xlabel("Feature 0")
pyplot.ylabel("Feature 1")
pyplot.show() # Result: There are number like Very large nuber of clusters, and very large number of outliers.
# Then we can try esp=0.8 to see its output
# In clustering, there is no absolute optimal solution. You need to try different combo of methods.

## Example 2: Isolation Forest Model on production.csv
df = pd.read_csv("production.csv")

# Creating a scatter plot
pyplot.scatter(df['raw_material'], df['time'])
pyplot.xlabel("Raw material")
pyplot.ylabel("Time")
pyplot.show()

# Building isolation forest model
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, contamination=.02)

pred = iforest.fit_predict(df)
score = iforest.decision_function(df)

# Extracting anomalies
from numpy import where
anomaly_index = where(pred==-1)
anomaly_values = df.iloc[anomaly_index]

# Scatter plot with anomalies
pyplot.scatter(df['raw_material'], df['time'])
pyplot.scatter(anomaly_values['raw_material'], anomaly_values['time'], color='r')
pyplot.xlabel("Raw material")
pyplot.ylabel("Time")
pyplot.show()


## Bonus example: Isolation Forest Model on Amtrak.csv
df = pd.read_csv("Amtrak.csv")
df["Month"] = pd.to_datetime(df["Month"], format="%d/%m/%Y")
df["Month"]  = df["Month"].apply(lambda x : x.strftime("%Y-%m-%d")) 
df = df.set_index(df["Month"])

# Plotting the data
pyplot.plot(df[['Ridership']])
fig, ax = pyplot.subplots(figsize=(10, 5))
pyplot.plot(df.index, df['Ridership'])
pyplot.xticks([])
pyplot.show()

# Create isolation forest model
iforest = IsolationForest(contamination=.1)

pred = iforest.fit_predict(df[['Ridership']])
score = iforest.decision_function(df[['Ridership']])

# Extracting anomalies
from numpy import where
anomaly_index = where(pred==-1)
anomaly_values = df.iloc[anomaly_index]

# Line plot with anomalies
fig, ax = pyplot.subplots(figsize=(10, 5))
pyplot.plot(df.index, df['Ridership'])
pyplot.scatter(anomaly_values.index, anomaly_values['Ridership'], color='r')
pyplot.xticks([])
pyplot.show()


