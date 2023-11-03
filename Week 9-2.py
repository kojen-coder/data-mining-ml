#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:32:56 2023

"""

## Example 1: Elbow method using Utilities.csv
# Import the data
import pandas as pd

utilities_df = pd.read_csv("~/PycharmProjects/data-mining-ml/Utilities.csv")
X = utilities_df[['Sales','Fuel_Cost']]

# Standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Calculate inertia for each value of k
from sklearn.cluster import KMeans
withinss = []
for i in range (2,8):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X_std)
    withinss.append(model.inertia_)

# Create a plot, elbow plot
from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7],withinss) # elbow may not be always reliable, it is subjective.


## Example 2: Silhouette method using Utilities.csv
# Import the data
utilities_df = pd.read_csv("~/PycharmProjects/data-mining-ml/Utilities.csv")
X = utilities_df[['Sales','Fuel_Cost']]

# Standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Silhouette analysis when k=3
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, n_init='auto') # add to prevent warning, but it wont change the code
model = kmeans.fit(X_std)
labels = model.labels_

from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std,labels) # label represents that observation assigned to the specific cluster. The Silhouette score is for individual observation.

df = pd.DataFrame({'label':labels,'silhouette':silhouette})

# Find the average score for each cluster
import numpy as np
print('Average Silhouette Score for Cluster 0: ',np.average(df[df['label'] == 0].silhouette)) # average score for cluster 0 (label = 0)
print('Average Silhouette Score for Cluster 1: ',np.average(df[df['label'] == 1].silhouette)) # average score for cluster 1 (label = 1)
print('Average Silhouette Score for Cluster 2: ',np.average(df[df['label'] == 2].silhouette)) # average score for cluster 2 (label = 2)

# We can also see the overall score of all clusters
from sklearn.metrics import silhouette_score
silhouette_score(X_std,labels) # Use silhouette score (not the silhouette_samples) to get the overall score / measure. It is an average of all the scores from cluster 0, 1, 2. The dataset we are using and the cluster labels we got are the same.
# Result: 0.54, which is over 0.5. This is pretty good, relfecting the situation of the reality

# Finding optimal K
for i in range (2,8):    
    kmeans = KMeans(n_clusters=i, n_init='auto')
    model = kmeans.fit(X_std)
    labels = model.labels_
    print(i,':',silhouette_score(X_std,labels))
# Result: Of out all of K values, 3 has the highest score of 0.54. We can say that 3 is the optimal K.

## Example 3: Pseudo-F statistics using Utilities.csv
# Import the data
utilities_df = pd.read_csv("~/PycharmProjects/data-mining-ml/Utilities.csv")
X = utilities_df[['Sales','Fuel_Cost']]

# Standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Run kmeans with k=3
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, n_init='auto')
model = kmeans.fit(X_std)
labels = model.labels_

# Calculate F-score
from sklearn.metrics import calinski_harabasz_score # essentially the F score
score = calinski_harabasz_score(X_std, labels)

# Calculate p-value
from scipy.stats import f
df1 = 2 # df1 = k-1, First degree of freedom
df2 = 19 # df2 = n-k, Second degree of freedom (22-3 - 19). N is the number of observations. K=3
pvalue = 1-f.cdf(score, df1, df2) # Result: we get a very small value. Near 0.

# Finding optimal K
for i in range (2,8):    
    df1=i-1
    df2=22-i
    kmeans = KMeans(n_clusters=i, n_init='auto')
    model = kmeans.fit(X_std)
    labels = model.labels_
    score = calinski_harabasz_score(X_std, labels)
    print(i,'F-score:',score)
    print(i,'p-value:',1-f.cdf(score, df1, df2)) # We find the highest F-score, and lowest p value. We usually go for the F value. K=7 is the optimal, but the p-value is not reliable (we know from before the K=3), So the K we go for 3.
    # We prioritize F-score to be higher becuase that is a direct measure of the ratio.