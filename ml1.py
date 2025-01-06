# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import loadtxt

# Step 2: Load your dataset
data = pd.loadtxt('data_Kclusters.txt')  # Make sure this file is in the right location

# Step 3: Use Elbow Method to find optimal K
wcss = []  # List to hold WCSS values
for k in range(1, 11):  # Try values of K from 1 to 10
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)  # inertia_ is the WCSS

# Step 4: Plot the Elbow Curve
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Step 5: Choose the optimal K (for example, 6 clusters based on the elbow) and fit the KMeans model
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(data)

# Step 6: Visualize the clusters (if data is 2D or can be projected to 2D)
plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.show()
