import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read the data
data = np.loadtxt('data_Kclusters.txt')

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Calculate WCSS for different values of K
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bx-')
plt.xlabel('K Value')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.show()

# Based on elbow method, let's use optimal K
optimal_k = 3  # We'll determine this from the elbow plot

# Perform K-means clustering with optimal K
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans_optimal.fit_predict(data_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'K-means Clustering (K={optimal_k})')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()