from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler




def silhouette_s(matrix):
    silhouette_scores = []

    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(matrix)
        
        score = silhouette_score(matrix, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_n_clusters = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]

    return optimal_n_clusters




def PCA_func(result_vectors, n_components=20, top_k = 10):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X_scaled = scaler.fit_transform(result_vectors)
    X_low = pca.fit_transform(X_scaled)

    k = silhouette_s(X_low)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_low)

    result_vectors_norm = result_vectors / np.linalg.norm(result_vectors, axis=1, keepdims=True)

    centroids_med = []
    for i in range(k):
        cluster_points = result_vectors_norm[labels == i]
        centroid = np.median(cluster_points, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids_med.append(centroid)


    # общий центральный кластер
    #result_vectors_norm = result_vectors / np.linalg.norm(result_vectors, axis=1, keepdims=True)
    centroid_norm = np.median(result_vectors_norm, axis=0)
    centroid_norm /= np.linalg.norm(centroid_norm)
    centroids_med.append(centroid_norm)


    centroids_med = np.array(centroids_med)


    # Косинусное сходство каждого вектора с каждым центроидом
    cos_sim = result_vectors_norm @ centroids_med.T  # (n_samples, 3)
    max_sim = np.max(cos_sim, axis=1)                # максимальное сходство с любым кластером

    top_farthest_points_idx = np.argsort(max_sim)[:top_k]

    return top_farthest_points_idx





def Kernal_PCA_func(result_vectors, n_components=20, top_k = 10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(result_vectors)  

    kpca = KernelPCA(n_components=20, kernel='rbf', gamma=None)   # gamma=None => 1/n_features ≈ 0.00143
    X_low = kpca.fit_transform(X_scaled)

    k = silhouette_s(X_low)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_low)

    result_vectors_norm = result_vectors / np.linalg.norm(result_vectors, axis=1, keepdims=True)

    centroids_med = []
    for i in range(k):
        cluster_points = result_vectors_norm[labels == i]
        centroid = np.median(cluster_points, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids_med.append(centroid)
    centroids_med = np.array(centroids_med)


    # Косинусное сходство каждого вектора с каждым центроидом
    cos_sim = result_vectors_norm @ centroids_med.T  # (n_samples, 3)
    max_sim = np.max(cos_sim, axis=1)                # максимальное сходство с любым кластером

    top_farthest_points_idx = np.argsort(max_sim)[:top_k]

    return top_farthest_points_idx

