# ======================================
# Análisis Exploratorio de Datos (EDA) 
# ======================================

# Importar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import joblib


# ======================================
# Carga de datos
# ======================================

df_pca = pd.read_csv("/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_PCA.csv")
df_original = pd.read_csv("/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_Clean.csv")

if 'Patient_ID' not in df_pca.columns:
    df_pca['Patient_ID'] = df_original['Patient_ID']

X_pca = df_pca.drop(columns=['Patient_ID'])

# ======================================
#  Método del Codo para elegir K
# ======================================

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Método del Codo para elegir el número de clusters")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.show()


# ======================================
# Aplicar K-Means K = 2
# ======================================
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_pca)
labels = kmeans.labels_

df_pca['Cluster'] = labels

print(df_pca[['Patient_ID', 'Cluster']].head())

# ======================================
# Definir funciones de evaluación
# ======================================

def dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    inter_cluster_distances = [
        np.min(cdist(X[labels == i], X[labels == j]))
        for i in unique_clusters for j in unique_clusters if i != j
    ]
    intra_cluster_diameters = [
        np.max(cdist(X[labels == i], X[labels == i]))
        for i in unique_clusters
    ]
    return np.min(inter_cluster_distances) / np.max(intra_cluster_diameters)

def evaluar_clusters(X, labels, kmeans):
    silhouette_avg = silhouette_score(X, labels)
    cohesion = kmeans.inertia_
    centroides = kmeans.cluster_centers_
    separacion = np.min(cdist(centroides, centroides)[np.triu_indices(len(centroides), k=1)])
    ch_index = calinski_harabasz_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    dunn = dunn_index(X, labels)

    print(f'Silhouette Score: {silhouette_avg:.4f}')
    print(f'Cohesión (SSE / Inercia): {cohesion:.4f}')
    print(f'Separación mínima entre centroides: {separacion:.4f}')
    print(f'Calinski-Harabasz Index: {ch_index:.4f}')
    print(f'Davies-Bouldin Index: {db_index:.4f}')
    print(f'Dunn Index: {dunn:.4f}')

# Evaluar clusters
evaluar_clusters(X_pca, labels, kmeans)


# ======================================
#  Guardar los datos y modelos
# ======================================

# Guardar el dataset con la columna de clusters
df_original['Cluster'] = df_pca['Cluster']
df_original.to_csv('/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_Clustered.csv', index=False)

# Guardar modelo K-Means
joblib.dump(kmeans, '/home/mario/Documents/SchizoHealth/models/kmeans_model.pkl')

# ======================================
#  Visualización de los Clusters
# ======================================

# Visualizar los clusters en 2D
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_pca)

plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=labels, cmap='viridis', s=50)
plt.title('Visualización de los Clusters (K-Means)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(label='Cluster')
plt.show()

# Visualizar los clusters en 3D
# Reducir la dimensionalidad a 3 componentes principales
pca = PCA(n_components=3)
pca_components = pca.fit_transform(X_pca)

# Crear una figura 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos con colores según el cluster asignado
sc = ax.scatter(pca_components[:, 0], pca_components[:, 1], pca_components[:, 2], 
                c=labels, cmap='viridis', s=50, alpha=0.8)

# Etiquetas de los ejes
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('Visualización 3D de Clusters (K-Means)')

# Agregar una barra de color para representar los clusters
plt.colorbar(sc, label='Cluster')
plt.show()
