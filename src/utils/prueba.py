


###################################### VALIDAR CLUSTERS  ########################################

# Silhouette Score para evaluar la calidad de los clusters
silhouette_avg = silhouette_score(X_pca, labels)
print(f'Silhouette Score: {silhouette_avg:.4f}')

# Cohesion : suma de la distancia de los cuadrados de cada punto de los centroides. 💡 Cuanto más bajo el valor, más compactos son los clusters.
print(f'Cohesión (SSE / Inercia): {kmeans.inertia_:.4f}')

#Separación: Mide qué tan bien separados están los clusters. 💡 Cuanto mayor sea el valor, mejor separados están los clusters.
centroides = kmeans.cluster_centers_ # Calcular la distancia entre cada par de centroides
distancias_centroides = cdist(centroides, centroides, metric='euclidean')
separacion = distancias_centroides[np.triu_indices(len(centroides), k=1)].min()  # Tomar la distancia mínima entre clusters
print(f'Separación mínima entre centroides: {separacion:.4f}')

# Calinski-Harabasz Index 
ch_index = calinski_harabasz_score(X_pca, labels)
print(f'Calinski-Harabasz Index: {ch_index:.4f}')

# Calcular el índice de Davies-Bouldin
db_index = davies_bouldin_score(X_pca, labels)
print(f'Davies-Bouldin Index: {db_index:.4f}')

# Dunn Index
def dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    
    # Calcular la distancia mínima entre clusters (Separación)
    inter_cluster_distances = []
    for i in unique_clusters:
        for j in unique_clusters:
            if i != j:
                points_i = X[labels == i]
                points_j = X[labels == j]
                inter_cluster_distances.append(np.min(cdist(points_i, points_j)))
    
    # Calcular el diámetro máximo dentro de un cluster (Cohesión)
    intra_cluster_diameters = []
    for i in unique_clusters:
        points_i = X[labels == i]
        intra_cluster_diameters.append(np.max(cdist(points_i, points_i)))
    
    # Dunn Index = (mínima distancia inter-cluster) / (máximo diámetro intra-cluster)
    return np.min(inter_cluster_distances) / np.max(intra_cluster_diameters)

# Calcular el Dunn Index para tu clustering
dunn = dunn_index(X_pca, labels)
print(f'Dunn Index: {dunn:.4f}')




# ======================================
# Visualizar los Clusters
# ======================================
plt.figure(figsize=(8, 6))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=labels, cmap='viridis')
plt.title("Visualización de los Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Cluster')
#plt.show()

# ======================================

# Asegúrate de importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar el DataFrame de las componentes principales
df_pca = pd.read_csv("/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_PCA.csv")

# Asegúrate de que las etiquetas del clustering se generen con KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_pca[['PC1', 'PC2', 'PC3']])  # Ajustar el modelo con las primeras tres componentes

# Asignar las etiquetas de los clusters al DataFrame
df_pca['Cluster'] = kmeans.labels_

# Verifica que la columna 'Cluster' esté correctamente agregada
print(df_pca[['PC1', 'PC2', 'PC3', 'Cluster']].head())  # Asegúrate de que 'Cluster' está presente

# Crear la visualización 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotear los puntos en 3D con los colores basados en los clusters
scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], c=df_pca['Cluster'], cmap='viridis')

# Añadir etiquetas a los ejes
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# Título
ax.set_title('Visualización 3D de los Clusters')

# Añadir barra de colores para los clusters
fig.colorbar(scatter)

# Mostrar la gráfica
#plt.show()











###################################### ANALISIS POR CLUSTER (DESCRIPTIVO ) ########################################

# Asegúrate de que df_original está correctamente cargado
df_original = pd.read_csv("/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_Clean.csv")

# Verifica que las columnas existen
required_columns = ['Age', 'Gender', 'Education_Level']
if all(column in df_original.columns for column in required_columns):
    df_pca['Cluster'] = labels  # Etiquetas de los clusters
    df_pca[['Age', 'Gender', 'Education_Level']] = df_original[['Age', 'Gender', 'Education_Level']]  # Agregar las columnas necesarias

    # Analizar estadísticas descriptivas por cluster
    cluster_age_summary = df_pca.groupby('Cluster')['Age'].describe()
    print(cluster_age_summary)

    # Crear un boxplot para ver la distribución de la edad por cluster
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_pca, x='Cluster', y='Age')
    plt.title('Distribución de Edad por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Edad')
    plt.show()

    # Analizar estadísticas descriptivas por cluster
    cluster_gender_summary = df_pca.groupby('Cluster')['Gender'].describe()
    print(cluster_gender_summary)
else:
    print("Las columnas 'Age', 'Gender' y/o 'Education_Level' no existen en df_original")


print(df_pca.head())
print(df_pca.columns)
print(df_original.head())
print(df_original.columns)