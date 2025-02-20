# ===============================================================================================================================================
# ................................................... Analisis de los clusters...................................................................
# ===============================================================================================================================================

# ======================================
# Importar librerías necesarias
# ======================================

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import scipy.stats as stats
from scipy.stats import chi2_contingency

# ================================================
# Cargar el dataset de distribución de clusters
# ================================================
df_clustered = pd.read_csv('/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_Clustered.csv')

# ======================================
# 1° Graficos y analisis de los cluster
# ======================================

print(f"Cantidad de pacientes en cada cluster: {df_clustered['Cluster'].value_counts()}")

plt.figure(figsize=(6, 6))
sns.set_palette(sns.color_palette("viridis", n_colors=len(df_clustered['Cluster'].unique())))
ax = df_clustered['Cluster'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, title='Distribución de Clusters')
plt.axis('equal')
plt.show()


# ========================================================================================
# 2° Graficos y test estadísticos de la distribución de los cluster (variables continuas)
# ========================================================================================

# Importar funciones de estadística
import sys
sys.path.append("/home/mario/Documents/SchizoHealth/src/utils")
from estadistica import plot_distribution, kolmogorov_smirnov_test, levene_test, t_test, mann_whitney_test

# ...................................................................................

# Comparar EDADES entre Clusters 0 y 1
cluster_0_age = df_clustered[df_clustered['Cluster'] == 0]['Age']
cluster_1_age = df_clustered[df_clustered['Cluster'] == 1]['Age']

print("Comparación de Edades entre Clusters 0 y 1")

# Visualizar distribución de grafica
plot_distribution(cluster_0_age, "Cluster 0 - Age")
plot_distribution(cluster_1_age, "Cluster 1 - Age")

# Test de normalidad
kolmogorov_smirnov_test(cluster_0_age)
kolmogorov_smirnov_test(cluster_1_age)

# Test de varianzas
levene_test(cluster_0_age, cluster_1_age)

# Elegir test de medias según normalidad
stat, p = kolmogorov_smirnov_test(cluster_0_age)
if p > 0.05:  # Si la distribución es normal
    t_test(cluster_0_age, cluster_1_age)
else:  # Si no es normal
    mann_whitney_test(cluster_0_age, cluster_1_age)

#............................................................................

# Comparar puntuacones de DURACION DE LA ENFERMEDAD entre Clusters 0 y 1
cluster_0_Disease_Duration = df_clustered[df_clustered['Cluster'] == 0]['Disease_Duration']
cluster_1_Disease_Duration = df_clustered[df_clustered['Cluster'] == 1]['Disease_Duration']

print("Comparación de duracion de la enfermedad entre Clusters 0 y 1")
# Visualizar distribución de grafica
plot_distribution(cluster_0_Disease_Duration, "Cluster 0 - Disease_Duration")
plot_distribution(cluster_1_Disease_Duration, "Cluster 1 - Disease_Duration")

# Test de normalidad
kolmogorov_smirnov_test(cluster_0_Disease_Duration)
kolmogorov_smirnov_test(cluster_1_Disease_Duration)

# Test de varianzas
levene_test(cluster_0_Disease_Duration, cluster_1_Disease_Duration)

# Elegir test de medias según normalidad
stat, p = kolmogorov_smirnov_test(cluster_0_Disease_Duration)
if p > 0.05:  # Si la distribución es normal
    t_test(cluster_0_Disease_Duration, cluster_1_Disease_Duration)
else:  # Si no es normal
    mann_whitney_test(cluster_0_Disease_Duration, cluster_1_Disease_Duration)

#............................................................................

# Comparar puntuacones de HOSPITALIZACIONES entre Clusters 0 y 1
cluster_0_Hospitalizations = df_clustered[df_clustered['Cluster'] == 0]['Hospitalizations']
cluster_1_Hospitalizations = df_clustered[df_clustered['Cluster'] == 1]['Hospitalizations']
print("Comparación de hospitalizaciones entre Clusters 0 y 1")

# Visualizar distribución de grafica
plot_distribution(cluster_0_Hospitalizations, "Cluster 0 - Hospitalizations")
plot_distribution(cluster_1_Hospitalizations, "Cluster 1 - Hospitalizations")

# Test de normalidad
kolmogorov_smirnov_test(cluster_0_Hospitalizations)
kolmogorov_smirnov_test(cluster_1_Hospitalizations)

# Test de varianzas
levene_test(cluster_0_Hospitalizations, cluster_1_Hospitalizations)

# Elegir test de medias según normalidad
stat, p = kolmogorov_smirnov_test(cluster_0_Hospitalizations)
if p > 0.05:  # Si la distribución es normal
    t_test(cluster_0_Hospitalizations, cluster_1_Hospitalizations)
else:  # Si no es normal
    mann_whitney_test(cluster_0_Hospitalizations, cluster_1_Hospitalizations)

#................................................................................

# comparar puntuaciones de SINTOMAS POSITIVOS entre Clusters 0 y 1
cluster_0_Positive_Symptom_Score = df_clustered[df_clustered['Cluster'] == 0]['Positive_Symptom_Score']
cluster_1_Positive_Symptom_Score = df_clustered[df_clustered['Cluster'] == 1]['Positive_Symptom_Score']
print("Comparación de sintomas positivos entre Clusters 0 y 1")

# Visualizar distribución de grafica
plot_distribution(cluster_0_Positive_Symptom_Score, "Cluster 0 - Positive_Symptom_Score")
plot_distribution(cluster_1_Positive_Symptom_Score, "Cluster 1 - Positive_Symptom_Score")

# Test de normalidad
kolmogorov_smirnov_test(cluster_0_Positive_Symptom_Score)
kolmogorov_smirnov_test(cluster_1_Positive_Symptom_Score)

# Test de varianzas
levene_test(cluster_0_Positive_Symptom_Score, cluster_1_Positive_Symptom_Score)

# Elegir test de medias según normalidad
stat, p = kolmogorov_smirnov_test(cluster_0_Positive_Symptom_Score)
if p > 0.05:  # Si la distribución es normal
    t_test(cluster_0_Positive_Symptom_Score, cluster_1_Positive_Symptom_Score)
else:  # Si no es normal
    mann_whitney_test(cluster_0_Positive_Symptom_Score, cluster_1_Positive_Symptom_Score)

#................................................................................
# comparar puntuaciones de SINTOMAS NEGATIVOS entre Clusters 0 y 1
cluster_0_Negative_Symptom_Score = df_clustered[df_clustered['Cluster'] == 0]['Negative_Symptom_Score']
cluster_1_Negative_Symptom_Score = df_clustered[df_clustered['Cluster'] == 1]['Negative_Symptom_Score']
print("Comparación de sintomas negativos entre Clusters 0 y 1")

# Visualizar distribución de grafica
plot_distribution(cluster_0_Negative_Symptom_Score, "Cluster 0 - Negative_Symptom_Score")
plot_distribution(cluster_1_Negative_Symptom_Score, "Cluster 1 - Negative_Symptom_Score")

# Test de normalidad
kolmogorov_smirnov_test(cluster_0_Negative_Symptom_Score)
kolmogorov_smirnov_test(cluster_1_Negative_Symptom_Score)

# Test de varianzas
levene_test(cluster_0_Negative_Symptom_Score, cluster_1_Negative_Symptom_Score)

# Elegir test de medias según normalidad
stat, p = kolmogorov_smirnov_test(cluster_0_Negative_Symptom_Score)
if p > 0.05:  # Si la distribución es normal
    t_test(cluster_0_Negative_Symptom_Score, cluster_1_Negative_Symptom_Score)
else:  # Si no es normal
    mann_whitney_test(cluster_0_Negative_Symptom_Score, cluster_1_Negative_Symptom_Score)

#................................................................................

#Comparar PUNTUACIONES de GAF_Score entre Clusters 0 y 1
cluster_0_GAF_Score = df_clustered[df_clustered['Cluster'] == 0]['GAF_Score']
cluster_1_GAF_Score = df_clustered[df_clustered['Cluster'] == 1]['GAF_Score']
print("Comparación de puntuaciones GAF entre Clusters 0 y 1")

# Visualizar distribución de grafica
plot_distribution(cluster_0_GAF_Score, "Cluster 0 - GAF_Score")
plot_distribution(cluster_1_GAF_Score, "Cluster 1 - GAF_Score")

# Test de normalidad
kolmogorov_smirnov_test(cluster_0_GAF_Score)
kolmogorov_smirnov_test(cluster_1_GAF_Score)

# Test de varianzas
levene_test(cluster_0_GAF_Score, cluster_1_GAF_Score)

# Elegir test de medias según normalidad
stat, p = kolmogorov_smirnov_test(cluster_0_GAF_Score)
if p > 0.05:  # Si la distribución es normal
    t_test(cluster_0_GAF_Score, cluster_1_GAF_Score)
else:  # Si no es normal
    mann_whitney_test(cluster_0_GAF_Score, cluster_1_GAF_Score)

#................................................................................

# ========================================================================================
# 3° Graficos y test estadísticos de la distribución de los cluster (variables categoricas)
# ========================================================================================
categorical_vars = [
    'Gender', 'Education_Level', 'Marital_Status', 'Occupation', 'Income_Level',
    'Living_Area', 'Diagnosis', 'Family_History', 'Substance_Use', 'Suicide_Attempt',
    'Social_Support', 'Stress_Factors', 'Medication_Adherence'
]

# DataFrame vacío para almacenar las proporciones
proportions = pd.DataFrame()

# Calcular la proporción de cada categoría dentro de cada cluster
for var in categorical_vars:
    temp_df = df_clustered.groupby('Cluster')[var].value_counts(normalize=True).unstack().fillna(0)
    temp_df.columns = [f"{var}_{col}" for col in temp_df.columns]  # Renombrar columnas para evitar conflictos
    proportions = pd.concat([proportions, temp_df], axis=1)
print(proportions)

# ....................................................................................................

# Graficar las proporciones de las categorías por cluster
color_palette = sns.color_palette("Set2", n_colors=len(categorical_vars))
for var in categorical_vars:
    cols = [col for col in proportions.columns if var in col]
    plot_data = proportions[cols]
    plot_data.plot(kind='bar', figsize=(10, 6), width=0.8, color=color_palette[:len(plot_data.columns)])
    plt.title(f"Distribución de {var} por Cluster")
    plt.ylabel("Proporción")
    plt.xlabel("Cluster")
    plt.xticks(rotation=0)
    plt.legend(title="Categorías", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ..............................................................................................

# Comparación con chi cuadrado : Si p-value es menor que 0.05 se puede rechazar la hipótesis nula y concluir que hay una diferencia significativa. 

# Calcular la tabla de contingencia para una variable categórica
for var in categorical_vars:
    table = pd.crosstab(df_clustered[var], df_clustered['Cluster'])
        
    chi2, p_value, _, _ = chi2_contingency(table)
    print(f'Chi-cuadrado para {var}: p-value = {p_value:.4f}')

# ..............................................................................................

# Análisis de frecuencia relativa
for var in categorical_vars:
    freq = pd.crosstab(df_clustered[var], df_clustered['Cluster'], margins=True, normalize='columns') * 100
    print(f'Frecuencias relativas de {var} por Cluster:')
    print(freq)
    print("\n")
