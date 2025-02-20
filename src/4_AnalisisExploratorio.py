# ======================================
# Análisis Exploratorio de Datos (EDA) 
# ======================================

# Importar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Importar dataset
df = pd.read_csv("/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_Clean.csv")

# ======================================
# Análisis de Correlación
# ======================================

# Eliminar columnas no necesarias (ID y diagnóstico)
df = df.drop(columns=["Patient_ID", "Diagnosis"])

# Calcular matriz de correlación
correlation_matrix = df.corr()

# Visualizar la matriz de correlación con un heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación")
plt.show()

# Eliminar variables redundantes
df_cleaned = df.drop(columns=["Disease_Duration", "GAF_Score", "Negative_Symptom_Score"])

# ======================================
# Normalización de los Datos
# ======================================

# Seleccionar solo las columnas numéricas (sin ID ni Diagnosis)
numerical_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns

# Escalar los datos
scaler = StandardScaler()
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

# ======================================
# Aplicar PCA
# ======================================

# Aplicar PCA con 14 Componentes
pca_14 = PCA(n_components=14)
df_pca = pca_14.fit_transform(df_cleaned[numerical_cols])

# Crear un DataFrame con las componentes principales
df_pca_df = pd.DataFrame(df_pca, columns=[f"PC{i+1}" for i in range(14)])

# Guardar el DataFrame con las componentes principales
df_pca_df.to_csv("/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_PCA.csv", index=False)

# ======================================
# Visualización de las Cargas de Componentes Principales
# ======================================

# Obtener las cargas de los componentes principales
componentes_cargas = pca_14.components_

# Crear un DataFrame para mostrar las cargas de cada variable en cada componente
cargas_df = pd.DataFrame(componentes_cargas, columns=df_cleaned[numerical_cols].columns, 
                          index=[f"PC{i+1}" for i in range(14)])

# Mostrar las cargas de las componentes
print("Cargas de los componentes principales:")
print(cargas_df)

# Graficar las cargas de los primeros componentes principales
plt.figure(figsize=(10, 6))
sns.heatmap(cargas_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Cargas de las Componentes Principales")
plt.xlabel("Variables")
plt.ylabel("Componentes Principales")
plt.show()
