
# ======================================
# Análisis Exploratorio de Datos (EDA) 
# ======================================

# Import librerias
import pandas as pd
import pandas as df 
import matplotlib.pyplot as plt
import seaborn as sns

# Importar dataset
dataset = pd.read_csv("/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_Clean.csv")

# CORRELACIONES ENTRE VARIABLES NUMÉRICAS
corr = dataset[['Age', 'Disease_Duration', 'Positive_Symptom_Score', 'Negative_Symptom_Score', 'GAF_Score']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlaciones')
plt.show()


# RELACIÓN ENTRE DIAGNÓSTICO Y VARIABLES NUMÉRICAS
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.boxplot(x='Diagnosis', y='Positive_Symptom_Score', data=dataset)
plt.title('Relación entre Diagnostico y Sintomas Positivos')

plt.subplot(1, 3, 2)
sns.boxplot(x='Diagnosis', y='Negative_Symptom_Score', data=dataset)
plt.title('Relación entre Diagnostico y Sintomas Negativos')

plt.subplot(1, 3, 3)
sns.boxplot(x='Diagnosis', y='GAF_Score', data=dataset)
plt.title('Relación entre Diagnostico y GAF Score')
plt.tight_layout() 
plt.show()


# ANALISIS DE VALORES ATIPICOS
plt.figure(figsize=(8, 6))
sns.boxplot(x=dataset['GAF_Score'])
plt.title('Valores Atípicos en GAF Score')
plt.show()

