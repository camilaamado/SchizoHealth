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

# ANALISIS DE VARIABLES NUMERICAS

# Histograma 
dataset[['Age', 'Disease_Duration', 'Positive_Symptom_Score', 'Negative_Symptom_Score', 'GAF_Score']].hist(bins=20, figsize=(15, 10))
plt.show()

# Boxplot para detectar valores atípicos
plt.figure(figsize=(12, 8))
sns.boxplot(data=dataset[['Age', 'Disease_Duration', 'Positive_Symptom_Score', 'Negative_Symptom_Score', 'GAF_Score', 'Hospitalizations']])
plt.show()

#ANALISIS DE VARIABLES CATEGORICAS

# Gráfico de barras para las variables categóricas


plt.figure(figsize=(18, 6))  # Tamaño de la figura en pulgadas

# Género
plt.subplot(1, 3, 1)
sns.countplot(x='Gender', data=dataset)
plt.title('Distribución de Género')

# Nivel de Educación
plt.subplot(1, 3, 2)
sns.countplot(x='Education_Level', data=dataset)
plt.title('Distribución de Nivel de Educación')

# Estado Civil
plt.subplot(1, 3, 3)
sns.countplot(x='Marital_Status', data=dataset)
plt.title('Distribución de Estado Civil')

plt.tight_layout()  # Ajusta el espaciado entre gráficos
plt.show()


# Ocupación
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.countplot(x='Occupation', data=dataset)
plt.title('Distribución de Ocupación')

#Nivel de Ingresos
plt.subplot(1, 3, 2)
sns.countplot(x='Income_Level', data=dataset)
plt.title('Distribución de Nivel de Ingresos')

#Área de Residencia
plt.subplot(1, 3, 3)
sns.countplot(x='Living_Area', data=dataset)
plt.title('Distribución de Área de Residencia')
plt.tight_layout() 
plt.show()

#Diagnóstico
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.countplot(x='Diagnosis', data=dataset)
plt.title('Distribución de Diagnóstico')

# Presencia de Historial Familiar
plt.subplot(1, 3, 2)
sns.countplot(x='Family_History', data=dataset)
plt.title('Distribución de Historial Genetica Familiar')

# Consumo de Sustancias
plt.subplot(1, 3, 3)
sns.countplot(x='Substance_Use', data=dataset)
plt.title('Distribución de Consumo de Sustancias')
plt.tight_layout() 
plt.show()

# Intento de Suicidio
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
sns.countplot(x='suicide_Attempt', data=dataset)
plt.title('Intentos de suicidio')

# Soporte social
plt.subplot(1, 3, 2)
sns.countplot(x='Social_Support', data=dataset)
plt.title('soporte social')

# Factor de estres
plt.subplot(1, 3, 3)
sns.countplot(x='Stress_Factors', data=dataset)
plt.title('Factores de estres')
plt.tight_layout() 
plt.show()

# Adherencia a la medicacion 
plt.figure(figsize=(10, 6))
sns.countplot(x='Medication_Adherence', data=dataset)
plt.title('Adeherencia a la medicacion')
plt.show()






