# ===========================================
# Transformación y Limpieza de Datos (ETL - Transform)
# =========================================== 


# importar librerías
import pandas as pd

# cargar datos
dataset = pd.read_csv("/home/mario/Documents/SchizoHealth/data/cleandData/schizophrenia_dataset.csv")


# Ver las primeras filas del dataset para obtener una visión general
print(dataset.head())

# Obtener información general del dataset (tipos de datos, valores nulos, etc.)
print(dataset.info())

# Ver los valores nulos y el porcentaje de nulos por columna
missing_values = dataset.isnull().sum()
missing_percentage = (missing_values / len(dataset)) * 100
print(pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}))

# Verificar los tipos de datos
print(dataset.dtypes)

# Estadísticas descriptivas para datos numéricos
print(dataset.describe())

# Estadísticas para datos categóricos
print(dataset['Eğitim_Seviyesi'].value_counts())

# cambiar el nombre de las columnas

# Diccionario con los nuevos nombres
new_column_names = {
    'Hasta_ID': 'Patient_ID',
    'Yaş': 'Age',
    'Cinsiyet': 'Gender',
    'Eğitim_Seviyesi': 'Education_Level',
    'Medeni_Durum': 'Marital_Status',
    'Meslek': 'Occupation',
    'Gelir_Düzeyi': 'Income_Level',
    'Yaşadığı_Yer': 'Living_Area',
    'Tanı': 'Diagnosis',
    'Hastalık_Süresi': 'Disease_Duration',
    'Hastaneye_Yatış_Sayısı': 'Hospitalizations',
    'Ailede_Şizofreni_Öyküsü': 'Family_History',
    'Madde_Kullanımı': 'Substance_Use',
    'İntihar_Girişimi': 'Suicide_Attempt',
    'Pozitif_Semptom_Skoru': 'Positive_Symptom_Score',
    'Negatif_Semptom_Skoru': 'Negative_Symptom_Score',
    'GAF_Skoru': 'GAF_Score',
    'Sosyal_Destek': 'Social_Support',
    'Stres_Faktörleri': 'Stress_Factors',
    'İlaç_Uyumu': 'Medication_Adherence'
}

# Renombrar las columnas del dataset
dataset.rename(columns=new_column_names, inplace=True)
print(dataset.head())

# Revisión Final y Validación

# Ver las primeras filas del dataset limpio
print(dataset.head())

# Ver el resumen estadístico para confirmar que no hay errores evidentes
print(dataset.describe())

# Asegurarte de que no hay valores nulos
print(dataset.isnull().sum())


# Guardar el dataset limpio en un nuevo archivo CSV
dataset.to_csv('/home/mario/Documents/SchizoHealth/data/cleanData/SchizoHealth_Clean.csv', index=False)

