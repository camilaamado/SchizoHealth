![Portada](portada.jpg)
# Análisis sobre los factores demográficos y clínicos en la esquizofrenia
                                               
                                                   

## Descripción
* Procesos de ETL (Extracción, Transformación y Carga), garantizando la consistencia y limpieza de los datos.
* Análisis Exploratorio de Datos (EDA), aplicando pruebas estadísticas como Kolmogorov-Smirnov, Levene y Chi-Cuadrado para evaluar distribuciones y diferencias entre grupos.
* Análisis de Componentes Principales (PCA) para analizar correlaciones entre variables y reducir la dimensionalidad.
* Clustering con K-Means, evaluando la calidad de los clusters mediante: Silhouette Score (cohesión y separación), Índice de Calinski-Harabasz, Índice de Davies-Bouldin, Índice de Dunn y Análisis de Centroides. 
* Evaluación de multicolinealidad entre variables utilizando el Factor de Inflación de Varianza (VIF).
* Analisis de los clusters resultantes a través de visualizaciones y pruebas estadísticas para variables tanto continuas como categóricas.

## Estructura del Proyecto

El proyecto está organizado en los siguientes directorios y archivos:

- `data/`: Contiene los conjuntos de datos utilizados para el análisis.
- `src/`: Contiene los scripts de Python utilizados para el procesamiento y análisis de datos.
  - `utils/`: Contiene funciones estadísticas y de visualización. 
- `scripts/`: Contiene los scripts de Python utilizados para el procesamiento y análisis de datos.
- `README.md`: Este archivo, que proporciona una descripción general del proyecto.

## Requisitos

Para ejecutar este proyecto, necesitarás tener instalados los siguientes paquetes de Python:

- numpy
- pandas
- matplotlib
- seaborn
- joblib
- scikit-learn
- scipy
- plotly

Puedes instalar estos paquetes utilizando `pip`:

```bash
pip install -r requirements.txt
```

## Uso

1. Clona este repositorio:

```bash
git clone https://github.com/camilaamado/SchizoHealth.git
```

2. Navega al directorio del proyecto:

```bash
cd SchizoHealth
```

3. Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

4. Ejecuta Jupyter Notebook:

```bash
jupyter notebook
```
5. Abre y ejecuta los notebooks en el directorio `src` para realizar el analisis de datos.


#

# REPORTE Y ANÁLISIS DE UNA BASE DE DATOS DEMOGRÁFICA Y CLÍNICA DE ESQUIZOFRENIA

## Transformación y Limpieza de Datos (ETL)

Se comenzó con la transformación y limpieza de datos. Primero, se realizó una revisión exploratoria general del dataset, analizando:

- **Tipos de datos**
- **Valores nulos**
- **Dimensión del dataset**
- **Estadísticas descriptivas** para variables numéricas y categóricas

Luego, se tradujeron los nombres de las columnas al inglés para estandarizar el dataset. Finalmente, se llevó a cabo una revisión y validación para asegurar que:

- Los cambios de nombre se aplicaron correctamente  
- Se calcularon estadísticas descriptivas finales  
- No existen valores nulos  

El dataset limpio fue almacenado en la carpeta `cleanData/`.
## Análisis Exploratorio de Datos (EDA)

### Distribución de Variables Numéricas

Se seleccionaron las siguientes variables numéricas para el análisis:

- **Edad**
- **Duración de la enfermedad**
- **Puntaje de síntomas positivos**
- **Puntaje de síntomas negativos**
- **Puntaje GAF**

Para analizar su distribución, se generaron **histogramas y boxplots**, permitiendo detectar valores atípicos.

#### Observaciones clave:

- La **distribución de la edad** es uniforme, con mayor concentración en los rangos de **18-21 años**, **49-52 años** y **76-80 años**.  
- La **duración de la enfermedad** se concentra principalmente en el rango de **0 a 3 años**.  
- El **puntaje de síntomas positivos** presenta una **distribución sesgada a la izquierda**, con valores concentrados entre **0 y 50**.  
- La **distribución del puntaje de síntomas negativos** también está **sesgada a la izquierda**, con la mayoría de los valores entre **0 y 50**.  
- El **puntaje GAF** tiene una **distribución sesgada a la derecha**, con la mayoría de los valores en el rango de **60 a 100**.  

#### Análisis de **boxplots**:

- La **duración de la enfermedad** presenta varios valores atípicos leves.  
- Tanto el **puntaje de síntomas positivos** como **negativos** tienen **bigotes superiores más pronunciados**, lo que refleja **colas largas** en sus distribuciones.  
- El **puntaje GAF** tiene un **bigote inferior muy largo**, concentrando el **50% de los casos en el rango superior**.  
- Las **hospitalizaciones** se concentran en el rango **0-1**, con valores atípicos que llegan hasta aproximadamente **10**.  

![Descripción](results/Analisis%20exploratorio/Histograma%20de%20las%20variables%20numéricas.png)

![Descripción](results/Analisis%20exploratorio/Boxplot%20para%20detectar%20valores%20atípicos.png)

### Distribución de Variables Categóricas

Se analizaron las siguientes variables categóricas:

- **Género**
- **Nivel de educación**
- **Estado civil**
- **Ocupación**
- **Nivel de ingresos**
- **Área de residencia**
- **Diagnóstico**
- **Historial familiar de esquizofrenia**
- **Consumo de sustancias**
- **Intentos de suicidio**
- **Soporte social**
- **Factores de estrés**
- **Adherencia a la medicación**

#### **Hallazgos principales:**

- El **género** se distribuye de manera **uniforme**.  
- El **nivel de educación** es **homogéneo**, con una leve mayoría en el **nivel 5 ("posgrado")**.  
- La distribución del **estado civil** es similar en todas las categorías.  
- La **ocupación**, el **nivel de ingresos** y el **área de residencia** se distribuyen de manera **uniforme**.  
- En la variable **diagnóstico**, la gran mayoría (**aproximadamente 7,000 casos**) corresponde a **"0: No esquizofrenia"**, mientras que solo **300 casos** corresponden a **"1: Esquizofrenia"**.  
- La **historia familiar de esquizofrenia** muestra que la mayoría de los casos (**más de la mitad**) no presenta predisposición genética.  
- El **consumo de sustancias** se concentra en **"0: No consumo de sustancias"** (alcohol, tabaco u otras drogas).  

![Análisis de Variables Categóricas 1](results/Analisis%20exploratorio/An%C3%A1lisis%20de%20Variables%20Categ%C3%B3ricas%201.png)

![Análisis de Variables Categóricas 2](results/Analisis%20exploratorio/An%C3%A1lisis%20de%20Variables%20Categ%C3%B3ricas%202.png)

![Análisis de Variables Categóricas 3](results/Analisis%20exploratorio/An%C3%A1lisis%20de%20Variables%20Categ%C3%B3ricas%203.png)

### Correlación entre Variables Numéricas

El análisis de correlación revela las siguientes relaciones significativas entre las variables numéricas:

- Existe una **correlación fuerte y positiva** entre la **duración de la enfermedad** y los **puntajes de síntomas positivos y negativos**. Esto indica que, a medida que aumenta el tiempo desde el diagnóstico, los síntomas tienden a intensificarse.  
- Se observa una **correlación fuerte y negativa** entre la **duración de la enfermedad** y el **puntaje GAF**, lo que sugiere que a mayor tiempo con la enfermedad, menor es el nivel de funcionamiento general del paciente.  
- Las variables clínicas **puntaje de síntomas positivos y negativos** muestran una **correlación fuerte y positiva** entre sí y con la **duración de la enfermedad**, lo que implica que los pacientes con mayor severidad en un tipo de síntoma tienden a presentar mayores niveles en el otro.  
- Tanto los **puntajes de síntomas positivos como negativos** presentan una **correlación negativa** con el **puntaje GAF**, lo que indica que, a mayor severidad de los síntomas, menor es el nivel de funcionamiento global del paciente.  
- La **edad** muestra una **correlación prácticamente nula** con las demás variables clínicas, con una leve tendencia negativa, lo que sugiere que no influye significativamente en la severidad de los síntomas ni en la duración de la enfermedad.  

![Matriz de correlación entre variables numéricas](results/Analisis%20exploratorio/Matriz%20de%20correlacion%20entre%20variables%20numericas.png)

### Relación entre Diagnóstico y Variables Clínicas

El análisis de la relación entre el **diagnóstico** y las **variables clínicas** muestra diferencias significativas en los **puntajes de síntomas** y el **nivel de funcionamiento global (GAF)**:

- Los sujetos con **diagnóstico 0** (sin esquizofrenia) presentan **puntajes de síntomas positivos y negativos** en el rango de **0 a 50**, concentrándose en valores bajos. Además, su **puntaje GAF** es predominantemente alto, ubicándose entre **60 y 100**, lo que indica un mejor nivel de funcionamiento general.  
- En contraste, los sujetos con **diagnóstico 1** (con esquizofrenia) muestran **puntajes más altos de síntomas positivos y negativos**, generalmente en el rango de **50 a 100**. Asimismo, su **puntaje GAF** es significativamente más bajo, oscilando entre **0 y 60**, lo que sugiere un deterioro en el funcionamiento global.  

![Relación entre dx y síntomas _gaf](results/Analisis%20exploratorio/Relacion%20entre%20dx%20y%20sintomas%20_gaf.png)

## Análisis de correlación entre variables

### 1. Correlaciones fuertes y significativas

- **Duración de la enfermedad vs. Puntajes de síntomas positivos y negativos**  
  Correlación fuerte y positiva con síntomas positivos (0.719) y síntomas negativos (0.702).  
  Esto indica que, a mayor duración de la enfermedad, mayor es la severidad de los síntomas.

- **Duración de la enfermedad vs. GAF Score**  
  Correlación fuerte y negativa (-0.704).  
  A medida que la enfermedad avanza, el nivel de funcionamiento global del paciente disminuye.

- **Hospitalizaciones vs. Puntajes de síntomas**  
  Correlaciones positivas con síntomas positivos (0.674) y síntomas negativos (0.674).  
  También una correlación negativa con GAF Score (-0.673).  
  Esto sugiere que los pacientes con síntomas más graves tienden a requerir más hospitalizaciones.

- **Puntajes de síntomas positivos y negativos**  
  Correlación muy fuerte entre sí (0.719).  
  Los pacientes con altos niveles de síntomas positivos suelen tener altos niveles de síntomas negativos.

- **Puntajes de síntomas negativos vs. GAF Score**  
  Correlación negativa fuerte (-0.713).  
  A mayor severidad de los síntomas negativos, peor es el funcionamiento global del paciente.

- **Adherencia a la medicación vs. Síntomas**  
  Correlación negativa con síntomas negativos (-0.290) y síntomas positivos (-0.287).  
  Correlación positiva con GAF Score (0.289).  
  Esto sugiere que una mejor adherencia al tratamiento está asociada con síntomas menos severos y mejor funcionamiento global.

### 2. Correlaciones moderadas

- **Intento de suicidio vs. Síntomas negativos (0.411) y positivos (0.410)**  
  Indica que los pacientes con síntomas más severos tienen mayor riesgo de intentos de suicidio.

- **Historial familiar vs. Síntomas negativos (0.334)**  
  Sugiere que los pacientes con antecedentes familiares de esquizofrenia tienden a tener síntomas negativos más pronunciados.

### 3. Correlaciones débiles o no significativas

- **Edad y variables clínicas**  
  La edad tiene correlaciones muy bajas con la mayoría de las variables clínicas, lo que sugiere que no es un factor determinante en la severidad de los síntomas.

- **Género y síntomas**  
  No muestra correlaciones significativas con los síntomas ni con la duración de la enfermedad.

- **Nivel socioeconómico (Ingresos, Ocupación, Área de Residencia)**  
  No presenta una relación fuerte con la enfermedad ni con el nivel de síntomas.

![Matriz de correlación (PCA)](results/Analisis%20exploratorio/matriz%20de%20correlacion%20%28pca%29.png)

## Análisis de Componentes Principales (PCA) y Análisis de Inflación de la Varianza (VIF)

### 1. Análisis de Inflación de la Varianza (VIF)

 **VIF Moderado:** 
  Las siguientes variables presentan un VIF moderado, lo que indica una moderada colinealidad:
  - **Duracion de la enfermedad:** VIF = 2.74
  - **Hospitalizaciones:** VIF = 2.43
  - **Puntaje de sintomas positivos:** VIF = 2.92
  - **Puntaje de sintomas negativos:** VIF = 2.83
  - **Puntaje GAF:** VIF = 2.84

Estas correlaciones moderadas entre las variables no deberían afectar gravemente al rendimiento del modelo, pero es importante tener en cuenta que las variables presentan cierta redundancia.

## Informe de Análisis de Componentes Principales (PCA)

### 1. Eliminación de Variables Redundantes

En el análisis anterior, se identificaron variables redundantes que podrían causar sobrecarga de información similar y ruido en el modelo. Específicamente, se decidió eliminar las siguientes variables:

- **Duración de la enfermedad**
- **GAF Puntaje**
- **Puntaje de síntomas negativos**

La razón detrás de esta eliminación es la alta correlación entre las variables, lo que podría generar multicolinealidad y afectar la interpretación del modelo. En particular, se observó que:

- Los **puntajes de síntomas positivos y negativos** presentan una fuerte correlación de 0.719, lo que indica que los dos están altamente relacionados.
- La **duración de la enfermedad y las hospitalizaciones** están correlacionadas con un valor de 0.674, sugiriendo que ambas variables reflejan aspectos similares del proceso clínico.
- **GAF Score y los puntajes de síntomas** presentan correlaciones negativas fuertes, lo que implica que el nivel de funcionamiento global (GAF) disminuye a medida que la severidad de los síntomas aumenta.

**Al eliminar estas variables, se reduce la redundancia y se mitiga el ruido en el modelo, lo que permite una interpretación más clara de los componentes principales.**

### 2. Normalización de los Datos

Dado que los datos contienen variables con diferentes escalas y unidades, se procedió a **normalizar** los datos. Esto es especialmente importante para técnicas como PCA y métodos de clustering basados en distancia, ya que la normalización asegura que todas las variables contribuyan de manera equitativa al análisis, evitando que las variables con mayor magnitud dominen el modelo.

### 3. Aplicación de Análisis de Componentes Principales (PCA)

Con los datos normalizados, se aplicó el **Análisis de Componentes Principales (PCA)** utilizando 14 componentes principales. Este enfoque permite reducir la dimensionalidad de los datos mientras conserva la mayor cantidad de varianza posible, facilitando la interpretación y visualización de los patrones subyacentes en los datos.

- **Número de Componentes Principales Seleccionados:** 14

Se creó un **DataFrame** con las 14 componentes principales, que se guardó para su posterior análisis.

### 4. Visualización de las Cargas de los Componentes Principales

Para entender mejor la contribución de cada variable a los componentes principales, se realizaron visualizaciones de las cargas de los primeros componentes principales. Estas gráficas permiten observar qué variables tienen un mayor peso en cada componente. 

![Heatmap componentes principales (PCA)](results/Analisis%20exploratorio/headmap%20componenetes%20principales%20%28pca%29.png)

![Cargas de componentes principales](results/Analisis%20exploratorio/Cargas%20de%20componentes%20principales.png)

### Interpretación de los primeros componentes principales

| **Componente** | **Variables con mayores cargas** | **Interpretación** |
|----------------|----------------------------------|--------------------|
| **PC1**        | Positive_Symptom_Score (0.541), GAF_Score (-0.305) | Relacionado con el estado de los síntomas positivos y la funcionalidad general. |
| **PC2**        | Education_Level (0.496), Occupation (0.516), Living_Area (0.014) | Asociado con factores socioeconómicos y educacionales (nivel educativo, ocupación, área de residencia). |
| **PC3**        | Positive_Symptom_Score (0.719), Negative_Symptom_Score (0.719) | Refleja la severidad de los síntomas (positivos y negativos). |
| **PC4**        | Disease_Duration (0.735), Hospitalizations (0.675) | Relacionado con la duración de la enfermedad y las hospitalizaciones, capturando la severidad crónica. |
| **PC5**        | Education_Level (-0.488), Marital_Status (0.483) | Refleja la relación entre el nivel educativo y el estado civil. |
| **PC6**        | Gender (0.654), Positive_Symptom_Score (0.452) | Relacionado con el género y la severidad de los síntomas positivos. |
| **PC7**        | Positive_Symptom_Score (0.722), Suicide_Attempt (0.411) | Captura la relación entre los intentos de suicidio y los síntomas positivos. |
#

## Aplicación del Método del Codo para Elegir el Número Óptimo de K
![Método codo para calcular número de clusters](results/Analisis%20exploratorio/metodo%20codo%20para%20calcular%20numero%20de%20clusters.png)

### 1. Selección del Número Óptimo de Clusters (K)

Antes de aplicar el algoritmo K-means, se utilizó el **método del codo** para determinar el número óptimo de clusters (K). Se evaluaron tres valores posibles de K: 2, 3 y 4 clusters. Para esto, se utilizaron las siguientes métricas:

- **Silhouette Score**: Mide la calidad de la agrupación, con valores cercanos a +1 indicando agrupaciones óptimas.
- **Cohesión y Separación**: Evalúan cuán cercanos están los puntos dentro de un cluster y la distancia entre clusters.
- **Calinski-Harabasz Index**: Compara la dispersión entre clusters con la dispersión interna; valores más altos indican mejor segmentación.
- **Davies-Bouldin Index**: Mide la similitud entre clusters; valores bajos indican buena separación.
- **Dunn Index**: Evalúa la densidad y separación de los clusters, buscando maximizar la distancia entre clusters y minimizar la interna.

### 2. Análisis y Resultados

Se probó K = 2, K = 3 y K = 4 clusters. Los resultados indicaron que **K = 2** era la mejor opción, ya que presentó el mejor **Silhouette Score** y una óptima separación entre los clusters.

### 3. Aplicación de K-means

Con K = 2 como número óptimo de clusters, se aplicó el algoritmo **K-means** para segmentar los datos en dos grupos significativos.

### 4. Almacenamiento de Resultados

Se guardaron tanto los **datos segmentados** como los **modelos generados**, lo que permite su reutilización en futuras etapas del análisis.

### 5. Visualización de los Clusters:
Finalmente, se realizó una visualización de los clusters obtenidos tanto en 2D como en 3D. Estas visualizaciones permiten observar la distribución y separación de los grupos, lo que facilita la interpretación de los resultados y la identificación de patrones.
![Cluster 2D](results/Analisis%20de%20los%20clusters/cluster%202d.png)
![Visualización 3D de los clusters](results/Analisis%20de%20los%20clusters/visualizacion%203d%20de%20los%20clusters.png)

### Distribución de Pacientes por Cluster:
* Cluster 0: 7361 pacientes (73.6%)
* Cluster 1: 2639 pacientes (26.4%) 

El Cluster 0 es el más grande, con casi el triple de pacientes que el Cluster 1.
![Distribución de clusters (gráfico de torta)](results/Analisis%20de%20los%20clusters/Distribucion%20de%20clusters%20%28grafico%20de%20torta%29.png)

## Graficos y test estadísticos de la distribución de los cluster (variables continuas)
## Cluster 0:

![Distribución de duración de la enfermedad](results/Analisis%20de%20los%20clusters/cluster%200/Distribucion%20de%20duracion%20enfermedad.png)

![Distribución de síntomas positivos](results/Analisis%20de%20los%20clusters/cluster%200/Distribucion%20de%20sintomas%20positivos.png)

![Distribución de síntomas negativos](results/Analisis%20de%20los%20clusters/cluster%200/Distribucion%20de%20sintomas%20negativos.png)
![Distribución de hospitalizaciones](results/Analisis%20de%20los%20clusters/cluster%200/Distribucion%20de%20hospitalizaciones.png)

![Distribución de GAF SCORE](results/Analisis%20de%20los%20clusters/cluster%200/Distribucion%20de%20GAF%20SCORE.png)

![Distribución de edad](results/Analisis%20de%20los%20clusters/cluster%200/Distribucion%20de%20edad.png)

## Cluster 1:

![Distribución de duración de la enfermedad](results/Analisis%20de%20los%20clusters/cluster%201/Distribucion%20de%20duracion%20enfermedad.png)

![Distribución de edad](results/Analisis%20de%20los%20clusters/cluster%201/Distribucion%20de%20edad.png)

![Distribución de GAF SCORE](results/Analisis%20de%20los%20clusters/cluster%201/Distribucion%20de%20GAF%20SCORE.png)

![Distribución de hospitalizaciones](results/Analisis%20de%20los%20clusters/cluster%201/Distribucion%20de%20hospitalizaciones.png)

![Distribución de síntomas negativos](results/Analisis%20de%20los%20clusters/cluster%201/Distribucion%20de%20sintomas%20negativos.png)

![Distribución de síntomas positivos](results/Analisis%20de%20los%20clusters/cluster%201/Distribucion%20de%20sintomas%20positivos.png)


| **Variable**               | **Estadístico KS** | **p-valor KS** | **Estadístico Levene** | **p-valor Levene** | **Estadístico Mann-Whitney U** | **p-valor Mann-Whitney U** | **Conclusión**                                               |
|----------------------------|--------------------|----------------|------------------------|-------------------|-------------------------------|----------------------------|-------------------------------------------------------------|
| **Edad**                   | 1.0000             | 0.0000         | 0.8023                 | 0.3704            | 9438804.0000                  | 0.0312                     | Diferencia significativa en la mediana de edad, sin diferencias en la varianza. |
| **Duración de la Enfermedad** | 0.5000             | 0.0000         | 7301.7711              | 0.0000            | 372452.0000                  | 0.0000                     | Diferencia significativa en la duración de la enfermedad. Mayor predominancia en el cluster 1. |
| **Hospitalizaciones**      | 0.5000             | 0.0000         | 17148.3431             | 0.0000            | 794917.0000                  | 0.0000                     | Diferencias significativas en el número de hospitalizaciones. Mayor predominancia en el cluster 1.  |
| **Síntomas Positivos**     | 0.9381             | 0.0000         | 14.8886                | 0.0001            | 210504.5000                  | 0.0000                     | Diferencias significativas en los síntomas positivos. Mayor predominancia en el cluster 1.|
| **Síntomas Negativos**     | 0.9423             | 0.0000         | 15.9587                | 0.0001            | 389546.5000                  | 0.0000                     | Diferencias significativas en los síntomas negativos. Con mayor predominanacia en el cluster 1. |
| **Puntuaciones GAF**       | 1.0000             | 0.0000         | 45.3159                | 0.0000            | 19040362.0000                | 0.0000                     | Diferencias significativas en el funcionamiento global. Con resultados mas altos en el cluster 1.  |

### Resumen:
- **Diferencias significativas** en todas las variables analizadas entre los dos clusters, lo que indica que los pacientes en cada cluster tienen características distintas en términos de edad, duración de la enfermedad, hospitalizaciones, síntomas positivos y negativos, y puntuaciones GAF.
- La **prueba de Kolmogorov-Smirnov** y la **prueba de Mann-Whitney U** muestran diferencias sustanciales, especialmente en la mediana de las distribuciones de cada variable.
- En algunos casos, la **prueba de Levene** indica diferencias en la **varianza**, lo que sugiere dispersión desigual en algunos clusters.






#
#
## Graficos y test estadísticos de la distribución de los cluster (variables categoricas)
al presentarse una diferencia cuantitativa entre los clusters bastante contundente, se eligio comparar mediante analisis de proporciones. 
* se calculo la porporcion de cada catgoria dentro de cada cluster.
* se grafico las porporciones de las categorias por cluster.
* test estadistico chi cuadrado para evaluar si habia diferencia significativa entre la variable de los clusters. 
* analisis de frcuencia relativa:

## Graficos de distribucion de cluster 0 y cluster 1: 

results/Analisis de los clusters/Distribucion de uso de sustancia.png
results/Analisis de los clusters/Distribucion de suicidios.png
results/Analisis de los clusters/Distribucion de nivel de ingresos.png
results/Analisis de los clusters/Distribucion de nivel de ingresos.png
results/Analisis de los clusters/Distribucion de nivel de educacion.png
results/Analisis de los clusters/Distribucion de genero.png
results/Analisis de los clusters/Distribucion de factores de estres.png
results/Analisis de los clusters/Distribucion de estado civil.png
results/Analisis de los clusters/Distribucion de dx.png
results/Analisis de los clusters/Distribucion de ayuda social.png
results/Analisis de los clusters/Distribucion de area de vivienda.png
results/Analisis de los clusters/Distribucion de antecedentes familiares.png
results/Analisis de los clusters/Distribucion de adherencia a la medicacion.png
results/Analisis de los clusters/Distribucion de  ocupacion.png

![Distribución de uso de sustancia](results/Analisis%20de%20los%20clusters/Distribucion%20de%20uso%20de%20sustancia.png)

![Distribución de suicidios](results/Analisis%20de%20los%20clusters/Distribucion%20de%20suicidios.png)
![Distribución de nivel de ingresos](results/Analisis%20de%20los%20clusters/Distribucion%20de%20nivel%20de%20ingresos.png)

![Distribución de nivel de ingresos](results/Analisis%20de%20los%20clusters/Distribucion%20de%20nivel%20de%20ingresos.png)

![Distribución de nivel de educación](results/Analisis%20de%20los%20clusters/Distribucion%20de%20nivel%20de%20educacion.png)

![Distribución de género](results/Analisis%20de%20los%20clusters/Distribucion%20de%20genero.png)
![Distribución de factores de estrés](results/Analisis%20de%20los%20clusters/Distribucion%20de%20factores%20de%20estres.png)
![Distribución de estado civil](results/Analisis%20de%20los%20clusters/Distribucion%20de%20estado%20civil.png)

![Distribución de diagnóstico](results/Analisis%20de%20los%20clusters/Distribucion%20de%20dx.png)

![Distribución de ayuda social](results/Analisis%20de%20los%20clusters/Distribucion%20de%20ayuda%20social.png)
![Distribución de área de vivienda](results/Analisis%20de%20los%20clusters/Distribucion%20de%20area%20de%20vivienda.png)

![Distribución de antecedentes familiares](results/Analisis%20de%20los%20clusters/Distribucion%20de%20antecedentes%20familiares.png)
![Distribución de adherencia a la medicación](results/Analisis%20de%20los%20clusters/Distribucion%20de%20adherencia%20a%20la%20medicacion.png)

![Distribución de ocupación](results/Analisis%20de%20los%20clusters/Distribucion%20de%20ocupacion.png)



### Variables Categóricas (Prueba de Chi-Cuadrado)

| **Variable**                | **p-Valor** | **Conclusión**                                      |
|-----------------------------|-------------|-----------------------------------------------------|
| **Género**                  | 0.3151      | No hay diferencia significativa                     |
| **Nivel Educativo**         | 0.4041      | No hay diferencia significativa                     |
| **Estado Civil**            | 0.8773      | No hay diferencia significativa                     |
| **Ocupación**               | 0.4006      | No hay diferencia significativa                     |
| **Nivel de Ingreso**        | 0.4189      | No hay diferencia significativa                     |
| **Área de Residencia**      | 0.0135      | Diferencia significativa                            |
| **Diagnóstico**             | 0.0000      | Diferencia significativa                            |
| **Historial Familiar**      | 0.0000      | Diferencia significativa                            |
| **Uso de Sustancias**       | 0.0000      | Diferencia significativa                            |
| **Intento de Suicidio**     | 0.0000      | Diferencia significativa                            |
| **Apoyo Social**            | 0.7769      | No hay diferencia significativa                     |
| **Factores de Estrés**      | 0.6784      | No hay diferencia significativa                     |
| **Adherencia a Medicación** | 0.0000      | Diferencia significativa                            |

#
*Las diferencias más marcadas entre clusters se dan en el diagnóstico, historial familiar, uso de sustancias, intentos de suicidio y adherencia a medicación.*
#

### Analisis de frecuencia relativa:

| **Variable**                  | **Cluster 0**      | **Cluster 1**      | **Mayor Preponderancia** |
|-------------------------------|--------------------|--------------------|--------------------------|
| **Género**                     |                    |                    |                          |
| Female                         | 50.05%             | 48.88%             | Cluster 0                |
| Male                           | 49.95%             | 51.12%             | Cluster 1                |
| **Nivel Educativo**            |                    |                    |                          |
| Nivel 1                        | 19.28%             | 19.89%             | Cluster 1                |
| Nivel 2                        | 19.58%             | 19.10%             | Cluster 0                |
| Nivel 3                        | 20.00%             | 19.82%             | Cluster 0                |
| Nivel 4                        | 19.37%             | 20.73%             | Cluster 1                |
| Nivel 5                        | 21.77%             | 20.46%             | Cluster 0                |
| **Estado Civil**               |                    |                    |                          |
| Soltero                        | 24.41%             | 23.76%             | Cluster 0                |
| Casado                         | 25.40%             | 25.35%             | Ambos                    |
| Divorciado                     | 24.51%             | 25.16%             | Cluster 1                |
| Viudo                          | 25.68%             | 25.73%             | Ambos                    |
| **Ocupación**                  |                    |                    |                          |
| Desempleado                    | 24.52%             | 23.61%             | Cluster 0                |
| Empleado                        | 25.23%             | 26.87%             | Cluster 1                |
| Estudiante                      | 24.83%             | 24.63%             | Cluster 0                |
| Otro                            | 25.42%             | 24.90%             | Cluster 0                |
| **Nivel de Ingreso**           |                    |                    |                          |
| Bajo                           | 33.28%             | 34.37%             | Cluster 1                |
| Medio                          | 33.11%             | 33.35%             | Ambos                    |
| Alto                           | 33.61%             | 32.28%             | Cluster 0                |
| **Área de Residencia**         |                    |                    |                          |
| Urbana                         | 49.31%             | 52.14%             | Cluster 1                |
| Rural                          | 50.69%             | 47.86%             | Cluster 0                |
| **Diagnóstico**                |                    |                    |                          |
| Diagnóstico A                  | 96.48%             | 0.42%              | Cluster 0                |
| Diagnóstico B                  | 3.52%              | 99.58%             | Cluster 1                |
| **Historial Familiar**         |                    |                    |                          |
| Sin antecedentes               | 79.46%             | 36.19%             | Cluster 0                |
| Con antecedentes               | 20.54%             | 63.81%             | Cluster 1                |
| **Uso de Sustancias**          |                    |                    |                          |
| No consume                     | 85.33%             | 58.43%             | Cluster 0                |
| Consume                        | 14.67%             | 41.57%             | Cluster 1                |
| **Intento de Suicidio**        |                    |                    |                          |
| No ha intentado                | 99.99%             | 66.54%             | Cluster 0                |
| Sí ha intentado                | 0.01%              | 33.46%             | Cluster 1                |
| **Apoyo Social**               |                    |                    |                          |
| Bajo                           | 33.28%             | 33.99%             | Cluster 1                |
| Medio                          | 33.84%             | 33.72%             | Cluster 0                |
| Alto                           | 32.88%             | 32.28%             | Cluster 0                |
| **Factores de Estrés**         |                    |                    |                          |
| Bajo                           | 33.22%             | 34.10%             | Cluster 1                |
| Medio                          | 33.42%             | 32.70%             | Cluster 0                |
| Alto                           | 33.37%             | 33.19%             | Cluster 0                |
| **Adherencia a la Medicación** |                    |                    |                          |
| Baja                           | 19.01%             | 52.33%             | Cluster 1                |
| Media                          | 30.65%             | 30.66%             | Ambos                    |
| Alta                           | 50.35%             | 17.01%             | Cluster 0                |




## Conclusiones finales: 


| **Variable**               | **Cluster 0**                               | **Cluster 1**                               | **Conclusión**                                                |
|----------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------------------------|
| **Diagnóstico**             | 96.48% en la categoría de *Not schizophrenic*, 3.52% en otra *Schizophrenic*     | 99.58% en la categoria de *Schizophrenic*           | El clustering separa a los pacientes con distintos diagnósticos. |
| **Historial Familiar**      | 79.46% sin antecedentes familiares          | 63.81% con antecedentes familiares          | Cluster 1 tiene mayor probabilidad de antecedentes familiares.  |
| **Uso de Sustancias**       | 85.3% no consume sustancias                 | 41.57% sí consume sustancias                | Cluster 1 tiene mayor prevalencia de consumo de sustancias.     |
| **Intento de Suicidio**     | 99.98% no ha intentado suicidarse           | 33.46% sí ha intentado suicidarse           | Cluster 1 tiene mayor incidencia de intentos de suicidio.       |
| **Adherencia a la Medicación** | 50.35% alta adherencia                    | 52.33% baja adherencia                      | Cluster 1 tiene más dificultades para seguir el tratamiento.   |






## Contribución

Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1. Haz un fork de este repositorio.
2. Crea una nueva rama para tu funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -am 'Añadir nueva funcionalidad'`).
4. Sube tus cambios a tu repositorio fork (`git push origin nueva-funcionalidad`).
5. Crea un Pull Request.

## Contacto

Si tienes alguna pregunta o sugerencia sobre el proyecto, no dudes en ponerte en contacto conmigo a través de [mi correo electrónico](amadocamilaines@gmail.com).
