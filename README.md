![Portada](portada.jpg)
# Análisis sobre los factores demográficos y clínicos en la esquizofrenia

## Descripción

Análisis exploratorio sobre una base de datos de factores demográficos y clínicos relacionados con la esquizofrenia. El objetivo de este análisis es explorar los datos para encontrar posibles patrones y/o relaciones. El flujo de trabajo fue el siguiente: 
* ETL (Transformación y limpieza de datos)
* EDA (Análisis exploratorio de datos) + Pruebas estadísticas preliminares (Kolmogorov-Smirnov, Levene, etc.)
* PCA (Análisis de los componentes principales) + Análisis de correlaciones
* K-means + Evaluación de la calidad de los clusters
* Análisis de los clusters (Gráficos y pruebas estadísticas para variables continuas y categóricas)

## Estructura del Proyecto

El proyecto está organizado en los siguientes directorios y archivos:

- `data/`: Contiene los conjuntos de datos utilizados para el análisis.
- `src/`: Contiene los scripts de Python utilizados para el procesamiento y análisis de datos.
  - `utils/`: Contiene funciones estadísticas y de visualización. 
- `scripts/`: Contiene los scripts de Python utilizados para el procesamiento y análisis de datos.
- `pagina/`: Contiene la página con los insights generados a partir del análisis de datos.
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

6. Para visualizar la pagina web, abre `index.html` en tu navegador: 
```bash
cd pagina
open index.html  # En macOS
xdg-open index.html  # En Linux
start index.html  # En Windows
```


## Contribución

Si deseas contribuir a este proyecto, por favor sigue estos pasos:

1. Haz un fork de este repositorio.
2. Crea una nueva rama para tu funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Realiza tus cambios y haz commit (`git commit -am 'Añadir nueva funcionalidad'`).
4. Sube tus cambios a tu repositorio fork (`git push origin nueva-funcionalidad`).
5. Crea un Pull Request.

## Contacto

Si tienes alguna pregunta o sugerencia sobre el proyecto, no dudes en ponerte en contacto conmigo a través de [mi correo electrónico](amadocamilaines@gmail.com).
