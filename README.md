<p align="center">
    <img src="https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png">
</p>

# PROYECTO INDIVIDUAL Nº1 
## LONDERO WALTER OSCAR 
### Machine Learning Operations (MLOps)

<p align="center">
    <img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png" height=300>
</p>
  
Bienvenidos al proyecto individual de **MLOps Engineer**.

---

Dirección del trabajo en Producción con <a href="https://pi-mlops-06.onrender.com" > FastAPI </a>

Dirección del video explicativo en <a href="https://pi-mlops-06.onrender.com" > YouTube </a>

Dirección del trabajo en el repositorio de <a href="https://github.com/londerowalteroscar/PI_MLOps_06" > GitHub </a>



# Instalación de Librerías

pip install 
  - pandas 
  - numpy 
  - matplotlib 
  - fastapi 
  - textblob 
  - scikit-learn
  - scipy 


# ETL (Extract, Transform, Load)

## Extracción (Extract):

- Se definen funciones para abrir conjuntos de datos comprimidos, considerando distintos formatos y codificaciones.
- Se cargan y descomprimen los conjuntos de datos relacionados con reseñas de usuarios, ítems y juegos de Steam.
- Se muestran los DataFrames resultantes.

## Transformación (Transform):

- Se desanidan los diccionarios internos presentes en ciertas columnas de los DataFrames, creando nuevas columnas para cada clave especificada.
 -Se realizan diversas operaciones de limpieza y transformación de los datos, como la conversión de tipos de datos, eliminación de filas nulas y detección y eliminación de filas duplicadas.
 -Se realiza un análisis de sentimiento en las reseñas de usuarios, categorizando estas en positivas, negativas o neutrales.
 -Se generan gráficos exploratorios para visualizar la distribución de sentimientos en las reseñas y su evolución a lo largo del tiempo.
 -Se guardan los DataFrames resultantes en archivos CSV.

## Carga (Load):

- Se cargan nuevamente los DataFrames desde los archivos CSV generados previamente.

## Análisis Exploratorio de Datos (EDA)

- Se realiza una descripción general de los DataFrames utilizados en el proyecto, mostrando información relevante como tipos de datos, estadísticas descriptivas, etc.
- Se exploran posibles outliers en ciertas columnas, realizando un análisis estadístico y visual para identificar valores atípicos.
- Se realizan análisis adicionales, como la generación de gráficos de pastel y de líneas para visualizar tendencias y distribuciones.

## Funciones para API

- Se definen varias funciones destinadas a ser utilizadas en una API, las cuales permiten realizar consultas y obtener información específica sobre desarrolladores, usuarios, géneros de juegos, etc.
- Modelos de ML para Sistema de Recomendación
- Se comparte un ejemplo de implementación de un sistema de recomendación item-item utilizando el algoritmo k-Nearest Neighbors (k-NN) y la similitud de cosenos.
- Se proporciona un script para obtener recomendaciones de juegos similares a uno dado, basándose en géneros y etiquetas.

El análisis EDA-ETL proporciona información valiosa sobre la calidad de los juegos en la plataforma de Steam, así como sobre las tendencias y patrones en las opiniones de los usuarios. Los resultados obtenidos de este análisis pueden ser utilizados para realizar mejoras en los productos y para comprender mejor las preferencias de los consumidores.



# Descripción de las Funciones de la API

## Funciones

1. **developer(developer: str)**:
   - Esta función devuelve la cantidad de ítems y el porcentaje de contenido gratuito por año según la empresa desarrolladora especificada. Toma como parámetro el nombre del desarrollador y retorna un diccionario con la cantidad de ítems y el porcentaje de contenido gratuito por año.

2. **userdata(user_id: str)**:
   - Esta función devuelve información sobre un usuario específico, incluyendo la cantidad de dinero gastado, el porcentaje de recomendación basado en las revisiones, y la cantidad de ítems adquiridos. Toma como parámetro el ID del usuario y retorna un diccionario con los datos correspondientes.

3. **UserForGenre(genero: str)**:
   - Esta función devuelve el usuario que acumula más horas jugadas para un género de juegos dado, junto con una lista de la acumulación de horas jugadas por año de lanzamiento para ese género. Toma como parámetro el género de juego y retorna un diccionario con el usuario con más horas jugadas y la acumulación de horas por año.

4. **best_developer_year(anio: int)**:
   - Esta función devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. Toma como parámetro el año y retorna un diccionario con los desarrolladores más recomendados.

5. **developer_reviews_analysis(desarrolladora: str)**:
   - Esta función analiza las reseñas de los usuarios asociadas a un desarrollador específico y devuelve la cantidad total de registros de reseñas categorizados con análisis de sentimiento positivo o negativo. Toma como parámetro el nombre del desarrollador y retorna un diccionario con los resultados del análisis.

## Modelos de Recomendación

#### a) k-Nearest Neighbors (k-NN)

6. **get_recommendations_knn(game_id: int)**:
   - Esta función devuelve una lista de juegos recomendados similares al juego especificado utilizando el algoritmo k-Nearest Neighbors. Toma como parámetro el ID del juego y retorna un diccionario con los juegos recomendados.

7. **recomendacion_juego_knn(id_game: int)**:
   - Esta ruta de la API devuelve recomendaciones de juegos similares al juego especificado utilizando el modelo k-NN. Toma como parámetro el ID del juego y retorna un diccionario con los juegos recomendados.

#### b) Similitud de Cosenos

8. **get_recommendations_sim_cos(game_id: int)**:
   - Esta función devuelve una lista de juegos recomendados similares al juego especificado utilizando la similitud de cosenos. Toma como parámetro el ID del juego y retorna un diccionario con los juegos recomendados.

9. **recomendacion_juego_sim_cos(id_game: int)**:
   - Esta ruta de la API devuelve recomendaciones de juegos similares al juego especificado utilizando la similitud de cosenos. Toma como parámetro el ID del juego y retorna un diccionario con los juegos recomendados.
