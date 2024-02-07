from fastapi import HTTPException,FastAPI
import pandas as pd
from fastapi.responses import HTMLResponse, JSONResponse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import json
import ast
from sklearn.neighbors import NearestNeighbors
import json
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI()

# Cargamos nuestros datasets:
df_items_items = pd.read_csv("./datasets/df_items_items.csv")
df_reviews_reviews = pd.read_csv("./datasets/df_reviews_reviews.csv")
df_games = pd.read_csv("./datasets/df_games.csv")
df_items = pd.read_csv("./datasets/df_items.csv")
df_reviews = pd.read_csv("./datasets/df_reviews.csv")
df_games["release_date"] = pd.to_datetime(df_games["release_date"], errors="coerce")
df_reviews_reviews["posted"] = pd.to_datetime(df_reviews_reviews["posted"], errors="coerce")
  
app = FastAPI()

# Crear punto de entreada o endpoint:
@app.get("/", tags=["Bienvenida"]) 
def mensaje():
    content = "<h2> Bienvenido al PI_MLOps_Engineer con <a href='http://127.0.0.1:8000/docs' > FastAPI </a> </h2>"
    return HTMLResponse(content=content)

"""
1 - def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
"""
# Ejemplo: Ubisoft

def developer(developer):
    # Filtrar el DataFrame para obtener solo los juegos del desarrollador especificado
    df_desarrollador = df_games[df_games["developer"] == developer].copy()

    # Convertir la columna "release_date" a tipo datetime
    df_desarrollador["release_date"] = pd.to_datetime(df_desarrollador["release_date"])

    # Extraer el año de la columna "release_date" y crear una nueva columna "year"
    df_desarrollador["year"] = df_desarrollador["release_date"].dt.year

    # Contar la cantidad de items por año
    items_por_anio = df_desarrollador.groupby("year").size().reset_index(name="Cantidad de Items")

    # Contar la cantidad de items gratuitos por año (precio igual a 0)
    items_gratuitos_por_anio = df_desarrollador[df_desarrollador["price"] == 0].groupby("year").size().reset_index(name="Contenido Free")

    # Fusionar los DataFrames por año
    resultado = pd.merge(items_por_anio, items_gratuitos_por_anio, on="year", how="left").fillna(0)

    # Calcular el porcentaje de contenido gratuito por año y redondear a 2 dígitos
    resultado["Porcentaje Contenido Free"] = round((resultado["Contenido Free"] / resultado["Cantidad de Items"]) * 100, 2)


    # Formatear los resultados en un diccionario
    resultado_dict = resultado.to_dict(orient="records")

    return resultado_dict

@app.get("/best_developer_year/{developer}", tags=["Funciones"])
async def best_developer_year(developer: str):
    try:
        # Llamar a la función "developer" para obtener el resultado
        resultado = developer(developer)
        # Devolver el resultado como un diccionario Python, FastAPI lo convertirá a JSON automáticamente
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    


"""
2 - def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
"""


# Ejemplo: js41637

@app.get("/userdata/{user_id}", tags=["Funciones"])
def userdata(user_id: str):
    try:
        # Fusionar los DataFrames para obtener la información necesaria
        merged_data = pd.merge(df_reviews_reviews[df_reviews_reviews["user_id"] == user_id], df_games, left_on="item_id", right_on="id", how="inner")

        # Calcular la cantidad de dinero gastado por el usuario
        dinero_gastado = merged_data["price"].sum()

        # Calcular el porcentaje de recomendación en base a reviews.recommend
        porcentaje_recomendacion = (merged_data["recommend"].sum() / len(merged_data)) * 100

        # Obtener la cantidad de items
        cantidad_items = len(merged_data)

        # Crear el diccionario de resultados
        resultado_dict = {
            "Usuario": user_id,
            "Dinero gastado": f"{round(dinero_gastado, 2)} USD",
            "% de recomendación": f"{porcentaje_recomendacion:.2f}%",
            "Cantidad de items": cantidad_items
        }

        # Devolver el resultado como JSONResponse
        return JSONResponse(content=resultado_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
"""
3 - def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
"""

# Ejemplo: Action

# Define la función buscar_genero_en_string
def buscar_genero_en_string(cadena, genero_buscado):
    try:
        # Convertir la cadena a una lista usando ast.literal_eval
        lista_generos = ast.literal_eval(cadena)
        
        # Buscar el género en la lista
        for elemento in lista_generos:
            if genero_buscado.lower() == elemento.lower():
                return True
        return False
    except (SyntaxError, ValueError):
        # En caso de que la cadena no sea un formato de lista válido
        return False

# Definir la función UserForGenre utilizando buscar_genero_en_string
@app.get("/UserForGenre/{genero}", tags=["Funciones"])
def UserForGenre(genero: str):
    try:
        # Filtrar los juegos que tienen el género buscado en la cadena de géneros
        df_filtered_games = df_games[df_games["genres"].apply(lambda x: buscar_genero_en_string(x, genero))]

        # Fusionar los DataFrames para obtener la información necesaria
        merged_data = pd.merge(df_filtered_games, df_items_items, left_on="id", right_on="item_id", how="inner")

        if merged_data.empty:
            raise HTTPException(status_code=404, detail="No se encontraron datos para el género especificado")

        # Encontrar al usuario que acumula más horas jugadas
        usuario_mas_horas = merged_data.loc[merged_data["playtime_forever"].idxmax()]["user_id"]

        # Calcular la acumulación de horas jugadas por año de lanzamiento
        merged_data["release_date"] = pd.to_datetime(merged_data["release_date"])
        merged_data["year"] = merged_data["release_date"].dt.year
        acumulacion_por_anio = merged_data.groupby("year")["playtime_forever"].sum().reset_index(name="Horas Acumuladas")

        # Crear el diccionario de resultados
        resultado_dict = {
            "Usuario con más horas jugadas para el Género": usuario_mas_horas,
            "Horas jugadas": acumulacion_por_anio.replace(np.nan, None).to_dict(orient="records")
        }

        # Devolver el resultado como JSONResponse
        
        return resultado_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
"""
4 - def best_developer_year( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos)
"""
# Ejemplo: 2015

@app.get("/best_developer_year/{anio}", tags=["Funciones"])
async def get_best_developer_year(anio: int):
    try:
        # Llamar a la función "best_developer_year" para obtener el resultado
        resultado = best_developer_year(df_reviews_reviews, df_games, anio)
        # Devolver el resultado como JSONResponse
        return resultado
    except HTTPException as e:
        # En caso de que ocurra un error HTTP, relanzarlo
        raise e
    except Exception as e:
        # En caso de que ocurra otro tipo de error, devolver un error HTTP interno
        raise HTTPException(status_code=500, detail=str(e))


"""
5 - def developer_reviews_analysis( desarrolladora : str ): Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
"""
# Ejemplo: Valve

def developer_reviews_analysis(df_reviews_reviews, df_games, desarrolladora):
    # Filtrar las reseñas asociadas al desarrollador específico
    reviews_desarrolladora = df_games[df_games["developer"].str.lower() == desarrolladora.lower()].merge(
        df_reviews_reviews, left_on="id", right_on="item_id", how="inner"
    )

    # Contar la cantidad total de registros de reseñas con análisis de sentimiento positivo y negativo
    count_positives = reviews_desarrolladora[reviews_desarrolladora["sentiment_analysis"] == 2].shape[0]
    count_negatives = reviews_desarrolladora[reviews_desarrolladora["sentiment_analysis"] == 0].shape[0]

    # Crear el diccionario de retorno
    resultado = {desarrolladora: {"Positive": count_positives, "Negative": count_negatives}}

    return resultado

# Definir la ruta y método para analizar las reseñas de un desarrollador
@app.get("/developer_reviews_analysis/{desarrolladora}", tags=["Funciones"])
async def get_developer_reviews_analysis(desarrolladora: str):
    try:
        # Llamar a la función "developer_reviews_analysis" para obtener el resultado
        resultado = developer_reviews_analysis(df_reviews_reviews, df_games, desarrolladora)
        # Devolver el resultado como JSONResponse
        return resultado
    except HTTPException as e:
        # En caso de que ocurra un error HTTP, relanzarlo
        raise e
    except Exception as e:
        # En caso de que ocurra otro tipo de error, devolver un error HTTP interno
        raise HTTPException(status_code=500, detail=str(e))

"""
###                                 - - - # Modelos de ML para Sistema de recomendación - - - 
"""
# La funcionalidad de las funciones esta comprobada en el archivo ETL-EDA.ipynb
"""
## a - k-Nearest Neighbors (k-NN):
"""
## - El algoritmo k-NN se basa en la idea de que objetos similares tienden a estar cerca unos de otros en un espacio de características. Funciona de la siguiente manera:

# Ejemplo: 774277


# Función para obtener recomendaciones
def get_recommendations_knn(game_id):
    # Tratar los valores faltantes en las columnas "genres" y "tags" con "Desconocido"
    df_games["genres"].fillna("Desconocido", inplace=True)
    df_games["tags"].fillna("Desconocido", inplace=True)

    # Asegurarse de que las columnas "genres" y "tags" sean de tipo string
    df_games["genres"] = df_games["genres"].astype(str)
    df_games["tags"] = df_games["tags"].astype(str)

    # Realizar la eliminación de comillas simples
    df_games["genres"] = df_games["genres"].str.replace(""", "")
    df_games["tags"] = df_games["tags"].str.replace(""", "")

    # Convertir las columnas "genres" y "tags" a variables categóricas
    label_encoder = LabelEncoder()
    df_games["genres"] = label_encoder.fit_transform(df_games["genres"])
    df_games["tags"] = label_encoder.fit_transform(df_games["tags"])

    # Entrenar el modelo KNN
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(df_games[["genres", "tags"]])
    
    # Encontrar el juego más similar utilizando el modelo KNN
    game = df_games.loc[df_games["id"] == game_id, ["genres", "tags"]].values.reshape(1, -1)
    distances, indices = knn.kneighbors(game)

    # Obtener los nombres de las aplicaciones recomendadas
    recommended_app_names = df_games.iloc[indices[0]]["app_name"].tolist()

    # Crear un JSON con los nombres de las aplicaciones recomendadas
    recommendations_json = {
        "games": recommended_app_names
    }

    return recommendations_json

@app.get("/recomendacion_juego_knn/{id_game}", tags=["Modelos de Recomendación Item-Item"])
def get_recomendacion_juego_knn(game_id:int):
    recommendations = get_recommendations_knn(game_id)
    return recommendations

"""
## b - Similitud de Cosenos
"""
## - El modelo utiliza la similitud de cosenos para encontrar juegos que son similares a un juego específico en función de sus características, como géneros y etiquetas. Este enfoque permite construir un sistema de recomendación basado en la relación ítem-ítem.

import json

def get_recommendations_sim_cos_internal(game_id):

    # Asegurarse de que las columnas "genres" y "tags" sean de tipo string
    df_games["genres"] = df_games["genres"].astype(str)
    df_games["tags"] = df_games["tags"].astype(str)

    # Realizar la eliminación de comillas simples
    df_games["genres"] = df_games["genres"].str.replace(""", "")
    df_games["tags"] = df_games["tags"].str.replace(""", "")

    # Convertir las columnas "genres" y "tags" a variables categóricas
    label_encoder = LabelEncoder()
    df_games["genres"] = label_encoder.fit_transform(df_games["genres"])
    df_games["tags"] = label_encoder.fit_transform(df_games["tags"])

    # Calcular la similitud de cosenos entre los juegos
    similarity_matrix = cosine_similarity(df_games[["genres", "tags"]])

    # Encontrar el índice del juego dado
    game_index = df_games[df_games["id"] == game_id].index[0]

    # Obtener las similitudes del juego dado con otros juegos
    game_similarities = similarity_matrix[game_index]

    # Obtener los índices de los juegos más similares
    similar_game_indices = game_similarities.argsort()[:-6:-1]

    # Obtener los nombres de las aplicaciones recomendadas
    recommended_app_names = df_games.iloc[similar_game_indices]["app_name"].tolist()

    # Crear un diccionario con los nombres de las aplicaciones recomendadas
    recommendations_dict = {"games": recommended_app_names}

    return recommendations_dict

# Ruta para la recomendación de usuario
@app.get("/recomendacion_juego_sim_cos/{id_game}", tags=["Modelos de Recomendación Item-Item"])
def get_recommendations_sim_cos(game_id:int):
    recommendations_dict = get_recommendations_sim_cos_internal(game_id)
    return recommendations_dict
