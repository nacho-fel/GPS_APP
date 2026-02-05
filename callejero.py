from grafo import *
import math
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from scipy.spatial import cKDTree

"""
callejero.py

Integrantes:
    - Ulises Díez Santaolalla
    - Ignacio Felices Vera

Descripción:
Librería con herramientas y clases auxiliares necesarias para la representación de un callejero en un grafo.

"""


# Constantes con las velocidades máximas establecidas por el enunciado para cada tipo de vía.
VELOCIDADES_CALLES = {
    "AUTOVIA": 100,
    "AVENIDA": 90,
    "CARRETERA": 70,
    "CALLEJON": 30,
    "CAMINO": 30,
    "ESTACION DE METRO": 20,
    "PASADIZO": 20,
    "PLAZUELA": 20,
    "COLONIA": 20,
}
VELOCIDAD_CALLES_ESTANDAR = 50


def datasets():
    """
    Función para cargar y procesar los datasets de cruces y direcciones.

    Returns:
        tuple: DataFrames procesados de cruces y direcciones.
    """
    path1 = "cruces.csv"
    path2 = "direcciones.csv"
    dfcruces_processed, dfdirecciones_processed = process_data(path1, path2)
    return dfcruces_processed, dfdirecciones_processed


class Cruce:
    """
    Clase que representa un cruce con coordenadas, calles que confluyen y nombre de la vía principal.
    """

    def __init__(self, coord_x, coord_y, calles, nombre_via):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.calles = calles
        self.nombre_via = nombre_via

    """
    Implementación de los métodos __eq__ y __hash__ para hacer la clase hashable.
    Dos cruces son iguales si sus coordenadas coinciden, ignorando otros atributos.
    """

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return (self.coord_x == other.coord_x) and (self.coord_y == other.coord_y)
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.coord_x, self.coord_y))


def distancia_entre_puntos(p1, p2):
    """
    Calcula la distancia euclidiana entre dos puntos.

    Args:
        p1 (tuple): Coordenadas del primer punto (x1, y1).
        p2 (tuple): Coordenadas del segundo punto (x2, y2).

    Returns:
        float: Distancia euclidiana entre p1 y p2.
    """

    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def unificar(r, dfcruces_processed):
    """
    Función para unificar coordenadas que estén dentro de un radio r, eliminando duplicados cercanos.

    Args:
        r (float): Radio máximo para considerar puntos cercanos.
        dfcruces_processed (DataFrame): DataFrame con cruces procesados.

    Returns:
        list: Lista de coordenadas unificadas.
    """
    dfcruces_processed["Coordenadas X, Y"] = dfcruces_processed.apply(
        lambda row: (
            row["Coordenada X (Guia Urbana) cm (cruce)"],
            row["Coordenada Y (Guia Urbana) cm (cruce)"],
        ),
        axis=1,
    )
    coordenadas = dfcruces_processed["Coordenadas X, Y"].unique()
    sorted_coordinates = sorted(coordenadas)
    grupos = []

    i = 0
    while i < len(sorted_coordinates) - 1:
        grupo = sorted_coordinates[i]
        j = i + 1
        nuevo_contador = False
        # Itera para eliminar puntos cercanos dentro del radio r
        while j < len(sorted_coordinates) and (j <= i + r):
            if (
                nuevo_contador == False
                and distancia_entre_puntos(sorted_coordinates[i], sorted_coordinates[j])
                > r
            ):
                i = j
                nuevo_contador = True
            elif (
                distancia_entre_puntos(sorted_coordinates[i], sorted_coordinates[j]) < r
            ):
                sorted_coordinates.remove(sorted_coordinates[j])
                j -= 1
            j += 1
        grupos.append(grupo)
    return grupos


def unificar_1(r, dfcruces_processed):
    """
    Optimización de la función unificar usando KD-Tree para búsquedas espaciales rápidas.

    Args:
        r (float): Radio máximo para considerar puntos cercanos.
        dfcruces_processed (DataFrame): DataFrame con cruces procesados.

    Returns:
        dict: Diccionario con coordenadas clave y listas de puntos cercanos.
    """
    dfcruces_processed["Coordenadas X, Y"] = dfcruces_processed.apply(
        lambda row: (
            row["Coordenada X (Guia Urbana) cm (cruce)"],
            row["Coordenada Y (Guia Urbana) cm (cruce)"],
        ),
        axis=1,
    )
    coordenadas = np.array(dfcruces_processed["Coordenadas X, Y"].unique().tolist())

    # Build KD-tree for efficient spatial searches
    tree = cKDTree(coordenadas)
    grupos = {}
    nodos = []
    for coord in coordenadas:
        # Hace una lista con los indices de la lista coordenadas cuyas cordnadas (x,y) se encuentrar dentro de una distancia 'r'
        indices = tree.query_ball_point(coord, r)
        # Eliminamos el índice del mismo punto
        indices.remove(indices[0])
        key = tuple(coord)
        if indices != []:
            grupos[key] = []
        for index in indices:
            punto = tuple(coordenadas[index])
            if distancia_entre_puntos(coord, punto) < r:
                grupos[key].append(punto)
    return grupos


def cruces(dfcruces_processed):
    """
    Obtiene una lista de objetos Cruce únicos con sus calles y nombre de vía principal.

    Args:
        dfcruces_processed (DataFrame): DataFrame con cruces procesados.

    Returns:
        list: Lista de objetos Cruce únicos.
    """
    dfcruces_processed["Coordenadas X, Y"] = dfcruces_processed.apply(
        lambda row: (
            row["Coordenada X (Guia Urbana) cm (cruce)"],
            row["Coordenada Y (Guia Urbana) cm (cruce)"],
        ),
        axis=1,
    )
    cruces_unicos = dfcruces_processed["Coordenadas X, Y"].unique()

    cruces_maps = []
    for cruce in cruces_unicos:
        filtered_rows = dfcruces_processed[
            dfcruces_processed["Coordenadas X, Y"] == cruce
        ]
        lista_calles = filtered_rows[
            ["Codigo de via que cruza o enlaza", "Codigo de vía tratado"]
        ].values.flatten()
        calles = list(set(lista_calles))
        nombre_via = filtered_rows["Nombre de la via tratado"].iloc[0]
        cruce = Cruce(cruce[0], cruce[1], calles, nombre_via)
        cruces_maps.append(cruce)

    return cruces_maps


def cruces_TODOS(dfcruces_processed):
    """
    Obtiene todos los cruces de Madrid, incluyendo cada nombre de vía para cada cruce.

    Args:
        dfcruces_processed (DataFrame): DataFrame con cruces procesados.

    Returns:
        list: Lista de objetos Cruce, con todos los nombres de vías.
    """
    dfcruces_processed["Coordenadas X, Y"] = dfcruces_processed.apply(
        lambda row: (
            row["Coordenada X (Guia Urbana) cm (cruce)"],
            row["Coordenada Y (Guia Urbana) cm (cruce)"],
        ),
        axis=1,
    )
    cruces_unicos = dfcruces_processed["Coordenadas X, Y"].unique()

    cruces_maps = []
    for cruce in cruces_unicos:
        filtered_rows = dfcruces_processed[
            dfcruces_processed["Coordenadas X, Y"] == cruce
        ]
        lista_calles = filtered_rows[
            ["Codigo de via que cruza o enlaza", "Codigo de vía tratado"]
        ].values.flatten()
        calles = list(set(lista_calles))
        for _, row in filtered_rows.iterrows():
            nombre_via = row["Nombre de la via tratado"]
            cruce_objeto = Cruce(cruce[0], cruce[1], calles, nombre_via)
            cruces_maps.append(cruce_objeto)

    return cruces_maps


class Calle:
    """
    Clase para representar una calle, con código, cruces, números, tipo de vía y nombre.
    """

    def __init__(
        self,
        codigo_calle,
        cruces,
        numero,
        diccionario_numeros,
        tipo_de_via,
        nombre_calle,
    ) -> None:
        self.codigo_calle = codigo_calle
        self.cruces = cruces
        self.numero = numero
        self.diccionario_numeros = diccionario_numeros
        self.tipo_de_via = tipo_de_via
        self.nombre_calle = nombre_calle


def calles(dfdirecciones_processed, dfcruces_processed):
    """
    Obtiene una lista de objetos Calle a partir de los datos procesados.

    Args:
        dfdirecciones_processed (DataFrame): DataFrame con direcciones procesadas.
        dfcruces_processed (DataFrame): DataFrame con cruces procesados.

    Returns:
        list: Lista de objetos Calle.
    """
    calles_maps = []

    lista_1 = list(dfcruces_processed["Codigo de vía tratado"])
    lista_2 = list(dfcruces_processed["Codigo de via que cruza o enlaza"])
    lista_juntas = lista_1 + lista_2

    codigos_unicos = list(set(lista_juntas))

    for codigo_calle in codigos_unicos:
        filtered_rows = dfcruces_processed[
            dfcruces_processed["Codigo de vía tratado"] == codigo_calle
        ]
        cruces = []
        for _, row in filtered_rows.iterrows():
            cruce = (
                row["Coordenada X (Guia Urbana) cm (cruce)"],
                row["Coordenada Y (Guia Urbana) cm (cruce)"],
            )
            cruces.append(cruce)

        filtered_rows_1 = dfdirecciones_processed[
            dfdirecciones_processed["Codigo de via"] == codigo_calle
        ]
        numeros = []
        diccionario = {}
        for _, row in filtered_rows_1.iterrows():
            numero = row["Literal de numeracion"]
            numeros.append(numero)
            tipo_de_via = row["Clase de la via"]
            nombre_calle = row["Nombre de la vía"]
            diccionario[numero] = (
                row["Coordenada X (Guia Urbana) cm"],
                row["Coordenada Y (Guia Urbana) cm"],
            )

        calle = Calle(
            codigo_calle,
            list(set(cruces)),
            numeros,
            diccionario,
            tipo_de_via,
            nombre_calle,
        )
        calles_maps.append(calle)

    return calles_maps


def agregar_vertices(grafo, lista_cruces):
    """
    Agrega vértices al grafo basándose en la lista de cruces.

    Args:
        grafo (Grafo): Grafo donde se agregarán los vértices.
        lista_cruces (list): Lista de objetos Cruce.

    Returns:
        dict: Diccionario que mapea coordenadas a sí mismas (para posición en graficación).
    """
    dicc = {}
    for cruce in lista_cruces:
        grafo.agregar_vertice((cruce.coord_x, cruce.coord_y))
        dicc[(cruce.coord_x, cruce.coord_y)] = (cruce.coord_x, cruce.coord_y)

    return dicc


def agregar_aristas_peso_dist_euc(grafo, lista_calles):
    """
    Agrega aristas al grafo entre cruces de cada calle, ponderadas por la distancia euclidiana.

    Args:
        grafo (Grafo): Grafo donde se agregarán las aristas.
        lista_calles (list): Lista de objetos Calle.
    """
    for calle in lista_calles:
        dict_numeros_calle = calle.diccionario_numeros
        claves = list(dict_numeros_calle.keys())
        if len(dict_numeros_calle) != 0:
            clave_min = min(claves)
            coordenadas_min = dict_numeros_calle[clave_min]
            coordenadas_ref = (coordenadas_min[0], coordenadas_min[1])
            cruces_ordenados = sorted(
                calle.cruces,
                key=lambda cruce: distancia_entre_puntos(
                    coordenadas_ref, (cruce[0], cruce[1])
                ),
            )

            for i in range(len(cruces_ordenados) - 1):
                ini = cruces_ordenados[i]
                fin = cruces_ordenados[i + 1]
                distancia = distancia_entre_puntos(ini, fin)
                datos = {
                    "codigo": calle.codigo_calle,
                    "vertice1": ini,
                    "vertice2": fin,
                    "distancia": distancia,
                }
                grafo.agregar_arista(ini, fin, datos, distancia)

        else:
            if len(calle.cruces) > 1:
                coordenadas_min = calle.cruces[0]
                coordenadas_ref = (coordenadas_min[0], coordenadas_min[1])
                cruces_ordenados = sorted(
                    calle.cruces,
                    key=lambda cruce: distancia_entre_puntos(
                        coordenadas_ref, (cruce[0], cruce[1])
                    ),
                )
                for i in range(len(cruces_ordenados) - 1):
                    ini = cruces_ordenados[i]
                    fin = cruces_ordenados[i + 1]
                    distancia = distancia_entre_puntos(ini, fin)
                    datos = {
                        "codigo": calle.codigo_calle,
                        "vertice1": ini,
                        "vertice2": fin,
                        "distancia": distancia,
                    }
                    grafo.agregar_arista(ini, fin, datos, distancia)
            else:
                pass


# 1. Encontrar el tipo de via
# 2. Aplicar la vel max
# 3. tiempo = distancia / velocidad


def calcular_tiempo(calle, distancia):
    """
    Calcula el tiempo de recorrido para una calle dada una distancia, considerando la velocidad máxima.

    Args:
        calle (Calle): Objeto Calle.
        distancia (float): Distancia entre dos cruces.

    Returns:
        float: Tiempo estimado de recorrido en minutos.
    """
    clase_via = calle.tipo_de_via

    if clase_via == "AUTOVIA":
        vel_max = 100
    elif clase_via == "AVENIDA":
        vel_max = 90
    elif clase_via == "CARRETERA":
        vel_max = 70
    elif clase_via == "CALLEJON" or clase_via == "CAMINO":
        vel_max = 30
    elif (
        clase_via == "PASADIZO"
        or clase_via == "PLAZUELA"
        or clase_via == "COLONIA"
        or clase_via == "ESTACION DE METRO"
    ):
        vel_max = 20
    else:
        vel_max = 50
    tiempo = distancia / vel_max

    return tiempo


def agregar_aristas_peso_vel_max(grafo, lista_calles):
    """
    Agrega aristas al grafo ponderadas por el tiempo estimado de recorrido usando velocidad máxima.

    Args:
        grafo (Grafo): Grafo donde se agregarán las aristas.
        lista_calles (list): Lista de objetos Calle.
    """
    for calle in lista_calles:
        dict_numeros_calle = calle.diccionario_numeros
        claves = list(dict_numeros_calle.keys())
        if len(dict_numeros_calle) != 0:
            clave_min = min(claves)
            coordenadas_min = dict_numeros_calle[clave_min]
            coordenadas_ref = (coordenadas_min[0], coordenadas_min[1])
            cruces_ordenados = sorted(
                calle.cruces,
                key=lambda cruce: distancia_entre_puntos(
                    coordenadas_ref, (cruce[0], cruce[1])
                ),
            )

            for i in range(len(cruces_ordenados) - 1):
                ini = cruces_ordenados[i]
                fin = cruces_ordenados[i + 1]
                distancia = distancia_entre_puntos(ini, fin)
                tiempo = calcular_tiempo(calle, distancia)
                datos = {
                    "codigo": calle.codigo_calle,
                    "vertice1": ini,
                    "vertice2": fin,
                    "distancia": distancia,
                }
                grafo.agregar_arista(ini, fin, datos, tiempo)

        else:
            if len(calle.cruces) > 1:
                coordenadas_min = calle.cruces[0]
                coordenadas_ref = (coordenadas_min[0], coordenadas_min[1])
                cruces_ordenados = sorted(
                    calle.cruces,
                    key=lambda cruce: distancia_entre_puntos(
                        coordenadas_ref, (cruce[0], cruce[1])
                    ),
                )

                for i in range(len(cruces_ordenados) - 1):
                    ini = cruces_ordenados[i]
                    fin = cruces_ordenados[i + 1]
                    distancia = distancia_entre_puntos(ini, fin)
                    tiempo = calcular_tiempo(calle, distancia)
                    datos = {
                        "codigo": calle.codigo_calle,
                        "vertice1": ini,
                        "vertice2": fin,
                        "distancia": distancia,
                    }
                    grafo.agregar_arista(ini, fin, datos, tiempo)
            else:
                pass


def generar_Madrid_1(lista_cruces, lista_calles):
    """
    Genera un grafo con pesos basados en la distancia euclidiana y lo dibuja.

    Args:
        lista_cruces (list): Lista de objetos Cruce.
        lista_calles (list): Lista de objetos Calle.

    Returns:
        Grafo: Grafo generado con pesos de distancia euclidiana.
    """

    grafo = Grafo()
    dicc = agregar_vertices(grafo, lista_cruces)
    agregar_aristas_peso_dist_euc(grafo, lista_calles)

    grafo_nx = grafo.convertir_a_NetworkX()

    plt.figure(figsize=(50, 50))
    plot = plt.plot()
    nx.draw(grafo_nx, pos=dicc, with_labels=False, node_size=0.1)
    nx.draw_networkx_edges(grafo_nx, pos=dicc, edge_color="b", width=0.4)
    plt.show()

    return grafo


def generar_Madrid_2(lista_cruces, lista_calles):
    """
    Genera un grafo con pesos basados en el tiempo estimado considerando la velocidad máxima
    y lo dibuja.

    Args:
        lista_cruces (list): Lista de objetos Cruce.
        lista_calles (list): Lista de objetos Calle.

    Returns:
        Grafo: Grafo generado con pesos de tiempo estimado.
    """

    grafo = Grafo()
    dicc = agregar_vertices(grafo, lista_cruces)
    agregar_aristas_peso_vel_max(grafo, lista_calles)

    grafo_nx = grafo.convertir_a_NetworkX()

    plt.figure(figsize=(50, 50))
    plot = plt.plot()
    nx.draw(grafo_nx, pos=dicc, with_labels=False, node_size=0.1)
    nx.draw_networkx_edges(grafo_nx, pos=dicc, edge_color="b", width=0.4)
    plt.show()

    return grafo


"""
Funciones auxiliares para la lectura y procesamiento de los datos originales DGT,
no forman parte del módulo callejero.py pero son útiles para preparar los datasets.
"""


def cruces_read(path_cruces: str):
    """
    Lee el archivo CSV de cruces y devuelve un DataFrame.

    Args:
        path_cruces (str): Ruta al archivo CSV de cruces.

    Returns:
        DataFrame: Datos de cruces cargados.
    """
    df_cruces = pd.read_csv(path_cruces, encoding="iso-8859-1", sep=";")
    return df_cruces


def clean_names(df_cruces):
    """
    Limpia y normaliza los nombres y clases de vías en el DataFrame de cruces.

    Args:
        df_cruces (DataFrame): DataFrame con datos de cruces.

    Returns:
        DataFrame: DataFrame con nombres y clases limpiados.
    """
    columnas = [
        "Literal completo del vial tratado",
        "Literal completo del vial que cruza",
        "Clase de la via tratado",
        "Clase de la via que cruza",
        "Particula de la via tratado",
        "Particula de la via que cruza",
        "Nombre de la via tratado",
        "Nombre de la via que cruza",
    ]
    for columna in columnas:
        df_cruces[columna] = df_cruces[columna].str.strip()
        df_cruces[columna] = df_cruces[columna].str.normalize("NFKD")
    return df_cruces


def cruces_as_int(df_cruces):
    """
    Convierte a enteros las columnas numéricas de cruces relevantes.

    Args:
        df_cruces (DataFrame): DataFrame de cruces.

    Returns:
        DataFrame: DataFrame con columnas convertidas a enteros.
    """
    columnas = [
        "Codigo de vía tratado",
        "Codigo de via que cruza o enlaza",
        "Coordenada X (Guia Urbana) cm (cruce)",
        "Coordenada Y (Guia Urbana) cm (cruce)",
    ]
    # Aplico una función lambda para convertir a numero
    funcion_correccion = lambda x: x if isinstance(x, int) else int(x)
    # Aplico la función de conversión a cada columna
    for columna in columnas:
        df_cruces[columna] = df_cruces[columna].apply(funcion_correccion)
    return df_cruces


def direcciones_read(path_direcciones: str):
    """
    Lee el archivo CSV de direcciones y devuelve un DataFrame.

    Args:
        path_direcciones (str): Ruta al archivo CSV de direcciones.

    Returns:
        DataFrame: Datos de direcciones cargados.
    """
    df_direcciones = pd.read_csv(path_direcciones, encoding="iso-8859-1", sep=";")
    return df_direcciones


def direcciones_as_int(df_direcciones):
    """
    Convierte a enteros las columnas numéricas de direcciones relevantes.

    Args:
        df_direcciones (DataFrame): DataFrame de direcciones.

    Returns:
        DataFrame: DataFrame con columnas convertidas a enteros.
    """
    columnas = [
        "Codigo de numero",
        "Codigo de via",
        "Coordenada X (Guia Urbana) cm",
        "Coordenada Y (Guia Urbana) cm",
    ]
    # Aplico una función lambda para convertir a numero y uso Regex por si contiene carácteres no numéricos
    funcion_correccion = (
        lambda x: x if isinstance(x, int) else int(float(re.sub(r"[^0-9.]", "", x)))
    )
    # Aplico la función de conversión a cada columna
    for columna in columnas:
        df_direcciones[columna] = df_direcciones[columna].apply(funcion_correccion)
    return df_direcciones


def literal_split(df_direcciones):
    """
    Separa la columna 'Literal de numeracion' en prefijo, número y sufijo.

    Args:
        df_direcciones (DataFrame): DataFrame de direcciones.

    Returns:
        DataFrame: DataFrame con columnas separadas para literal de numeración.
    """
    df_direcciones_split = df_direcciones.copy()
    # Usando Regex obtengo los tres grupos al separar la cadena de caracteres
    # El primer grupo son letras y puede contener puntos
    # El segundo grupo son números
    # El tercer grupo son letras y puede estar o no
    regex_pattern = r"(^[A-Za-z.]+)([0-9]+)([A-Za-z]+)?"
    # Creo una nueva columna con las coincidencias al aplicar regex del primer grupo
    df_direcciones_split["Prefijo de numeración"] = (
        (df_direcciones_split["Literal de numeracion"].str.findall(regex_pattern))
        .str[0]
        .str[0]
    )
    # Creo una nueva columna con las coincidencias al aplicar regex del segundo grupo
    df_direcciones_split["Número"] = (
        (df_direcciones_split["Literal de numeracion"].str.findall(regex_pattern))
        .str[0]
        .str[1]
    )
    # Creo una nueva columna con las coincidencias al aplicar regex del tercer grupo
    df_direcciones_split["Sufijo de numeración"] = (
        (df_direcciones_split["Literal de numeracion"].str.findall(regex_pattern))
        .str[0]
        .str[2]
    )
    # Convierto la columna de Numeros en int
    funcion_correccion = lambda x: x if isinstance(x, int) else int(x)
    # Aplico la función de conversión a cada columna
    df_direcciones_split["Número"] = df_direcciones_split["Número"].apply(
        funcion_correccion
    )
    # Devuelvo el DataFrame
    return df_direcciones_split


def process_data(path_cruces: str, path_direcciones: str):
    """
    Procesa los archivos de cruces y direcciones y devuelve DataFrames listos para usar.

    Args:
        path_cruces (str): Ruta al CSV de cruces.
        path_direcciones (str): Ruta al CSV de direcciones.

    Returns:
        tuple: DataFrames procesados de cruces y direcciones.
    """
    df_cruces = cruces_read(path_cruces)
    # Limpio los errores del DataFrame de los cruces
    df_cruces_limpio = clean_names(df_cruces)
    # Cambio a numeros las columnas correspondientes
    df_cruces_limpio_int = cruces_as_int(df_cruces_limpio)
    # Obtengo el DataFrame de las direcciones
    df_direcciones = direcciones_read(path_direcciones)
    # Cambio a numeros las columnas correspondientes
    df_direcciones_int = direcciones_as_int(df_direcciones)
    # Creo las tres nuevas columnas en el Dataframe a partir de la columna “Literal de Numeración”
    df_direcciones_int_split = literal_split(df_direcciones_int)

    # Devuelvo los dos DataFrames procesados y normalizados
    return df_cruces_limpio_int, df_direcciones_int_split
