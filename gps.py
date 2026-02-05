from callejero import *
from grafo import *
import matplotlib.pyplot as plt
import pandas as pd
import math
import networkx as nx


def arrancar_gps(
    dfdirecciones_processed, dfcruces_processed, lista_calles, lista_cruces
):
    # Peso: distancia eucl´ıdea entre los nodos
    grafo_1 = generar_Madrid_1(lista_cruces, lista_calles)

    # Peso: tiempo que tarda un coche en recorrer dicha arista
    grafo_2 = generar_Madrid_2(lista_cruces, lista_calles)

    return grafo_1, grafo_2


def seleccionar_direcciones(calle, numero, df_direcciones):
    # Convertimos el nombre de la calle a como esta en el dataset
    calle = (calle.upper()).strip()
    # Iteramos para encontrar el indice
    for i in range(df_direcciones.shape[0]):
        if df_direcciones["Nombre de la vía"][i].strip() == calle and df_direcciones[
            "Número"
        ][i] == int(numero):
            seleccion1 = i + 1

    # Obtenemos las siguientes series para poder acceder a ellas posteriormente
    direcciones = df_direcciones["Direccion completa para el numero"]
    codigos = df_direcciones["Codigo de via"]

    # Se elige un numero del indice del dataset de direcciones
    direccion = direcciones[seleccion1]
    codigo = codigos[seleccion1]
    coordenadas = (
        int(df_direcciones["Coordenada X (Guia Urbana) cm"][seleccion1]),
        int(df_direcciones["Coordenada Y (Guia Urbana) cm"][seleccion1]),
    )

    return coordenadas, codigo, direccion


def seleccionar_opcion_ruta(grafo_1, grafo_2):
    # Imprimimos las opciones de ruta
    print("Opciones de ruta:")
    print("1. Ruta más corta")
    print("2. Ruta más rápida")

    try:
        seleccion = int(input("Seleccione la opción de ruta (1 o 2): "))
        if seleccion == 1:
            grafo = grafo_1
            return (
                "corta",
                grafo,
            )  # Devolvemos la opcion de ruta y el grafo que se va a utilizar con el
        elif seleccion == 2:
            grafo = grafo_2
            return "rapida", grafo
        else:
            print("Opción no válida. Por favor, seleccione 1 o 2.")
            return None

    except ValueError:
        print("Entrada inválida. Por favor, ingrese un número.")
        return None


def encontrar_cruce_mas_cercano(direccion, cord_x, cord_y, lista_cruces):
    # Inicializar con una distancia grande para asegurar la actualización en la primera iteración
    distancia_minima = float("inf")
    cruce_mas_cercano = None

    df_direcciones = pd.read_csv("direcciones.csv", encoding="iso-8859-1", sep=";")
    df_cruces = pd.read_csv("cruces.csv", encoding="iso-8859-1", sep=";")

    columna = df_direcciones[
        df_direcciones["Direccion completa para el numero"] == direccion
    ]
    nombre_via = columna["Nombre de la vía"].iloc[0]

    cruces_df = df_cruces[df_cruces["Nombre de la via tratado"] == nombre_via]

    for _, cruce in cruces_df.iterrows():
        # Convertir a números y manejar NaN
        coord_x_cruce = pd.to_numeric(
            cruce["Coordenada X (Guia Urbana) cm (cruce)"], errors="coerce"
        )
        coord_y_cruce = pd.to_numeric(
            cruce["Coordenada Y (Guia Urbana) cm (cruce)"], errors="coerce"
        )

        distancia = math.sqrt(
            (coord_x_cruce - cord_x) ** 2 + (coord_y_cruce - cord_y) ** 2
        )
        if distancia < distancia_minima:
            distancia_minima = distancia
            cruce_cercano_x = coord_x_cruce
            cruce_cercano_y = coord_y_cruce

    # Una vez encontrado el cruce, cogemos sus cordeandas y buscamos el objeto cruce para ese dado
    for cruce in lista_cruces:
        if (cruce_cercano_x, cruce_cercano_y) == (cruce.coord_x, cruce.coord_y):
            cruce_mas_cercano = cruce

    return cruce_mas_cercano


def encontrar_ruta(direccion_origen, direccion_destino, tipo_ruta, grafo=Grafo()):
    # Elegimos el tipo de ruta que queremos tomar
    if tipo_ruta == "corta":
        # implementar la clase grafo
        ruta = grafo.camino_minimo(direccion_origen, direccion_destino)
    elif tipo_ruta == "rapida":
        # implementar la clase grafo
        ruta = grafo.camino_minimo(direccion_origen, direccion_destino)
    else:
        print("Tipo de ruta no reconocido.")
        return None

    return ruta


def gps_ruta(origen, destino, grafo_1, grafo_2):
    # Cargamos el dataset de direcciones
    df_direcciones = pd.read_csv("direcciones.csv", encoding="iso-8859-1", sep=";")

    # Cargamos el dataset de cruces
    df_cruces = pd.read_csv("cruces.csv", encoding="iso-8859-1", sep=";")
    lista_cruces = cruces(df_cruces)

    # Encontramos las coordendas del origen y destino
    row_origen = df_direcciones[
        df_direcciones["Direccion completa para el numero"] == origen
    ]
    coordenadas_origen = (
        float(row_origen["Coordenada X (Guia Urbana) cm"].iloc[0]),
        float(row_origen["Coordenada Y (Guia Urbana) cm"].iloc[0]),
    )

    row_destino = df_direcciones[
        df_direcciones["Direccion completa para el numero"] == destino
    ]
    coordenadas_destino = (
        float(row_destino["Coordenada X (Guia Urbana) cm"].iloc[0]),
        float(row_destino["Coordenada Y (Guia Urbana) cm"].iloc[0]),
    )

    # Encontramos el cruce mas cercano al origen
    cruce_origen = encontrar_cruce_mas_cercano(
        origen, coordenadas_origen[0], coordenadas_origen[1], lista_cruces
    )
    cordenadas_cruce_origen = (cruce_origen.coord_x, cruce_origen.coord_y)

    # Encontramos el cruce mas cercano al destino
    cruce_destino = encontrar_cruce_mas_cercano(
        destino, coordenadas_destino[0], coordenadas_destino[1], lista_cruces
    )
    cordenadas_cruce_destino = (cruce_destino.coord_x, cruce_destino.coord_y)

    # Elegimos el tipo de ruta
    tipo_ruta, grafo = seleccionar_opcion_ruta(grafo_1, grafo_2)

    # Encontramos la ruta como una lista de cordenadas a seguir
    ruta = encontrar_ruta(
        cordenadas_cruce_origen, cordenadas_cruce_destino, tipo_ruta, grafo
    )

    return ruta, grafo


def recorrido_por_calles(
    grafo, camino, coordenadas_orig, destino_coord, cod_orig, cod_destino
):
    # Lista que almacenará las aristas a recorrer
    aristas_recorrer = []

    # Iterar sobre la ruta para obtener las aristas correspondientes
    for i in range(len(camino) - 1):
        arista_actual = grafo.obtener_arista(camino[i], camino[i + 1])

        # Actualizar los vértices de la arista
        arista_actual[0]["vertice1"] = camino[i]
        arista_actual[0]["vertice2"] = camino[i + 1]
        aristas_recorrer.append(arista_actual)

    # Variables para el seguimiento de la calle actual
    calle_anterior = None
    dicc_recorrido_calles = {}
    distancias = []
    paso = 0
    calle_dist = 0

    # Primer elemento del diccionario con información del origen
    dicc_recorrido_calles[0] = (
        cod_orig,
        round(
            distancia_entre_puntos(coordenadas_orig, aristas_recorrer[0][0]["vertice1"])
            / 100
        ),
    )

    # Iterar sobre las aristas para calcular distancias y segmentar por calles
    for arista in aristas_recorrer:
        calle_actual = arista[0]["codigo"]

        if calle_actual != calle_anterior:
            if calle_anterior is not None:
                distancias.append(round(calle_dist / 100))
            paso += 1
            calle_dist = 0
            dicc_recorrido_calles[paso] = []
            dicc_recorrido_calles[paso].append(arista)
            calle_dist = calle_dist + arista[0]["distancia"]

        elif calle_actual == calle_anterior:
            dicc_recorrido_calles[paso].append(arista)
            calle_dist += arista[0]["distancia"]

        calle_anterior = calle_actual
    distancias.append(calle_dist)

    # Último elemento del diccionario con información del destino
    dicc_recorrido_calles[paso + 1] = (
        cod_destino,
        distancia_entre_puntos(destino_coord, aristas_recorrer[-1][0]["vertice2"]),
    )

    return dicc_recorrido_calles, distancias


def direccion_giro(cruce_anterior, cruce_actual, cruce_siguiente):
    # Determina la dirección del giro en una intersección
    delta_x_anterior = cruce_actual[0] - cruce_anterior[0]
    delta_y_anterior = cruce_actual[1] - cruce_anterior[1]
    delta_x_siguiente = cruce_siguiente[0] - cruce_actual[0]
    delta_y_siguiente = cruce_siguiente[1] - cruce_actual[1]

    if delta_x_anterior == 0:  # Si la recta anterior es vertical
        if delta_x_siguiente == 0:
            return "recto"
        elif delta_x_siguiente < 0:
            return "izquierda"
        else:
            return "derecha"
    else:
        # primero calculamos la pendiente
        m_anterior = delta_y_anterior / delta_x_anterior
        # despues calculamos el punto de intersección
        x = 1 / m_anterior * delta_y_siguiente + cruce_actual[0]

        # determinamos la direccion del giro
        if delta_x_siguiente > 0:
            return "izquierda" if cruce_siguiente[0] > x else "derecha"
        else:
            return "derecha" if cruce_siguiente[0] > x else "izquierda"


def obtener_nombre(codigo, df_cruces):
    # Obtiene el nombre de una calle a partir de su código
    row = df_cruces[df_cruces["Codigo de vía tratado"] == codigo]

    if not row.empty:
        nombre = row.iloc[0]["Literal completo del vial tratado"]
        return nombre
    else:
        return None


def generar_instrucciones(diccionario_recorrido_calles, distancias, df_cruces):
    giros = []
    instrucciones = []
    n_calle = len(diccionario_recorrido_calles.keys())
    valores = list(diccionario_recorrido_calles.values())
    instrucciones.append(
        (diccionario_recorrido_calles[0][0], diccionario_recorrido_calles[0][1])
    )

    # Iterar sobre las calles para generar instrucciones y giros
    for i in range(1, len(diccionario_recorrido_calles.keys()) - 1):
        calle = obtener_nombre(valores[i][0][0]["codigo"], df_cruces)
        cm = distancias[i - 1]
        instrucciones.append((calle, cm))
        if i + 1 < len(diccionario_recorrido_calles.keys()) - 1:
            punto_anterior = valores[i][-1][0]["vertice1"]
            punto_interseccion = valores[i][-1][0]["vertice2"]
            punto_siguiente = valores[i + 1][0][0]["vertice2"]
            sentido_giro = direccion_giro(
                punto_anterior, punto_interseccion, punto_siguiente
            )
            giros.append(sentido_giro)

    # Último elemento del diccionario con información del destino
    instrucciones.append(
        (
            diccionario_recorrido_calles[n_calle - 1][0],
            diccionario_recorrido_calles[n_calle - 1][1],
        )
    )

    return instrucciones, giros


def instrucciones(instrucciones, giros, df_cruces):
    # En esta funcion se crea la estructura de las instrucciones
    final = len(instrucciones) - 1
    print(
        f"Usted comienza {instrucciones[0][1]} metros por {obtener_nombre(instrucciones[0][0], df_cruces)}"
    )
    for i in range(1, len(instrucciones) - 1):
        print(
            f"Continue por la calle {instrucciones[i][0]} durante {instrucciones[i][1]} metros"
        )
        if i < len(giros):
            print(f"Gire a la {giros[i]}, para la calle {instrucciones[i+1][0]}")
    print(
        f"Continue por la calle  {obtener_nombre(instrucciones[final][0], df_cruces)} durante {instrucciones[final][1]}"
    )
    print("Usted se encuentra en su destino.")


def pintar_camino_rojo(lista_cruces, lista_calles, camino):
    camino_final = []
    # Primero pasamos el camino a una lista de nos nodos de las aristas a pintar en rojo
    for i in range(len(camino)):
        if i == 0:
            camino_final.append((camino[i], camino[i + 1]))
        elif i == (len(camino) - 1):
            camino_final.append((camino[i - 1], camino[i]))
        else:
            camino_final.append((camino[i - 1], camino[i]))
            camino_final.append((camino[i], camino[i + 1]))

    # creamos el grafo de nuevo como anteriormente
    grafo = Grafo()
    dicc = agregar_vertices(grafo, lista_cruces)
    agregar_aristas_peso_vel_max(grafo, lista_calles)
    grafo_nx = grafo.convertir_a_NetworkX()

    plt.figure(figsize=(50, 50))
    plot = plt.plot()
    nx.draw(grafo_nx, pos=dicc, with_labels=False, node_size=0.1)
    # especificamos las aristas a pintar son las del camino final
    nx.draw_networkx_edges(
        grafo_nx, pos=dicc, edgelist=camino_final, edge_color="r", width=1.5
    )
    plt.show()


# El main
if __name__ == "__main__":
    # Esto solo hace falta cargarlo en la primera pasada
    path1 = "CRUCES.csv"
    path2 = "DIRECCIONES.csv"
    df_cruces, df_direcciones = process_data(path1, path2)
    lista_calles = calles(df_direcciones, df_cruces)
    lista_cruces = cruces(df_cruces)
    grafo_1, grafo_2 = arrancar_gps(
        df_direcciones, df_cruces, lista_calles, lista_cruces
    )  # tarda sobre 2mins

    while True:
        print("DATOS DE ORIGEN")
        origen_calle = input("Calle:")
        origen_numero = input("Número:")
        coordenadas_orig, codigo_orig, direccion_orig = seleccionar_direcciones(
            origen_calle, origen_numero, df_direcciones
        )
        print(direccion_orig)

        print("DATOS DE DESTINO")
        destino_calle = input("Calle:")
        destino_numero = input("Número:")
        coordenadas_dest, codigo_dest, direccion_dest = seleccionar_direcciones(
            destino_calle, destino_numero, df_direcciones
        )
        print(direccion_dest)

        camino_minimo, grafo = gps_ruta(
            direccion_orig, direccion_dest, grafo_1, grafo_2
        )  # tarda sobre 2mins

        dicc_recorrido_calles, distancias = recorrido_por_calles(
            grafo,
            camino_minimo,
            coordenadas_orig,
            coordenadas_dest,
            codigo_orig,
            codigo_dest,
        )
        instrucciones_, giros = generar_instrucciones(
            dicc_recorrido_calles, distancias, df_cruces
        )
        instrucciones(instrucciones_, giros, df_cruces)

        pintar_camino_rojo(lista_cruces, lista_calles, camino_minimo)

        salida = input("¿Quieres realizar otra búsqueda? (Si/No): ").lower()
        if salida != "si":
            break  
