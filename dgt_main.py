import pandas as pd


def cruces_read(path):
    df = pd.read_csv(path, sep=";", encoding="iso-8859-1")
    return df


def clean_names(df):
    columnas_df = df.columns

    for columna in columnas_df:
        if df[columna].dtype == "object":
            df[columna] = df[columna].str.strip()

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
        df[columna].str.normalize("NFKD")

    """
    "NFKD" (Normalización de Compatibilidad de Composición Canónica con Descomposición). Esta funcion maneja 
    caracteres Unicode que tienen representaciones equivalentes y a elimina posibles errores de codificación, 
    como los pedidos en esta pregunta. Esta función no elimina los acentos ya que al leer los datos con el
    formato encoding='iso-8859-1' acepta los acentos, que como se pueden observar en el dataset muestran un error.
    """
    """
    De todos modos también he creado un codigo (mirar siguiente codigo) donde utilizando REGEX he eliminado los 
    posibles errores de codificación generados por los acentos, como se puede observar en el csv (dataset) ya que 
    la funcion anterior (NFKD) no eliminaba los acentos. Este codigo si que proporciona lo que seria los datos 
    habiendo eliminado lo que se supone que eran los errores de codificacion (los acentos como se refleja en el csv (dataset). 
    """
    import re

    pattern = re.compile(r"\w*([ÁÉÍÓÚ]+)\w*")

    copy_1 = False

    for columna in columnas:
        new_df = []
        for _, row in df.iterrows():
            calle = row[columna]
            no_accent = True

            # se eliminan los errores de la fila para la esa columna, iterando por los matches con REGEX.
            for match in re.finditer(pattern, calle):
                no_accent = False
                copy_1 = True
                row[columna] = row[columna].replace(match.group(1), "")

            if (
                copy_1 == True
            ):  # copia la fila 'row' ya habiendo eliminado los errores (los acentos)
                new_df.append(row[columna])
            copy_1 = False

            if no_accent == True:
                new_df.append(
                    row[columna]
                )  # copia la fila como estaba, al no tener que cambiar nada (no acentos)

        df[columna] = new_df

    return df


def cruces_as_int(df):
    columnas = [
        "Codigo de vía tratado",
        "Codigo de via que cruza o enlaza",
        "Coordenada X (Guia Urbana) cm (cruce)",
        "Coordenada Y (Guia Urbana) cm (cruce)",
    ]

    for columna in columnas:
        if df[columna].dtype != "int64":
            import re

            pattern = re.compile(r"(\d+)")
            transform_to_int = lambda item: int(re.search(pattern, item).group(1))
            df[columna] = df[columna].apply(transform_to_int)

    return df


def direcciones_read(path):
    df = pd.read_csv(path, sep=";", encoding="iso-8859-1")
    return df


def direcciones_as_int(df):
    columnas = [
        "Codigo de numero",
        "Codigo de via",
        "Coordenada X (Guia Urbana) cm",
        "Coordenada Y (Guia Urbana) cm",
    ]

    for columna in columnas:
        if df[columna].dtype != "int64":
            import re

            pattern = re.compile(r"(\d+)")
            transform_to_int = lambda item: int(re.search(pattern, item).group(1))
            df[columna] = df[columna].apply(transform_to_int)

    return df


def literal_split(df):
    import re

    pattern = re.compile(r"([A-Z]+\.?)(0+)?(\d+)(\s+)?([A-Z]+)?")
    new_column_1 = []
    new_column_2 = []
    new_column_3 = []

    for _, row in df.iterrows():
        content = row["Literal de numeracion"]
        for match in re.finditer(pattern, content):
            new_column_1.append(match.group(1))
            new_column_2.append(match.group(3))
            new_column_3.append(match.group(5))

    df["Prefijo de numeración"] = new_column_1
    df["Número"] = new_column_2
    df["Sufijo de numeración"] = new_column_3

    return df


def process_data(path1, path2):
    dfcruces = cruces_read(path1)
    dfdirecciones = direcciones_read(path2)

    dfcruces_1 = clean_names(dfcruces)
    dfcruces_processed = cruces_as_int(dfcruces_1)

    dfdirecciones_1 = direcciones_as_int(dfdirecciones)
    dfdirecciones_processed = literal_split(dfdirecciones_1)

    return dfcruces_processed, dfdirecciones_processed


if __name__ == "__main__":
    path_cruces = 'datasets/CALLEJERO_VIGENTE_CRUCES_202310.csv'
    path_direcciones = 'datasets/CALLEJERO_VIGENTE_NUMERACIONES_202310.csv'
    df_cruces, df_direcciones = process_data(path_cruces, path_direcciones)
    df_cruces.to_csv("CRUCES2.csv", index=False, sep=";", encoding="iso-8859-1")
    df_direcciones.to_csv("DIRECCIONES2.csv", index=False, sep=";", encoding="iso-8859-1")