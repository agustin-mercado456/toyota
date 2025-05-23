import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde , zscore
from scipy import stats
import math
# BOXPLOTS
def boxplot(feature,title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(feature)
    plt.title(f'Boxplot de {title}')
    plt.xlabel(title)
    plt.show()


# HISTOGRAMA Y DENSIDAD

def histogram(feature,title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(feature, bins=30, kde=True)
    plt.title(f'Distribución de {title}')
    plt.xlabel(title)
    plt.ylabel('Frecuencia')
    plt.show()




def scatter_plot(feature1, feature2):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(feature1, feature2)
    ax.set_xlabel(feature1.name)
    ax.set_ylabel(feature2.name)
    plt.show()      

# limpieza de outliers con z-core

def limpiar_outliers_z_core(feature: pd.Series , umbral=2):
    z_cores=stats.zscore(feature)
    mask=abs(z_cores)< umbral
    feature = feature[mask]
    return feature , mask


# limpieza de outliers con IQR



def limpiar_outliers_iqr(feature: pd.Series ):
    """
    Elimina outliers usando el método del IQR.
    
    Parámetros:
        feature (pd.Series): feature numérica del DataFrame.

    Retorna:
        feature_limpia (pd.Series): Serie con outliers eliminados.
        mascara (pd.Series): Máscara booleana para aplicar al DataFrame original.
    """
    Q1 = feature.quantile(0.25)
    Q3 = feature.quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    mascara = (feature >= limite_inferior) & (feature <= limite_superior)
    feature = feature[mascara]

    return feature, mascara


def resumen_outliers(df):
  

    # Filtrar solo columnas numéricas
    numericas = df.select_dtypes(include=[np.number])

    # Diccionarios para guardar los resultados
    outliers_iqr = {}
    outliers_zscore = {}
   

    for col in numericas.columns:
        # ----- Cálculo de IQR -----
        Q1 = numericas[col].quantile(0.25)
        Q3 = numericas[col].quantile(0.75)
        IQR = Q3 - Q1
        condicion_iqr = (numericas[col] < Q1 - 1.5 * IQR) | (numericas[col] > Q3 + 1.5 * IQR)
        outliers_iqr[col] = condicion_iqr.sum()

        # ----- Cálculo de Z-score (±2) -----
        col_z = numericas[col].dropna()
        zscores = zscore(col_z)
        condicion_z = (zscores < -2) | (zscores > 2)
        outliers_zscore[col] = condicion_z.sum()

       

    # Crear el DataFrame resumen
    df_resumen = pd.DataFrame({
        'Outliers_IQR': outliers_iqr,
        'Outliers_Zscore': outliers_zscore,
        
    })

    return df_resumen




def histogram_por_lotes(df, por_lote=6):
    columnas = df.select_dtypes(include=['number']).columns  
    total_columnas = len(columnas)

    for inicio in range(0, total_columnas, por_lote):
        fin = min(inicio + por_lote, total_columnas)
        subset = columnas[inicio:fin]
        n = len(subset)
        filas = math.ceil(n / 3)
        fig, axes = plt.subplots(filas, 3, figsize=(5 * 3, 4 * filas))
        axes = axes.flatten()

        for i, col in enumerate(subset):
            datos = df[col].dropna()

            try:
                datos = datos.astype(float)  # Forzamos conversión
                if datos.nunique() > 1:
                    axes[i].hist(datos, color='green', bins=30, alpha=0.7, density=True, edgecolor='black', label='Histograma')
                    density = gaussian_kde(datos)
                    x_vals = np.linspace(min(datos), max(datos), 1000)
                    axes[i].plot(x_vals, density(x_vals), color='red', linewidth=2.5, label='Curva de Densidad')
                else:
                    axes[i].text(0.5, 0.5, 'Valor constante', ha='center', va='center')
                axes[i].set_title(col)
                axes[i].set_xlabel(col)

            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()



def boxplots_por_lotes(df, por_lote=6):
    columnas = df.select_dtypes(include=['number']).columns  
    print(columnas)
    total_columnas = len(columnas)

    for inicio in range(0, total_columnas, por_lote):
        fin = min(inicio + por_lote, total_columnas)
        subset = columnas[inicio:fin]
        n = len(subset)
        filas = math.ceil(n / 3)
        fig, axes = plt.subplots(filas, 3, figsize=(5 * 3, 4 * filas))
        axes = axes.flatten()

        for i, col in enumerate(subset):
            datos = df[col].dropna()

            try:
                datos = datos.astype(float)  # Forzamos conversión
                if datos.nunique() > 1:
                    axes[i].boxplot(datos, vert=False, patch_artist=True,
                                    boxprops=dict(facecolor='lightblue', color='blue'),
                                    medianprops=dict(color='red'))
                else:
                    axes[i].text(0.5, 0.5, 'Valor constante', ha='center', va='center')
                axes[i].set_title(col)
                axes[i].set_xlabel(col)

            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()



def scatter_por_lotes(df, eje_y='price', por_lote=6):
    # Filtrar solo columnas numéricas excluyendo la columna del eje y
    columnas = df.select_dtypes(include=['number']).columns
    columnas = [col for col in columnas if col != eje_y and df[col].nunique() > 1]

    total_columnas = len(columnas)

    for inicio in range(0, total_columnas, por_lote):
        fin = min(inicio + por_lote, total_columnas)
        subset = columnas[inicio:fin]
        n = len(subset)
        filas = math.ceil(n / 3)
        fig, axes = plt.subplots(filas, 3, figsize=(5 * 3, 4 * filas))
        axes = axes.flatten()

        for i, col in enumerate(subset):
            try:
                x = df[col].astype(float)
                y = df[eje_y].astype(float)
                axes[i].scatter(x, y, alpha=0.5, color='blue', edgecolors='w')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(eje_y)
                axes[i].set_title(f'{col} vs {eje_y}')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def bar_por_lotes(df, por_lote=6):
    columnas = df.select_dtypes(include=['number', 'category', 'object']).columns
    

    total_columnas = len(columnas)

    for inicio in range(0, total_columnas, por_lote):
        fin = min(inicio + por_lote, total_columnas)
        subset = columnas[inicio:fin]
        n = len(subset)
        filas = math.ceil(n / 3)
        fig, axes = plt.subplots(filas, 3, figsize=(15, 4 * filas))
        axes = axes.flatten()

        for i, col in enumerate(subset):
            datos = df[col].dropna()

            try:
                # Si tiene muchas categorías únicas, se descarta
                if datos.nunique() > 50:
                    axes[i].text(0.5, 0.5, 'Demasiados valores únicos', ha='center', va='center')
                    axes[i].set_title(col)
                    continue

                # Conteo de frecuencias
                conteo = datos.value_counts().sort_index()

                axes[i].bar(conteo.index.astype(str), conteo.values, color='skyblue')
                axes[i].set_title(col)
                axes[i].set_xlabel("Valores")
                axes[i].set_ylabel("Frecuencia")
                axes[i].tick_params(axis='x', rotation=45)

            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[i].set_title(col)

        # Eliminar ejes vacíos
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()



def mostrar_matriz_correlacion(df, metodo='pearson', mostrar_grafico=True):
    """
    Calcula y opcionalmente muestra la matriz de correlación de un DataFrame.
    
    Parámetros:
    - df: DataFrame de entrada.
    - metodo: 'pearson' (default), 'spearman', o 'kendall'.
    - mostrar_grafico: Si True, muestra un heatmap.

    Retorna:
    - matriz de correlación (DataFrame).
    """
    # Seleccionar solo columnas numéricas
    df_numerico = df.select_dtypes(include=['number'])
    
    # Calcular la matriz de correlación
    correlacion = df_numerico.corr(method=metodo)

    if mostrar_grafico:
        plt.figure(figsize=(20, 20))
        sns.heatmap(correlacion, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title(f'Matriz de correlación ({metodo})')
        plt.tight_layout()
        plt.show()

    return correlacion



def pares_correlacion_altas(corr_matrix, umbral=0.65):
    """
    Retorna un DataFrame con los pares de columnas con correlación absoluta >= umbral.

    Parámetros:
    - corr_matrix: Matriz de correlación (DataFrame).
    - umbral: Umbral mínimo absoluto de correlación (default=0.65).

    Retorna:
    - DataFrame con columnas: 'Variable_1', 'Variable_2', 'Correlación'
    """
    pares_altamente_correlacionados = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            correlacion = corr_matrix.iloc[i, j]
            if abs(correlacion) >= umbral:
                pares_altamente_correlacionados.append({
                    'Variable_1': col1,
                    'Variable_2': col2,
                    'Correlación': correlacion
                })

    df_resultado = pd.DataFrame(pares_altamente_correlacionados)
    df_resultado = df_resultado.sort_values(by='Correlación', key=lambda x: abs(x), ascending=False).reset_index(drop=True)

    return df_resultado



def split (dataframe):
    x= dataframe.drop(columns=['price'])
    y= dataframe['price']
    return x , y