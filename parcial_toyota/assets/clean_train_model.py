import pandas as pd
import numpy as np
from dagster import asset, AssetIn, AssetKey, Output, multi_asset, AssetOut, AssetExecutionContext
from dagstermill import define_dagstermill_asset 
from dagster import file_relative_path
from parcial_toyota.assets.helper_asset import dividir_datos , regresion_modelo
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge , RidgeCV
@asset(
    group_name="clean_dataset",
    description="Carga el dataset de Toyota Corolla desde un archivo CSV"
    
)
def carga_dataset():
    url = "https://raw.githubusercontent.com/dodobeatle/dataeng-datos/refs/heads/main/ToyotaCorolla.csv"
    df_toyota = pd.read_csv(url)
    return df_toyota


@asset(
    deps=[carga_dataset] ,
     group_name="clean_dataset",
     description="Limpia el dataset de Toyota Corolla eliminando columnas innecesarias",
    ins={
        "df_toyota": AssetIn(key=AssetKey("carga_dataset"))  # Conexión al output de carga_dataset
    }
)
def reasignacion(df_toyota:pd.DataFrame) -> pd.DataFrame:
    # reasignacion de columnas
    df_toyota.columns = [
    col.strip().lower().replace(' ', '_') for col in df_toyota.columns]

    return   df_toyota





analysis_notebook = define_dagstermill_asset(
    deps=['reasignacion'],
    name="eda_toyota",  
    notebook_path=file_relative_path(__file__, "../notebooks/dataset.ipynb"),  # JOIN PATH
    group_name="clean_dataset",  
    ins={
        "df_toyota": AssetIn("reasignacion")  # Conexión al output de clean_dataset
    },
    io_manager_key="io_manager",
)



@asset(
        deps=['eda_toyota'],
        group_name="regresion_ridge",
        ins={"df_toyota": AssetIn(key=AssetKey("eda_toyota"))}
        

)

def seleccion_variables_ridge(context,df_toyota):

    url='//home/agustin/Escritorio/notebbooks/parcial_toyota/output/dataset_clean.csv'
    df_toyota_ridge = pd.read_csv(url)

    x = df_toyota_ridge.drop(columns=['price'])
    y=df_toyota_ridge['price']
    lambdas = np.logspace(2, 10, num=100)  # de 10^2 a 10^10
    coefs = []

    for alpha in lambdas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(x, y)
        coefs.append(ridge.coef_)

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, coefs)
    plt.xscale("log")
    plt.xlabel("alpha (λ)")
    plt.ylabel("Coeficientes")
    plt.title("Coeficientes Ridge vs. Lambda")
    plt.grid(True)

    output_path = "ridge_coeff_plot.png"
    plt.savefig(output_path)
    plt.close()


    return df_toyota_ridge



      

































@asset(
    deps=['eda_toyota'],
    group_name="mco_regresion",
    ins={
        "df_toyota": AssetIn(key=AssetKey("eda_toyota"))  # Conexión al output de clean_dataset
    }

)
def seleccion_columnas_mco(context ,df_toyota) :
    """
    Selecciona las columnas relevantes del DataFrame.
    """
    url='//home/agustin/Escritorio/notebbooks/parcial_toyota/output/dataset_clean.csv'
    df_toyota_clean = pd.read_csv(url)
    # Selección de columnas
    df_toyota_clean.drop(columns=['airco','cd_player','mistlamps','km','sport_model','quarterly_tax'],axis=1,inplace=True)
    
    
    # Guardar el DataFrame limpio en un archivo CSV
    
    
    return df_toyota_clean




@multi_asset(
    deps=[seleccion_columnas_mco],
    group_name="mco_regresion",
    ins={
        "df_toyota": AssetIn(key=AssetKey("seleccion_columnas_mco"))  # Conexión al output de clean_dataset
    }
    ,
    outs={
        "x_train": AssetOut(key=AssetKey("x_train")),
        "x_test": AssetOut(key=AssetKey("x_test")),
        "y_train": AssetOut(key=AssetKey("y_train")),
        "y_test": AssetOut(key=AssetKey("y_test")),
    },
)
def split_data(context,df_toyota):

    
    x_train, x_test, y_train, y_test = dividir_datos(df_toyota)

    
   
    return x_train, x_test, y_train, y_test


@asset(
    deps=[split_data],

    required_resource_keys={"mlflow"},
   
    group_name="mco_regresion",
    ins={
        "x_train": AssetIn(key=AssetKey("x_train")),
        "y_train": AssetIn(key=AssetKey("y_train")),
        "x_test": AssetIn(key=AssetKey("x_test")),
        "y_test": AssetIn(key=AssetKey("y_test")),
    }
)
def entrenar_modelo_evaluar(context, x_train, y_train, x_test, y_test):
    mlflow = context.resources.mlflow

    # Cerrás el run automático que abrió Dagster
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Abrís un nuevo run con nombre personalizado
    with mlflow.start_run(run_name="entrenar_evaluar_modelo_mco"):
        mlflow.statsmodels.autolog()

        # Entrenamiento
        modelo = regresion_modelo(x_train, y_train)

        # Predicciones
        x_test = sm.add_constant(x_test)
        y_pred = modelo.predict(x_test)

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Loguear métricas
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)

        # Gráfico 1
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("Valores reales")
        plt.ylabel("Predicciones")
        plt.title("Gráfico de residuos")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.grid(True)
        plot_path = "residuals_plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        os.remove(plot_path)

        # Gráficos diagnósticos
        residuals = y_test - y_pred
        fitted_vals = y_pred
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        ax[0, 0].hist(residuals, bins=30, color='skyblue', edgecolor='black')
        ax[0, 0].set_title("Histograma de residuos")
        ax[0, 0].set_xlabel("Error")
        ax[0, 0].set_ylabel("Frecuencia")

        sm.qqplot(residuals, line='45', fit=True, ax=ax[0, 1])
        ax[0, 1].set_title("QQ plot de los residuos (test)")
        ax[1, 0].scatter(fitted_vals, residuals, alpha=0.5)
        ax[1, 0].axhline(0, color='red', linestyle='--')
        ax[1, 0].set_title("Residuos vs Valores ajustados")
        ax[1, 0].set_xlabel("Valores ajustados")
        ax[1, 0].set_ylabel("Residuos")
        ax[1, 1].axis('off')
        plt.tight_layout()
        diag_plot_path = "diagnosticos_residuos.png"
        plt.savefig(diag_plot_path)
        mlflow.log_artifact(diag_plot_path)
        plt.close()
        os.remove(diag_plot_path)

        context.log.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    return modelo




# def entrenar_modelo_evaluar(context, x_train, y_train,x_test, y_test):
#     mlflow = context.resources.mlflow
#     mlflow.statsmodels.autolog()


#         # Entrenamiento
#     modelo = regresion_modelo(x_train, y_train)

#         # Predicciones
#     x_test=sm.add_constant(x_test)
#     y_pred = modelo.predict(x_test)

#         # Calcular métricas
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

#         # Loguear métricas manualmente
#     mlflow.log_metric("test_mae", mae)
#     mlflow.log_metric("test_rmse", rmse)
#     mlflow.log_metric("test_mape", mape)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(y_test, y_pred, alpha=0.7)
#     plt.xlabel("Valores reales")
#     plt.ylabel("Predicciones")
#     plt.title("Gráfico de residuos")
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
#     plt.grid(True)

#         # Guardar y registrar como artefacto
#     plot_path = "residuals_plot.png"
#     plt.savefig(plot_path)
#     mlflow.log_artifact(plot_path)

#         # Eliminar el archivo local si querés evitar basura
#     os.remove(plot_path)

#     # 🔹 Cálculo de residuos y valores ajustados
#     residuals = y_test - y_pred
#     fitted_vals = y_pred

#     # 🔹 Gráfico 2-4: Diagnósticos múltiples
#     fig, ax = plt.subplots(2, 2, figsize=(14, 10))

#     # Histograma de residuos
#     ax[0, 0].hist(residuals, bins=30, color='skyblue', edgecolor='black')
#     ax[0, 0].set_title("Histograma de residuos")
#     ax[0, 0].set_xlabel("Error")
#     ax[0, 0].set_ylabel("Frecuencia")

#     # QQ-plot
#     sm.qqplot(residuals, line='45', fit=True, ax=ax[0, 1])
#     ax[0, 1].set_title("QQ plot de los residuos (test)")

#     # Residuos vs valores ajustados
#     ax[1, 0].scatter(fitted_vals, residuals, alpha=0.5)
#     ax[1, 0].axhline(0, color='red', linestyle='--')
#     ax[1, 0].set_title("Residuos vs Valores ajustados")
#     ax[1, 0].set_xlabel("Valores ajustados")
#     ax[1, 0].set_ylabel("Residuos")

#     # Espacio vacío (puede usarse para futuros gráficos)
#     ax[1, 1].axis('off')

#     # Guardar y loguear
#     diag_plot_path = "diagnosticos_residuos.png"
#     plt.tight_layout()
#     plt.savefig(diag_plot_path)
#     mlflow.log_artifact(diag_plot_path)
#     plt.close()
#     os.remove(diag_plot_path)

#         # Log informativo (opcional)
#     context.log.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

#     return modelo




