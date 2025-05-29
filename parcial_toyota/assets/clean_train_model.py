import pandas as pd
import numpy as np
from dagster import asset, AssetIn, AssetKey
from dagstermill import define_dagstermill_asset 
from dagster import file_relative_path
from parcial_toyota.assets.helper_asset import filtrar_columna_por_rango
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.linear_model import Ridge , RidgeCV , Lasso , LassoCV
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
from sklearn.decomposition import PCA






analysis_notebook = define_dagstermill_asset(
    #deps=['reasignacion'],
    name="eda_toyota",  
    notebook_path=file_relative_path(__file__, "../notebooks/dataset.ipynb"),  # JOIN PATH
    group_name="RAW_DATA_PREPARATION",  
    ins={
        "df_toyota": AssetIn(key=AssetKey("df_toyota"),input_manager_key= "postgres_io_manager")  # Conexión al output de clean_dataset
    },
    io_manager_key="io_manager",
)



@asset(
        deps=['eda_toyota'],
        group_name="MODEL_TRAING_TEST",
        ins={"df_toyota": AssetIn(key=AssetKey("eda_toyota"))}
        

)

def seleccion_lambda_ridge(context,df_toyota):

    url=Path(__file__).parent.parent.parent / 'output' / 'data_clean.csv'
    df_toyota_ridge = pd.read_csv(url)

    x = df_toyota_ridge.drop(columns=['price'])
    y = df_toyota_ridge['price']

    # Escalado
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Buscar mejor alpha con RidgeCV
    lambdas = np.logspace(-1, 10, num=1000)
    ridge_cv = RidgeCV(alphas=lambdas, cv=5)
    ridge_cv.fit(x_scaled, y)
    best_alpha = ridge_cv.alpha_

    ridge = {
        "alpha": best_alpha,
        "df_toyota_ridge": df_toyota_ridge
    }



    return ridge


@asset(
        required_resource_keys={"mlflow_toyota_ridge"},
        deps=[seleccion_lambda_ridge],
        group_name="MODEL_TRAING_TEST",
        ins={"ridge": AssetIn(key=AssetKey("seleccion_lambda_ridge"))}

)
def entrenar_evalular_modelo_ridge(context, ridge):

    best_alpha = ridge['alpha']
    df_toyota_ridge = ridge['df_toyota_ridge']
    
    mlflow = context.resources.mlflow_toyota_ridge

    X = df_toyota_ridge.drop(columns=['price'])
    y = df_toyota_ridge['price']

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = (scaler_y.fit_transform(y.values.reshape(-1, 1))).ravel()


    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    mse_scores, mae_scores, mape_scores ,r2_scores , rmse_scores = [],   [], [], [], []

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="validacion_cruzada_regresion_ridge") as main_run:
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.sklearn.autolog()  
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

                y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

                model = Ridge(alpha=best_alpha)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                coef_df = pd.DataFrame({
                        'Variable': X.columns,
                        'Coeficiente': model.coef_
                        })

                coef_path = f"coeficientes_fold_{fold}.csv"
                coef_df.to_csv(coef_path, index=False)
                mlflow.log_artifact(coef_path)
                os.remove(coef_path)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
                r2_scores.append(r2)
                rmse_scores.append(rmse)
                mlflow.log_metric("mse_prueba", mse)
                mlflow.log_metric("mae_prueba", mae)
                mlflow.log_metric("mape_prueba", mape)
                mlflow.log_metric("r2_prueba", r2)
                mlflow.log_metric("rmse_prueba", rmse)

              
                # ---------- analisis residuales ----------
                residuals = y_test - y_pred
                fitted_vals = y_pred

                fig, ax = plt.subplots(2, 2, figsize=(14, 10))
                ax[0, 0].hist(residuals, bins=30, color='skyblue', edgecolor='black')
                ax[0, 0].set_title("Histograma de residuos")
                ax[0, 0].set_xlabel("Error")
                ax[0, 0].set_ylabel("Frecuencia")

                # grafico de predeciiones vs valores reales cin recta de 45 grados
                ax[1,1].scatter(y_test, y_pred, alpha=0.7)
                ax[1,1].set_title("Predicciones vs Valores reales")
                ax[1,1].set_xlabel("Valores reales")
                ax[1,1].set_ylabel("Predicciones")
                ax[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax[1,1].grid(True)


                sm.qqplot(residuals, line='45', fit=True, ax=ax[0, 1])
                ax[0, 1].set_title("QQ plot de los residuos")

                ax[1, 0].scatter(fitted_vals, residuals, alpha=0.5)
                ax[1, 0].axhline(0, color='red', linestyle='--')
                ax[1, 0].set_title("Residuos vs Valores ajustados")
                ax[1, 0].set_xlabel("Valores ajustados")
                ax[1, 0].set_ylabel("Residuos")

            
                plt.tight_layout()
                diag_plot_path = f"diagnosticos_residuos_fold_{fold}.png"
                plt.savefig(diag_plot_path)
                mlflow.log_artifact(diag_plot_path)
                plt.close()
                os.remove(diag_plot_path)

        # ---------- Métricas promedio global ----------
        mlflow.log_metric("avg_mse", np.mean(mse_scores))
        mlflow.log_metric("avg_mae", np.mean(mae_scores))
        mlflow.log_metric("avg_mape", np.mean(mape_scores))
        mlflow.log_metric("avg_r2", np.mean(r2_scores))
        mlflow.log_metric("avg_rmse", np.mean(rmse_scores))

        context.log.info(f"MSE promedio: {np.mean(mse_scores):.2f}")
        context.log.info(f"MAE promedio: {np.mean(mae_scores):.2f}")
        context.log.info(f"MAPE promedio: {np.mean(mape_scores):.2f}%")
        context.log.info(f"R2 promedio: {np.mean(r2_scores):.2f}")
        context.log.info(f"RMSE promedio: {np.mean(rmse_scores):.2f}")

    metricas_ridge = {
            "mse": np.mean(mse_scores),
            "mae": np.mean(mae_scores),
            "mape": np.mean(mape_scores),
            "r2": np.mean(r2_scores),
            "rmse": np.mean(rmse_scores)
        }

    return metricas_ridge

      

@asset(
        deps=['eda_toyota'],
        group_name="MODEL_TRAING_TEST",
        ins={"df_toyota": AssetIn(key=AssetKey("eda_toyota"))}
        

)

def seleccion_variables_lasso(context,df_toyota):

    url=Path(__file__).parent.parent.parent / 'output' / 'data_clean.csv'
    df_toyota_lasso = pd.read_csv(url)

    dummies = pd.get_dummies(df_toyota_lasso['fuel_type_encoded'], prefix='fuel_type')

# Unir las dummies al DataFrame original
    df_toyota_lasso = pd.concat([df_toyota_lasso, dummies], axis=1)

    # Eliminar la columna original 'fuel_type_encoded'
    df_toyota_lasso = df_toyota_lasso.drop('fuel_type_encoded', axis=1)

    X = df_toyota_lasso.drop(columns=['price'])
    y = df_toyota_lasso['price']


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = (scaler_y.fit_transform(y.values.reshape(-1, 1))).ravel()


    alphas = np.logspace(-1, 10, 1000)  


    lasso_cv_model = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
    lasso_cv_model.fit(X_scaled, y_scaled)


    optimal_alpha = lasso_cv_model.alpha_
    print(f"Alpha óptimo encontrado por LassoCV: {optimal_alpha:.6f}")
    feature_names = X.columns

    # Obtener coeficientes del modelo entrenado
    coefficients = lasso_cv_model.coef_

    # Crear DataFrame para visualización
    coef_df = pd.DataFrame({
        'Variable': feature_names,
        'Coeficiente': coefficients
    })

    # Filtrar variables cuyo coeficiente es distinto de cero
    variables_seleccionadas = coef_df[coef_df['Coeficiente'] != 0]['Variable'].tolist()

    # Filtrar X para que contenga solo esas variables y que tenga tambien la columna price
    df_toyota_lasso_filtrado = df_toyota_lasso[variables_seleccionadas + ['price']]







    return df_toyota_lasso_filtrado


@asset(
        required_resource_keys={"mlflow_toyota_lasso"},
        deps=['seleccion_variables_lasso'],
        group_name="MODEL_TRAING_TEST",
        ins={"df_toyota_lasso_filtrado": AssetIn(key=AssetKey("seleccion_variables_lasso"))}

)
def entrenar_evalular_modelo_lasso(context, df_toyota_lasso_filtrado):

    mlflow = context.resources.mlflow_toyota_lasso

    X = df_toyota_lasso_filtrado.drop(columns=['price'])
    y = df_toyota_lasso_filtrado['price']

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = (scaler_y.fit_transform(y.values.reshape(-1, 1))).ravel()

    lambdas = np.logspace(-1, 10, num=1000)
    lasso_cv = LassoCV(alphas=lambdas, cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled, y_scaled)
    best_alpha = lasso_cv.alpha_

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    mse_scores, mae_scores, mape_scores ,r2_scores , rmse_scores = [], [], [], [], []

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="validacion_cruzada_regresion_lasso") as main_run:
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.sklearn.autolog()  
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

                y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

                model = Lasso(alpha=best_alpha, max_iter=10000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                coef_df = pd.DataFrame({
                        'Variable': X.columns,
                        'Coeficiente': model.coef_
                        })

                coef_path = f"coeficientes_fold_{fold}.csv"
                coef_df.to_csv(coef_path, index=False)
                mlflow.log_artifact(coef_path)
                os.remove(coef_path)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
                r2_scores.append(r2)
                rmse_scores.append(rmse)
                mlflow.log_metric("mse_prueba", mse)
                mlflow.log_metric("mae_prueba", mae)
                mlflow.log_metric("mape_prueba", mape)
                mlflow.log_metric("r2_prueba", r2)
                mlflow.log_metric("rmse_prueba", rmse)

                # ---------- Gráfico de residuos ----------
                
               

                # ---------- Gráficos diagnósticos ----------
                residuals = y_test - y_pred
                fitted_vals = y_pred

                # grafico de predeciiones vs valores reales cin recta de 45 grados
                


                fig, ax = plt.subplots(2, 2, figsize=(14, 10))
                ax[0, 0].hist(residuals, bins=30, color='skyblue', edgecolor='black')
                ax[0, 0].set_title("Histograma de residuos")
                ax[0, 0].set_xlabel("Error")
                ax[0, 0].set_ylabel("Frecuencia")


                ax[1,1].scatter(y_test, y_pred, alpha=0.7)
                ax[1,1].set_title("Predicciones vs Valores reales")
                ax[1,1].set_xlabel("Valores reales")
                ax[1,1].set_ylabel("Predicciones")
                ax[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax[1,1].grid(True)

                sm.qqplot(residuals, line='45', fit=True, ax=ax[0, 1])
                ax[0, 1].set_title("QQ plot de los residuos")

                ax[1, 0].scatter(fitted_vals, residuals, alpha=0.5)
                ax[1, 0].axhline(0, color='red', linestyle='--')
                ax[1, 0].set_title("Residuos vs Valores ajustados")
                ax[1, 0].set_xlabel("Valores ajustados")
                ax[1, 0].set_ylabel("Residuos")

                
                plt.tight_layout()
                diag_plot_path = f"diagnosticos_residuos_fold_{fold}.png"
                plt.savefig(diag_plot_path)
                mlflow.log_artifact(diag_plot_path)
                plt.close()
                os.remove(diag_plot_path)

        # ---------- Métricas promedio global ----------
        mlflow.log_metric("avg_mse", np.mean(mse_scores))
        mlflow.log_metric("avg_mae", np.mean(mae_scores))
        mlflow.log_metric("avg_mape", np.mean(mape_scores))
        mlflow.log_metric("avg_r2", np.mean(r2_scores))
        mlflow.log_metric("avg_rmse", np.mean(rmse_scores))

        context.log.info(f"MSE promedio: {np.mean(mse_scores):.2f}")
        context.log.info(f"MAE promedio: {np.mean(mae_scores):.2f}")
        context.log.info(f"MAPE promedio: {np.mean(mape_scores):.2f}%")
        context.log.info(f"R2 promedio: {np.mean(r2_scores):.2f}")
        context.log.info(f"RMSE promedio: {np.mean(rmse_scores):.2f}")

    metricas_lasso = {
            "mse": np.mean(mse_scores),
            "mae": np.mean(mae_scores),
            "mape": np.mean(mape_scores),
            "r2": np.mean(r2_scores),
            "rmse": np.mean(rmse_scores)
        }



    return metricas_lasso





@asset(
    deps=['eda_toyota'],
    group_name="MODEL_TRAING_TEST",
    ins={
        "df_toyota": AssetIn(key=AssetKey("eda_toyota"))  # Conexión al output de clean_dataset
    }

)
def filtrado_y_normalizado(context ,df_toyota) :
    """
    Selecciona las columnas relevantes del DataFrame.
    """
    
    url=Path(__file__).parent.parent.parent / 'output' / 'data_clean.csv'
    df_toyota_limpio = pd.read_csv(url)

    correlations = df_toyota_limpio.corr(numeric_only=True)['price'].abs()

    relevant_features = correlations[correlations >= 0.20].index

# Creamos un nuevo DataFrame con solo las variables relevantes
    df_filtrado_normalizado = df_toyota_limpio[relevant_features]
# Normalizamos las variables age, km y price para que tengan un rango de 0 a 1
    
    scaler = MinMaxScaler()
    df_filtrado_normalizado['age_08_04_calculada'] = scaler.fit_transform(df_filtrado_normalizado['age_08_04_calculada'].values.reshape(-1, 1))

    df_filtrado_normalizado['km'] = scaler.fit_transform(df_filtrado_normalizado['km'].values.reshape(-1, 1))

    df_filtrado_normalizado['price'] = scaler.fit_transform(df_filtrado_normalizado['price'].values.reshape(-1, 1))

    #filtarmos kilometros
    _,mascara_km = filtrar_columna_por_rango(df_filtrado_normalizado['km'],0.1,0.8)

    df_filtrado_normalizado = df_filtrado_normalizado[mascara_km]

   
    
    #
    
    return df_filtrado_normalizado





@asset(
    deps=[filtrado_y_normalizado],
    required_resource_keys={"mlflow"},
    group_name="MODEL_TRAING_TEST",
    ins={"df_filtrado_normalizado": AssetIn(key=AssetKey("filtrado_y_normalizado"))}
)
def entrenar_modelo_evaluar_mco(context, df_filtrado_normalizado):
    mlflow = context.resources.mlflow

   
    X = df_filtrado_normalizado.drop(columns=['price'])
    y = df_filtrado_normalizado['price']

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    rmse_scores = []
    summaries = []

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="validacion_cruzada_regresion_mco") as main_run:
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.statsmodels.autolog()

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test, has_constant='add')

                model = sm.OLS(y_train, X_train_const).fit()
                y_pred = model.predict(X_test_const)

                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
                r2_scores.append(r2)
                rmse_scores.append(rmse)
                mlflow.log_metric("mse_prueba", mse)
                mlflow.log_metric("mae_prueba", mae)
                mlflow.log_metric("mape_prueba", mape)
                mlflow.log_metric("r2_prueba", r2)
                mlflow.log_metric("rmse_prueba", rmse)

                # Guardar summary
                summary_text = model.summary().as_text()
                summaries.append(summary_text)
                summary_file = f"summary_fold_{fold}.txt"
                with open(summary_file, "w") as f:
                    f.write(summary_text)
                mlflow.log_artifact(summary_file)
                os.remove(summary_file)

               

                # ---------- Gráficos diagnósticos ----------
                residuals = y_test - y_pred
                fitted_vals = y_pred

                fig, ax = plt.subplots(2, 2, figsize=(14, 10))
                ax[0, 0].hist(residuals, bins=30, color='skyblue', edgecolor='black')
                ax[0, 0].set_title("Histograma de residuos")
                ax[0, 0].set_xlabel("Error")
                ax[0, 0].set_ylabel("Frecuencia")

                sm.qqplot(residuals, line='45', fit=True, ax=ax[0, 1])
                ax[0, 1].set_title("QQ plot de los residuos")

                ax[1, 0].scatter(fitted_vals, residuals, alpha=0.5)
                ax[1, 0].axhline(0, color='red', linestyle='--')
                ax[1, 0].set_title("Residuos vs Valores ajustados")
                ax[1, 0].set_xlabel("Valores ajustados")
                ax[1, 0].set_ylabel("Residuos")
                # valores predichos vs valores reales
                ax[1, 1].scatter(y_test, y_pred, alpha=0.7)
                ax[1, 1].set_title("Predicciones vs Valores reales")
                ax[1, 1].set_xlabel("Valores reales")
                ax[1, 1].set_ylabel("Predicciones")
                ax[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax[1, 1].grid(True)
                
                plt.tight_layout()
                diag_plot_path = f"diagnosticos_residuos_fold_{fold}.png"
                plt.savefig(diag_plot_path)
                mlflow.log_artifact(diag_plot_path)
                plt.close()
                os.remove(diag_plot_path)

        # ---------- Métricas promedio global ----------
        mlflow.log_metric("avg_mse", np.mean(mse_scores))
        mlflow.log_metric("avg_mae", np.mean(mae_scores))
        mlflow.log_metric("avg_mape", np.mean(mape_scores))
        mlflow.log_metric("avg_r2", np.mean(r2_scores))
        mlflow.log_metric("avg_rmse", np.mean(rmse_scores))

        context.log.info(f"MSE promedio: {np.mean(mse_scores):.2f}")
        context.log.info(f"MAE promedio: {np.mean(mae_scores):.2f}")
        context.log.info(f"MAPE promedio: {np.mean(mape_scores):.2f}%")
        context.log.info(f"R2 promedio: {np.mean(r2_scores):.2f}")
        context.log.info(f"RMSE promedio: {np.mean(rmse_scores):.2f}")

    metricas_mco = {
            "mse": np.mean(mse_scores),
            "mae": np.mean(mae_scores),
            "mape": np.mean(mape_scores),
            "r2": np.mean(r2_scores),
            "rmse": np.mean(rmse_scores)
        }


    return metricas_mco



@asset(
    deps=['eda_toyota'],
    group_name="MODEL_TRAING_TEST",
    ins={
        "df_toyota": AssetIn(key=AssetKey("eda_toyota"))  # Conexión al output de clean_dataset
    }

)
def seleccion_componentes(context ,df_toyota) :
    """
    Selecciona las columnas relevantes del DataFrame.
    """
    
    url=Path(__file__).parent.parent.parent / 'output' / 'data_clean.csv'
    df_toyota_pca = pd.read_csv(url)
    df_toyota_pca=pd.get_dummies(df_toyota_pca,columns=['fuel_type_encoded']) 
    X=df_toyota_pca.drop(columns=['price'])
   

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_temp=PCA()
    pca_temp.fit(X_scaled)

    varianza_acumulada=np.cumsum(pca_temp.explained_variance_ratio_)

    componentes_optimos = np.argmax(varianza_acumulada >= 0.95) + 1

    pca={
        "componentes_optimos": componentes_optimos,
        "df_toyota_pca": df_toyota_pca
    }

    

   
    
    #
    
    return pca


@asset(
    deps=['seleccion_componentes'],
    required_resource_keys={"mlflow_toyota_pca"},
    group_name="MODEL_TRAING_TEST",
    ins={
        "pca": AssetIn(key=AssetKey("seleccion_componentes"))  # Conexión al output de clean_dataset
    }

)
def entrenar_evaluar_modelo_pca(context ,pca) :
   
    mlflow = context.resources.mlflow_toyota_pca
    
    df_toyota_pca=pca['df_toyota_pca']
    n_componentes=pca['componentes_optimos']


    X=df_toyota_pca.drop(columns=['price'])
    y=df_toyota_pca['price']

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    scaler_y =  StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    pca_model = PCA(n_components=n_componentes)
    X_pca = pca_model.fit_transform(X_scaled)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    rmse_scores = []
    if mlflow.active_run():
        mlflow.end_run()
    fold = 1
    with mlflow.start_run(run_name="validacion_cruzada_regresion_pca") as main_run:
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_pca), 1):
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                mlflow.statsmodels.autolog()

                X_train, X_test = X_pca[train_idx], X_pca[test_idx]
                y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test, has_constant='add')

                model = sm.OLS(y_train, X_train_const).fit()
                y_pred = model.predict(X_test_const)
                y_test_flat = y_test.ravel()
                y_pred_flat = y_pred.ravel()

                mse = mean_squared_error(y_test_flat, y_pred_flat)
                mae = mean_absolute_error(y_test_flat, y_pred_flat)
                mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_flat)) * 100
                r2 = r2_score(y_test_flat, y_pred_flat)
                rmse = np.sqrt(mse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
                r2_scores.append(r2)
                rmse_scores.append(rmse)
                mlflow.log_metric("mse_prueba", mse)
                mlflow.log_metric("mae_prueba", mae)
                mlflow.log_metric("mape_prueba", mape)
                mlflow.log_metric("r2_prueba", r2)
                mlflow.log_metric("rmse_prueba", rmse)
                # guardar summary en artifact
                summary_text = model.summary().as_text()
                summary_file = f"summary_fold_{fold}.txt"
                with open(summary_file, "w") as f:
                    f.write(summary_text)
                mlflow.log_artifact(summary_file)
                os.remove(summary_file)

                # hacer una fgi de 4*4 de los residuos con orientacion a objetos
                fig, ax = plt.subplots(2, 2, figsize=(14, 10))
                #histograma de residuos
                ax[0, 0].hist((y_test_flat - y_pred_flat), bins=30, color='skyblue', edgecolor='black')
                ax[0, 0].set_title("Histograma de residuos")
                ax[0, 0].set_xlabel("Error")
                ax[0, 0].set_ylabel("Frecuencia")
                #valores predichos vs valores reales
                ax[0, 1].scatter(y_test_flat, y_pred_flat, alpha=0.7)
                ax[0, 1].set_title("Predicciones vs Valores reales")
                ax[0, 1].set_xlabel("Valores reales")
                ax[0, 1].set_ylabel("Predicciones")

                #qq plot de los residuos
                sm.qqplot((y_test_flat - y_pred_flat), line='45', fit=True, ax=ax[1, 0])
                ax[1, 0].set_title("QQ plot de los residuos")

                #residuos vs valores ajustados
                ax[1, 1].scatter(y_pred_flat, (y_test_flat - y_pred_flat), alpha=0.5)    
                plt.tight_layout()
                diag_plot_path = f"diagnosticos_residuos_fold_{fold}.png"
                plt.savefig(diag_plot_path)
                mlflow.log_artifact(diag_plot_path)
                plt.close()
                os.remove(diag_plot_path)
                
                
            # ---------- Métricas promedio global ----------
        mlflow.log_metric("avg_mse", np.mean(mse_scores))
        mlflow.log_metric("avg_mae", np.mean(mae_scores))
        mlflow.log_metric("avg_mape", np.mean(mape_scores))
        mlflow.log_metric("avg_r2", np.mean(r2_scores))
        mlflow.log_metric("avg_rmse", np.mean(rmse_scores))

        context.log.info(f"MSE promedio: {np.mean(mse_scores):.2f}")
        context.log.info(f"MAE promedio: {np.mean(mae_scores):.2f}")
        context.log.info(f"MAPE promedio: {np.mean(mape_scores):.2f}%")
        context.log.info(f"R2 promedio: {np.mean(r2_scores):.2f}")
        context.log.info(f"RMSE promedio: {np.mean(rmse_scores):.2f}")


    metricas_pca = {
            "mse": np.mean(mse_scores),
            "mae": np.mean(mae_scores),
            "mape": np.mean(mape_scores),
            "r2": np.mean(r2_scores),
            "rmse": np.mean(rmse_scores)
        }
        
   
    
    return metricas_pca