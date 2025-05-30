import pandas as pd

from dagster import asset, AssetIn, AssetKey
import os

@asset(
    deps=[
        'entrenar_evalular_modelo_ridge',
        'entrenar_evalular_modelo_lasso',
        'entrenar_modelo_evaluar_mco',
        'entrenar_evaluar_modelo_pca'
    ],
    required_resource_keys={"comparacion_modelos"},
    group_name="MODEL_SELECTION",
    ins={
        "modelo_ridge": AssetIn(key=AssetKey("entrenar_evalular_modelo_ridge")),
        "modelo_lasso": AssetIn(key=AssetKey("entrenar_evalular_modelo_lasso")),
        "modelo_mco": AssetIn(key=AssetKey("entrenar_modelo_evaluar_mco")),
        "modelo_pca": AssetIn(key=AssetKey("entrenar_evaluar_modelo_pca"))
    }
)
def seleccion_modelo(context, modelo_ridge, modelo_lasso, modelo_mco, modelo_pca):
    """
    Compara los cuatro modelos entrenados y selecciona el mejor basado en las métricas individuales.
    """
    mlflow = context.resources.comparacion_modelos
    # Crear DataFrame para comparar métricas
    metricas = {
        'Modelo': ['Ridge', 'Lasso', 'MCO', 'PCA'],
        'MSE': [
            modelo_ridge['mse'],
            modelo_lasso['mse'],
            modelo_mco['mse'],
            modelo_pca['mse']
        ],
        'MAE': [
            modelo_ridge['mae'],
            modelo_lasso['mae'],
            modelo_mco['mae'],
            modelo_pca['mae']
        ],
        'MAPE': [
            modelo_ridge['mape'],
            modelo_lasso['mape'],
            modelo_mco['mape'],
            modelo_pca['mape']
        ],
        'R2': [
            modelo_ridge['r2'],
            modelo_lasso['r2'],
            modelo_mco['r2'],
            modelo_pca['r2']
        ],
        'RMSE': [
            modelo_ridge['rmse'],
            modelo_lasso['rmse'],
            modelo_mco['rmse'],
            modelo_pca['rmse']
        ]

        
    }
    
    df_metricas = pd.DataFrame(metricas)
    
    # Encontrar el mejor modelo para cada métrica
    mejor_mse = df_metricas.loc[df_metricas['MSE'].idxmin()]
    mejor_mae = df_metricas.loc[df_metricas['MAE'].idxmin()]
    mejor_mape = df_metricas.loc[df_metricas['MAPE'].idxmin()]
    mejor_r2 = df_metricas.loc[df_metricas['R2'].idxmax()]
    mejor_rmse = df_metricas.loc[df_metricas['RMSE'].idxmin()]
    
    # Crear resultado final con los mejores modelos por métrica
    resultado = {
        'mejor_por_mse': {
            'modelo': mejor_mse['Modelo'],
            'valor': mejor_mse['MSE']
        },
        'mejor_por_mae': {
            'modelo': mejor_mae['Modelo'],
            'valor': mejor_mae['MAE']
        },
        'mejor_por_mape': {
            'modelo': mejor_mape['Modelo'],
            'valor': mejor_mape['MAPE']
        },
        'mejor_por_r2': {
            'modelo': mejor_r2['Modelo'],
            'valor': mejor_r2['R2']
        },
        'mejor_por_rmse': {
            'modelo': mejor_rmse['Modelo'],
            'valor': mejor_rmse['RMSE']
        },
        'comparacion_completa': df_metricas.to_dict('records')
    }

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="mejor_modelo") as main_run:
        
     

        
        mlflow.log_metric("best_mse_model", resultado['mejor_por_mse']['valor'])
        mlflow.log_metric("best_mae_model", resultado['mejor_por_mae']['valor'])
        mlflow.log_metric("best_mape_model", resultado['mejor_por_mape']['valor'])
        mlflow.log_metric("best_r2_model", resultado['mejor_por_r2']['valor'])
        mlflow.log_metric("best_rmse_model", resultado['mejor_por_rmse']['valor'])

        # Log model names as parameters
        mlflow.log_param("best_mse_model_name", resultado['mejor_por_mse']['modelo'])
        mlflow.log_param("best_mae_model_name", resultado['mejor_por_mae']['modelo'])
        mlflow.log_param("best_mape_model_name", resultado['mejor_por_mape']['modelo'])
        mlflow.log_param("best_r2_model_name", resultado['mejor_por_r2']['modelo'])
        mlflow.log_param("best_rmse_model_name", resultado['mejor_por_rmse']['modelo'])

        

    # context.log.info("Mejores modelos por métrica:")
    # context.log.info(f"MSE: {resultado['mejor_por_mse']['modelo']} ({resultado['mejor_por_mse']['valor']:.4f})")
    # context.log.info(f"MAE: {resultado['mejor_por_mae']['modelo']} ({resultado['mejor_por_mae']['valor']:.4f})")
    # context.log.info(f"MAPE: {resultado['mejor_por_mape']['modelo']} ({resultado['mejor_por_mape']['valor']:.4f}%)")
    # context.log.info(f"R2: {resultado['mejor_por_r2']['modelo']} ({resultado['mejor_por_r2']['valor']:.4f})")
    # context.log.info(f"RMSE: {resultado['mejor_por_rmse']['modelo']} ({resultado['mejor_por_rmse']['valor']:.4f})")
    
    return resultado
