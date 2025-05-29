import pandas as pd
import numpy as np
from dagster import asset, AssetIn, AssetKey
from sklearn.linear_model import Ridge, Lasso
import statsmodels.api as sm
from sklearn.decomposition import PCA
from pathlib import Path

@asset(
    deps=[
        'entrenar_evalular_modelo_ridge',
        'entrenar_evalular_modelo_lasso',
        'entrenar_modelo_evaluar_mco',
        'entrenar_evaluar_modelo_pca'
    ],
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
    # Crear DataFrame para comparar métricas
    metricas = {
        'Modelo': ['Ridge', 'Lasso', 'MCO', 'PCA'],
        'MSE': [
            modelo_ridge.get_metric('avg_mse'),
            modelo_lasso.get_metric('avg_mse'),
            modelo_mco.get_metric('avg_mse'),
            modelo_pca.get_metric('avg_mse')
        ],
        'MAE': [
            modelo_ridge.get_metric('avg_mae'),
            modelo_lasso.get_metric('avg_mae'),
            modelo_mco.get_metric('avg_mae'),
            modelo_pca.get_metric('avg_mae')
        ],
        'MAPE': [
            modelo_ridge.get_metric('avg_mape'),
            modelo_lasso.get_metric('avg_mape'),
            modelo_mco.get_metric('avg_mape'),
            modelo_pca.get_metric('avg_mape')
        ]
    }
    
    df_metricas = pd.DataFrame(metricas)
    
    # Encontrar el mejor modelo para cada métrica
    mejor_mse = df_metricas.loc[df_metricas['MSE'].idxmin()]
    mejor_mae = df_metricas.loc[df_metricas['MAE'].idxmin()]
    mejor_mape = df_metricas.loc[df_metricas['MAPE'].idxmin()]
    
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
        'comparacion_completa': df_metricas.to_dict('records')
    }
    
    # Loggear resultados
    context.log.info("Mejores modelos por métrica:")
    context.log.info(f"MSE: {resultado['mejor_por_mse']['modelo']} ({resultado['mejor_por_mse']['valor']:.4f})")
    context.log.info(f"MAE: {resultado['mejor_por_mae']['modelo']} ({resultado['mejor_por_mae']['valor']:.4f})")
    context.log.info(f"MAPE: {resultado['mejor_por_mape']['modelo']} ({resultado['mejor_por_mape']['valor']:.4f}%)")
    
    return resultado
