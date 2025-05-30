# Toyota Price Prediction Project

Este proyecto implementa un sistema de predicción de precios para vehículos Toyota utilizando diferentes modelos de regresión y técnicas de machine learning.

## Configuración del Entorno

### Docker Compose
El proyecto incluye una configuración de Docker Compose para la base de datos y pgAdmin:
```
docker-compose.yml
```

Servicios incluidos:
- PostgreSQL: Base de datos principal
- pgAdmin: Interfaz web para administración de la base de datos

Para iniciar los servicios:
```bash
docker-compose up -d
```

### Conda Environment
El archivo de entorno conda se encuentra en la raíz del proyecto:
```
environment.yml
```

Para crear y activar el entorno:
```bash
# Crear el entorno
conda env create -f environment.yml

# Activar el entorno
conda activate toyota

# Instalar dependencias con Poetry
poetry install
```

### DBT Profile
El perfil de DBT se encuentra en:
```
dbt/profiles.yml
```

## Documentación Detallada

Para una explicación detallada del proceso de limpieza de datos y entrenamiento de modelos, consulta el notebook en:
`parcial_toyota/documentacion/`

El notebook contiene:
- Proceso completo de limpieza y preprocesamiento de datos
- Análisis exploratorio de datos
- Detalles del entrenamiento de cada modelo
- Visualizaciones y resultados detallados

### Resultados y Conclusiones
En el notebook de documentación encontrarás:
- Comparación detallada de los modelos implementados
- Métricas de rendimiento para cada modelo
- Análisis de las variables más importantes
- Conclusiones sobre el mejor modelo y su rendimiento
- Recomendaciones para futuras mejoras

## Estructura del Proyecto

El proyecto está organizado en assets que manejan diferentes aspectos del pipeline de datos y modelado:

### Assets Principales

1. **Ingestión de Datos**
   - `cargar_datos`: Carga el dataset inicial desde la fuente de datos al origen

2. **Preparación de Datos**
   - `preparar_datos`: Eda y limpieza de datos
   - `transformar_datos`: Aplica transformaciones necesarias para el modelado

3. **Entrenamiento de Modelos**
   - `entrenar_evalular_modelo_ridge`: Implementa regresión Ridge
   - `entrenar_evalular_modelo_lasso`: Implementa regresión Lasso
   - `entrenar_modelo_evaluar_mco`: Implementa regresión MCO (Mínimos Cuadrados Ordinarios)
   - `entrenar_evaluar_modelo_pca`: Implementa regresión con PCA

   Cada modelo es entrenado utilizando validación cruzada con k=5 folds y sus resultados son registrados en MLflow como experimentos separados.

4. **Selección de Modelos**
   - `seleccion_modelo`: Compara y selecciona el mejor modelo basado en múltiples métricas

## Métricas de Evaluación

Los modelos son evaluados utilizando las siguientes métricas:
- MSE (Error Cuadrático Medio)
- MAE (Error Absoluto Medio)
- MAPE (Error Porcentual Absoluto Medio)
- R² (Coeficiente de Determinación)
- RMSE (Raíz del Error Cuadrático Medio)

## Integración con MLflow

El proyecto utiliza MLflow para el seguimiento de experimentos y modelos:

### Seguimiento de Métricas
- Registro de métricas de rendimiento para cada modelo
- Comparación automática de modelos
- Almacenamiento de artefactos y resultados

### Características de MLflow
- Registro de métricas de rendimiento
- Almacenamiento de parámetros de modelos
- Seguimiento de experimentos
- Visualización de resultados

### Experimentos en MLflow
Cada modelo tiene su propio experimento en MLflow donde se registran:
- Métricas de cada fold de la validación cruzada
- Parámetros del modelo
- Artefactos relevantes
- Resultados finales del modelo

## Uso

1. Asegúrate de tener todas las dependencias instaladas
2. Ejecuta el pipeline de Dagster
3. Los resultados y métricas se registrarán automáticamente en MLflow



## Notas
- Los modelos se entrenan y evalúan automáticamente
- Los resultados se registran en MLflow para fácil comparación
- Se selecciona automáticamente el mejor modelo basado en múltiples métricas

