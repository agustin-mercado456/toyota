from dagster import Definitions, fs_io_manager, load_assets_from_modules , define_asset_job , AssetSelection
from dagster_mlflow import mlflow_tracking
from dagstermill import ConfigurableLocalOutputNotebookIOManager
from parcial_toyota.assets import clean_train_model , dbt , seleccion_models
from parcial_toyota.resource import  dbt_resource, postgres_io_manager
from parcial_toyota.assets.airbyte import airbyte_connections

#--------------------------------Assets Definitions--------------------------------   
dbt_assets = load_assets_from_modules([dbt], group_name="RAW_DATA_PREPARATION")

clean_train_model_assets = load_assets_from_modules([clean_train_model])

model_seleccion_assets = load_assets_from_modules([seleccion_models])
#--------------------------------Jobs Definitions--------------------------------  

airbyte_sync_job = define_asset_job(name="airbyte_sync_job",  selection=AssetSelection.groups("RAW_DATA_INGESTION"))

dbt_sync_job = define_asset_job("dbt_sync_job" , selection=AssetSelection.groups("RAW_DATA_PREPARATION"))

data_prep_job = define_asset_job("data_prep_job", selection=AssetSelection.groups("RAW_DATA_PREPARATION"))

sync_all_jobs = define_asset_job(name="sync_all_jobs", selection=AssetSelection.all())


#--------------------------------Definitions-------------------------------- 


defs = Definitions(
    assets=[airbyte_connections, *dbt_assets, *clean_train_model_assets, *model_seleccion_assets],
    jobs=[airbyte_sync_job, dbt_sync_job, data_prep_job, sync_all_jobs],
    resources={
        "dbt": dbt_resource, 
        "postgres_io_manager": postgres_io_manager.configured({
            "connection_string": "env:POSTGRES_CONNECTION_STRING",
            "schema": "target"}),
        "io_manager": fs_io_manager,
        "mlflow": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota_corolla_experiment",
            
        }),
    
        "mlflow_toyota_ridge": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota_ridge",
        }),
        "mlflow_toyota_lasso": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
                "experiment_name": "toyota_lasso",
            }),
        "mlflow_toyota_pca": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota_pca",
        }),
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
    },
)