from dagster import Definitions, fs_io_manager, load_assets_from_modules
from dagster_mlflow import mlflow_tracking
from dagstermill import ConfigurableLocalOutputNotebookIOManager

from parcial_toyota.assets import clean_train_model  # noqa: TID252

all_assets = load_assets_from_modules([clean_train_model])

defs = Definitions(
    assets=all_assets,
    resources={
        "io_manager": fs_io_manager,
        "mlflow": mlflow_tracking.configured({
            "mlflow_tracking_uri": "http://localhost:5000",
            "experiment_name": "toyota_corolla_experiment",
            
        }),
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
    },
)