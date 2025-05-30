from pathlib import Path
from dagster import EnvVar
from dagster_dbt import DbtCliResource, DbtProject

dbt_project = DbtProject(
    #project_dir="/home/agustin/Escritorio/_/proyectos_facu/proyectos_datamining/toyota/dbt_toyota",
    project_dir=Path(__file__).joinpath("..","..","dbt_toyota").resolve(),
    profiles_dir=Path.home().joinpath(".dbt") ,   
    
    target="dev",
    profile="dbt_toyota_1"
)

dbt_project.prepare_if_dev()