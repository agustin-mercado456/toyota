from dagster import EnvVar, IOManager, io_manager
from dagster_airbyte import AirbyteResource
from dagster_dbt import DbtCliResource, DbtProject
import sqlalchemy
import pandas as pd
import os
from pathlib import Path

airbyte_resource = AirbyteResource(
    host=EnvVar("AIRBYTE_HOST"),
    port=EnvVar("AIRBYTE_PORT"),
    username=EnvVar("AIRBYTE_USERNAME"),
    password=EnvVar("AIRBYTE_PASSWORD")
)


dbt_resource = DbtCliResource(    
   
    project_dir=Path(__file__).joinpath("..",".." ,"..","dbt_toyota").resolve(),
    profiles_dir=Path.home().joinpath(".dbt"),
    #profiles_dir=Path("/home/agustin/.dbt"),
    target="dev",
    profile="dbt_toyota_1"       
)


class PostgresIOManager(IOManager):
    def __init__(self, engine, schema="target"):
        self.engine = engine
        self.schema = schema

    def load_input(self, context):
      
        table_name = context.upstream_output.asset_key.path[-1]
        full_table_name = f"{self.schema}.{table_name}"
        query = f"SELECT * FROM {full_table_name}"
        return pd.read_sql(query, self.engine)

    def handle_output(self, context, obj):
        pass

@io_manager(config_schema={"connection_string": str, "schema": str})
def postgres_io_manager(init_context):
    connection_string = init_context.resource_config["connection_string"]

  
    if connection_string.startswith("env:"):
        env_var = connection_string.split("env:")[1]
        connection_string = os.environ[env_var]  

    schema = init_context.resource_config.get("schema", "target")  
    engine = sqlalchemy.create_engine(connection_string)
    return PostgresIOManager(engine, schema)