from dagster import AssetExecutionContext
from dagster_dbt import dbt_assets, DbtCliResource
from parcial_toyota.dbt_integrations import dbt_project

@dbt_assets(
    manifest=dbt_project.manifest_path,
 
)
def dbt_analytics(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["run"], context=context).stream()