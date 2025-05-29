from dagster_airbyte import load_assets_from_airbyte_instance
from parcial_toyota.resource import airbyte_resource

#airbyte_assets = load_assets_from_airbyte_instance(airbyte_resource)

airbyte_connections = load_assets_from_airbyte_instance(airbyte_resource, 
                                                        key_prefix="toyota_raw_data", 
                                                        connection_to_group_fn=lambda connection_name: "RAW_DATA_INGESTION")