# Databricks notebook source
# MAGIC %md
# MAGIC # Publish Features to Online Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.8.0 databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

feature_table_name = f"{catalog}.{schema}.customer_features"
online_table_name = f"{catalog}.{schema}_online.customer_features"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Online Schema

# COMMAND ----------

# Create online schema for online tables
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}_online")
print(f"Created online schema: {catalog}.{schema}_online")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Online Table

# COMMAND ----------

w = WorkspaceClient()

# Create online table specification
spec = OnlineTableSpec(
    primary_key_columns=["customer_id"],
    source_table_full_name=feature_table_name,
    run_triggered={
        "refresh_interval": "1 hour"  # Refresh every hour
    }
)

try:
    online_table = w.online_tables.create(
        name=online_table_name,
        spec=spec
    )
    print(f"✅ Created online table: {online_table_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Online table already exists: {online_table_name}")
        # Get existing table
        online_table = w.online_tables.get(online_table_name)
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Online Table Status

# COMMAND ----------

# Check online table status
status = w.online_tables.get(online_table_name)

print(f"Online Table: {status.name}")
print(f"Status: {status.status}")
print(f"Source: {status.spec.source_table_full_name}")
print(f"Primary Keys: {status.spec.primary_key_columns}")

print("\n✅ Online feature store setup complete!")
print("Features will be automatically refreshed every hour")