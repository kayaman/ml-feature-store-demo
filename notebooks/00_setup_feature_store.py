# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Feature Store Infrastructure
# MAGIC One-time setup to create catalog, schema, and initial feature tables

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.8.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.types import (
    StructType, StructField, IntegerType, 
    LongType, DoubleType, StringType
)

# Get parameters
catalog = dbutils.widgets.get("catalog") if dbutils.widgets.get("catalog") else "ml"
schema = dbutils.widgets.get("schema") if dbutils.widgets.get("schema") else "churn_features"

print(f"Setting up Feature Store in {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Catalog and Schema

# COMMAND ----------

# Create catalog (if not exists)
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"USE CATALOG {catalog}")

# Create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")

print(f"Created catalog and schema: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Feature Table

# COMMAND ----------

fe = FeatureEngineeringClient()

# Define feature table schema
feature_schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("age", IntegerType(), True),
    StructField("customer_tenure_days", IntegerType(), True),
    StructField("total_transactions_30d", LongType(), True),
    StructField("total_spend_30d", DoubleType(), True),
    StructField("avg_transaction_value_30d", DoubleType(), True),
    StructField("days_since_last_transaction", IntegerType(), True),
    StructField("is_high_value", IntegerType(), True),
    StructField("is_frequent_buyer", IntegerType(), True)
])

# Create feature table
try:
    feature_table = fe.create_table(
        name=f"{catalog}.{schema}.customer_features",
        primary_keys=["customer_id"],
        schema=feature_schema,
        description="Customer features for churn prediction model"
    )
    print(f"Created feature table: {catalog}.{schema}.customer_features")
except Exception as e:
    if "already exists" in str(e):
        print(f"Feature table already exists: {catalog}.{schema}.customer_features")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Setup

# COMMAND ----------

# Check table exists
tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
print(f"Tables in {catalog}.{schema}:")
for table in tables:
    print(f"  - {table.tableName}")

# Get feature table metadata
metadata = fe.get_table(name=f"{catalog}.{schema}.customer_features")
print("\nFeature Table Metadata:")
print(f"  Primary Keys: {metadata.primary_keys}")
print(f"  Features: {[f.name for f in metadata.features]}")

print("\n✅ Setup complete!")