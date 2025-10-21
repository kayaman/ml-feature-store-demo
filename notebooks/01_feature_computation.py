# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Computation Pipeline
# MAGIC Computes customer features and writes to Feature Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.8.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
sys.path.append("/Workspace/Repos/your-repo-path/src")

from databricks.feature_engineering import FeatureEngineeringClient
from features.customer_features import compute_customer_features, validate_features
from datetime import datetime, timedelta

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

print(f"Computing features for {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Raw Data

# COMMAND ----------

# In production, load from actual data sources
# For demo, create sample data

from pyspark.sql import Row
from datetime import date

# Sample customer data
customers_data = [
    Row(customer_id=1, age=25, customer_since_date=date(2023, 1, 15)),
    Row(customer_id=2, age=34, customer_since_date=date(2022, 6, 20)),
    Row(customer_id=3, age=45, customer_since_date=date(2021, 3, 10)),
    Row(customer_id=4, age=29, customer_since_date=date(2023, 8, 5)),
    Row(customer_id=5, age=52, customer_since_date=date(2020, 11, 25))
]
customers_df = spark.createDataFrame(customers_data)

# Sample transaction data (last 30 days)
transactions_data = [
    Row(transaction_id=1, customer_id=1, amount=150.0, transaction_date=date.today() - timedelta(days=2)),
    Row(transaction_id=2, customer_id=1, amount=200.0, transaction_date=date.today() - timedelta(days=5)),
    Row(transaction_id=3, customer_id=2, amount=500.0, transaction_date=date.today() - timedelta(days=1)),
    Row(transaction_id=4, customer_id=2, amount=750.0, transaction_date=date.today() - timedelta(days=10)),
    Row(transaction_id=5, customer_id=3, amount=1200.0, transaction_date=date.today() - timedelta(days=3)),
    Row(transaction_id=6, customer_id=5, amount=80.0, transaction_date=date.today() - timedelta(days=25))
]
transactions_df = spark.createDataFrame(transactions_data)

print(f"Loaded {customers_df.count()} customers")
print(f"Loaded {transactions_df.count()} transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Features

# COMMAND ----------

# Compute features using our feature engineering logic
features_df = compute_customer_features(transactions_df, customers_df)

print(f"Computed features for {features_df.count()} customers")
features_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Features

# COMMAND ----------

# Run validation checks
try:
    validate_features(features_df)
    print("✅ Feature validation passed")
except Exception as e:
    print(f"❌ Feature validation failed: {str(e)}")
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Feature Store

# COMMAND ----------

fe = FeatureEngineeringClient()

# Write features (merge mode for upserts)
fe.write_table(
    name=f"{catalog}.{schema}.customer_features",
    df=features_df,
    mode="merge"
)

print(f"✅ Features written to {catalog}.{schema}.customer_features")

# Verify
updated_table = spark.table(f"{catalog}.{schema}.customer_features")
print(f"Total rows in feature table: {updated_table.count()}")