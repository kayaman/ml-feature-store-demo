# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference with Feature Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.8.0 mlflow>=2.14.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import Row

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_uri = dbutils.widgets.get("model_uri")  # e.g., "models:/ml.churn_features.churn_model/1"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Batch Data

# COMMAND ----------

# Batch data only needs primary keys
# Features are automatically looked up from Feature Store

batch_data = [
    Row(customer_id=1),
    Row(customer_id=2),
    Row(customer_id=3),
    Row(customer_id=4),
    Row(customer_id=5)
]
batch_df = spark.createDataFrame(batch_data)

print(f"Batch size: {batch_df.count()} customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score Batch

# COMMAND ----------

fe = FeatureEngineeringClient()

# Features are automatically retrieved and predictions generated
predictions_df = fe.score_batch(
    model_uri=model_uri,
    df=batch_df
)

print("Predictions:")
predictions_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Predictions

# COMMAND ----------

# Write predictions to Delta table
predictions_df.write.mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.churn_predictions"
)

print(f"✅ Predictions saved to {catalog}.{schema}.churn_predictions")