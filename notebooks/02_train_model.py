# Databricks notebook source
# MAGIC %md
# MAGIC # Train Churn Model with Feature Store

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.8.0 mlflow>=2.14.0 scikit-learn>=1.5.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
sys.path.append("/Workspace/Repos/your-repo-path/src")

from models.churn_model import ChurnModel
from pyspark.sql import Row
import pandas as pd

# Get parameters
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
experiment_name = dbutils.widgets.get("experiment_name")

print(f"Training model with features from {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Training Labels

# COMMAND ----------

# In production, load actual churn labels
# For demo, create sample labels

labels_data = [
    Row(customer_id=1, churn_label=0),
    Row(customer_id=2, churn_label=0),
    Row(customer_id=3, churn_label=1),
    Row(customer_id=4, churn_label=0),
    Row(customer_id=5, churn_label=1)
]
labels_df = spark.createDataFrame(labels_data)

print(f"Created labels for {labels_df.count()} customers")
labels_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

# Initialize model trainer
model_trainer = ChurnModel(catalog=catalog, schema=schema)

# Create training set with feature lookups
training_set, training_df = model_trainer.create_training_set(labels_df)

print(f"Training set shape: {training_df.shape}")
print(f"Features: {list(training_df.columns)}")

# Train model and log to MLflow
metrics = model_trainer.train(
    training_set=training_set,
    training_df=training_df,
    experiment_name=experiment_name
)

print("\n✅ Model training complete!")
print(f"Metrics: {metrics}")