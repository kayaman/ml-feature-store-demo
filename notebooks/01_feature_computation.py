# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Synthetic Data for Feature Store
# MAGIC 
# MAGIC This notebook generates realistic synthetic data for the churn prediction use case.
# MAGIC 
# MAGIC **Generated Tables:**
# MAGIC - `raw_customers` - Customer demographics
# MAGIC - `raw_transactions` - Transaction history
# MAGIC - `customer_labels` - Churn labels for training
# MAGIC - `daily_metrics` - Time-series metrics for monitoring

# COMMAND ----------

# MAGIC %pip install faker==24.0.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Widget parameters
dbutils.widgets.dropdown("data_size", "medium", ["small", "medium", "large"], "Data Size")
dbutils.widgets.text("catalog", "ml", "Catalog")
dbutils.widgets.text("schema", "churn_raw", "Schema")
dbutils.widgets.dropdown("add_corruption", "false", ["true", "false"], "Add Data Quality Issues")

data_size = dbutils.widgets.get("data_size")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
add_corruption = dbutils.widgets.get("add_corruption") == "true"

print(f"Configuration:")
print(f"  Data size: {data_size}")
print(f"  Target: {catalog}.{schema}")
print(f"  Add corruption: {add_corruption}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Infrastructure

# COMMAND ----------

# Create catalog and schema for raw data
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
spark.sql(f"USE SCHEMA {schema}")

print(f"✅ Created {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Synthetic Data Generator

# COMMAND ----------

import sys
sys.path.append("/Workspace/Repos/your-repo-path/src")

from synthetic_data_generator import SyntheticDataGenerator, DataConfig
import pandas as pd
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Data

# COMMAND ----------

# Configuration by size
size_configs = {
    'small': DataConfig(num_customers=1_000, avg_transactions_per_customer=10, date_range_days=180),
    'medium': DataConfig(num_customers=10_000, avg_transactions_per_customer=15, date_range_days=365),
    'large': DataConfig(num_customers=100_000, avg_transactions_per_customer=20, date_range_days=730)
}

config = size_configs[data_size]
print(f"Generating {config.num_customers:,} customers...")

# Generate data
generator = SyntheticDataGenerator(config)
customers_pd, transactions_pd, labels_pd = generator.generate_all()
time_series_pd = generator.generate_time_series(customers_pd)

# Add corruption if requested
if add_corruption:
    print("\nAdding data quality issues...")
    customers_pd, transactions_pd = generator.add_data_quality_issues(
        customers_pd, transactions_pd, corruption_rate=0.03
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Report

# COMMAND ----------

print("="*60)
print("DATA QUALITY REPORT")
print("="*60)

print(f"\n📊 CUSTOMERS ({len(customers_pd):,} rows)")
print(f"  Age range: {customers_pd['age'].min():.0f} - {customers_pd['age'].max():.0f}")
print(f"  Null ages: {customers_pd['age'].isna().sum()}")
print(f"  Invalid ages: {((customers_pd['age'] < 18) | (customers_pd['age'] > 100)).sum()}")
print(f"\n  Segment distribution:")
for segment, count in customers_pd['segment'].value_counts().items():
    print(f"    {segment}: {count:,} ({count/len(customers_pd):.1%})")

print(f"\n💳 TRANSACTIONS ({len(transactions_pd):,} rows)")
print(f"  Date range: {transactions_pd['transaction_date'].min()} to {transactions_pd['transaction_date'].max()}")
print(f"  Amount range: ${transactions_pd['amount'].min():.2f} - ${transactions_pd['amount'].max():.2f}")
print(f"  Total revenue: ${transactions_pd['amount'].sum():,.2f}")
print(f"  Negative amounts: {(transactions_pd['amount'] < 0).sum()}")
print(f"  Avg txns per customer: {len(transactions_pd) / len(customers_pd):.1f}")

print(f"\n🎯 LABELS ({len(labels_pd):,} rows)")
print(f"  Churn rate: {labels_pd['churn_label'].mean():.2%}")
print(f"  Churned customers: {labels_pd['churn_label'].sum():,}")
print(f"  Active customers: {(1 - labels_pd['churn_label']).sum():,}")

print(f"\n📈 TIME SERIES ({len(time_series_pd):,} days)")
print(f"  Date range: {time_series_pd['date'].min()} to {time_series_pd['date'].max()}")
print(f"  Avg daily transactions: {time_series_pd['transaction_count'].mean():.0f}")
print(f"  Avg daily revenue: ${time_series_pd['total_revenue'].mean():,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to Spark DataFrames

# COMMAND ----------

# Remove internal segment column before saving
customers_spark = spark.createDataFrame(customers_pd.drop('segment', axis=1))
transactions_spark = spark.createDataFrame(transactions_pd)
labels_spark = spark.createDataFrame(labels_pd)
time_series_spark = spark.createDataFrame(time_series_pd)

print("✅ Converted to Spark DataFrames")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta Tables

# COMMAND ----------

# Write customers
customers_spark.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.raw_customers")

print(f"✅ Wrote {customers_spark.count():,} customers to {catalog}.{schema}.raw_customers")

# Write transactions
transactions_spark.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.raw_transactions")

print(f"✅ Wrote {transactions_spark.count():,} transactions to {catalog}.{schema}.raw_transactions")

# Write labels
labels_spark.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.customer_labels")

print(f"✅ Wrote {labels_spark.count():,} labels to {catalog}.{schema}.customer_labels")

# Write time series
time_series_spark.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{catalog}.{schema}.daily_metrics")

print(f"✅ Wrote {time_series_spark.count():,} daily metrics to {catalog}.{schema}.daily_metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Tables

# COMMAND ----------

# Optimize and collect statistics for better query performance
for table_name in ['raw_customers', 'raw_transactions', 'customer_labels', 'daily_metrics']:
    full_table_name = f"{catalog}.{schema}.{table_name}"
    spark.sql(f"OPTIMIZE {full_table_name}")
    spark.sql(f"ANALYZE TABLE {full_table_name} COMPUTE STATISTICS")
    print(f"✅ Optimized {full_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Data Preview

# COMMAND ----------

print("SAMPLE CUSTOMERS:")
display(spark.table(f"{catalog}.{schema}.raw_customers").limit(5))

print("\nSAMPLE TRANSACTIONS:")
display(spark.table(f"{catalog}.{schema}.raw_transactions").limit(10))

print("\nSAMPLE LABELS:")
display(spark.table(f"{catalog}.{schema}.customer_labels").limit(5))

print("\nSAMPLE TIME SERIES:")
display(spark.table(f"{catalog}.{schema}.daily_metrics").orderBy("date", ascending=False).limit(7))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Lineage Documentation

# COMMAND ----------

# Add table comments for documentation
spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.raw_customers IS 
    'Synthetic customer demographic data generated for churn prediction modeling. 
    Contains customer_id (PK), age, customer_since_date, and derived segment information.'
""")

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.raw_transactions IS 
    'Synthetic transaction history with realistic temporal patterns and correlations. 
    Contains transaction_id (PK), customer_id (FK), amount, and transaction_date.'
""")

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.customer_labels IS 
    'Target labels for churn prediction. Churn defined as no transactions in last 60 days. 
    Contains customer_id (PK), churn_label (0/1), and true churn_probability for analysis.'
""")

spark.sql(f"""
    COMMENT ON TABLE {catalog}.{schema}.daily_metrics IS 
    'Daily aggregated transaction metrics for monitoring and trend analysis. 
    Contains date (PK), transaction_count, total_revenue, and avg_transaction_value.'
""")

print("✅ Added table documentation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verification Queries

# COMMAND ----------

# Check data quality
print("VERIFICATION CHECKS:")
print("\n1. Primary Key Uniqueness:")
print(f"   Customers: {spark.table(f'{catalog}.{schema}.raw_customers').count()} rows, " +
      f"{spark.table(f'{catalog}.{schema}.raw_customers').select('customer_id').distinct().count()} unique IDs")

print("\n2. Referential Integrity:")
txn_customers = spark.sql(f"""
    SELECT COUNT(DISTINCT t.customer_id) as txn_customers,
           (SELECT COUNT(*) FROM {catalog}.{schema}.raw_customers) as total_customers
    FROM {catalog}.{schema}.raw_transactions t
""").collect()[0]
print(f"   Customers with transactions: {txn_customers.txn_customers} / {txn_customers.total_customers}")

print("\n3. Date Consistency:")
date_check = spark.sql(f"""
    SELECT 
        MIN(transaction_date) as first_txn,
        MAX(transaction_date) as last_txn,
        DATEDIFF(MAX(transaction_date), MIN(transaction_date)) as date_span_days
    FROM {catalog}.{schema}.raw_transactions
""").collect()[0]
print(f"   Transaction date range: {date_check.first_txn} to {date_check.last_txn} ({date_check.date_span_days} days)")

print("\n4. Label Coverage:")
label_coverage = spark.sql(f"""
    SELECT 
        COUNT(*) as customers_with_labels,
        SUM(churn_label) as churned_count,
        AVG(churn_label) as churn_rate
    FROM {catalog}.{schema}.customer_labels
""").collect()[0]
print(f"   Labeled customers: {label_coverage.customers_with_labels}")
print(f"   Churned: {label_coverage.churned_count} ({label_coverage.churn_rate:.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("="*60)
print("✅ SYNTHETIC DATA GENERATION COMPLETE")
print("="*60)
print(f"\nGenerated tables in {catalog}.{schema}:")
print(f"  1. raw_customers ({customers_spark.count():,} rows)")
print(f"  2. raw_transactions ({transactions_spark.count():,} rows)")
print(f"  3. customer_labels ({labels_spark.count():,} rows)")
print(f"  4. daily_metrics ({time_series_spark.count():,} rows)")
print("\nNext steps:")
print(f"  1. Update feature computation notebook to read from {catalog}.{schema}")
print("  2. Run feature engineering pipeline")
print("  3. Train churn prediction model")
print("="*60)