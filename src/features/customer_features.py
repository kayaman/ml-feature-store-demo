"""
Customer feature computation logic.
All feature engineering transformations are defined here.
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, count, sum, avg, datediff, current_date, 
    when, lit, max as spark_max
)


def compute_customer_features(transactions_df: DataFrame, 
                              customers_df: DataFrame) -> DataFrame:
    """
    Compute customer features for churn prediction.
    
    Args:
        transactions_df: Raw transaction data
        customers_df: Customer demographic data
    
    Returns:
        DataFrame with customer_id as primary key and computed features
    """
    
    # Aggregate transaction features (last 30 days)
    transaction_features = transactions_df.groupBy("customer_id").agg(
        count("transaction_id").alias("total_transactions_30d"),
        sum("amount").alias("total_spend_30d"),
        avg("amount").alias("avg_transaction_value_30d"),
        spark_max("transaction_date").alias("last_transaction_date")
    )
    
    # Calculate days since last transaction
    transaction_features = transaction_features.withColumn(
        "days_since_last_transaction",
        datediff(current_date(), col("last_transaction_date"))
    )
    
    # Join with customer demographics
    features = customers_df.join(
        transaction_features, 
        on="customer_id", 
        how="left"
    )
    
    # Fill nulls for customers with no transactions
    features = features.fillna({
        "total_transactions_30d": 0,
        "total_spend_30d": 0.0,
        "avg_transaction_value_30d": 0.0,
        "days_since_last_transaction": 999
    })
    
    # Compute derived features
    features = features.withColumn(
        "is_high_value",
        when(col("total_spend_30d") > 1000, 1).otherwise(0)
    ).withColumn(
        "is_frequent_buyer",
        when(col("total_transactions_30d") >= 5, 1).otherwise(0)
    ).withColumn(
        "customer_tenure_days",
        datediff(current_date(), col("customer_since_date"))
    )
    
    # Select final feature columns
    feature_columns = [
        "customer_id",
        "age",
        "customer_tenure_days",
        "total_transactions_30d",
        "total_spend_30d",
        "avg_transaction_value_30d",
        "days_since_last_transaction",
        "is_high_value",
        "is_frequent_buyer"
    ]
    
    return features.select(feature_columns)


def validate_features(features_df: DataFrame) -> bool:
    """
    Validate feature quality before writing to feature store.
    
    Args:
        features_df: Computed features
        
    Returns:
        True if validation passes, raises exception otherwise
    """
    # Check for nulls in critical columns
    null_counts = features_df.select([
        count(when(col(c).isNull(), c)).alias(c) 
        for c in features_df.columns if c != "customer_id"
    ])
    
    null_dict = null_counts.collect()[0].asDict()
    if any(count > 0 for count in null_dict.values()):
        raise ValueError(f"Null values found in features: {null_dict}")
    
    # Check value ranges
    invalid_age = features_df.filter(
        (col("age") < 18) | (col("age") > 100)
    ).count()
    if invalid_age > 0:
        raise ValueError(f"Invalid age values found: {invalid_age} rows")
    
    # Check for negative values where not expected
    negative_spend = features_df.filter(col("total_spend_30d") < 0).count()
    if negative_spend > 0:
        raise ValueError(f"Negative spend values found: {negative_spend} rows")
    
    return True