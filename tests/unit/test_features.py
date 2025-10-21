"""
Unit tests for feature computation logic.
"""
import pytest
from pyspark.sql import SparkSession
from datetime import date, timedelta
from pyspark.sql import Row


@pytest.fixture(scope="session")
def spark():
    """Create Spark session for tests."""
    return SparkSession.builder \
        .appName("feature-tests") \
        .master("local[2]") \
        .getOrCreate()


def test_compute_customer_features_basic(spark):
    """Test basic feature computation."""
    from src.features.customer_features import compute_customer_features
    
    # Create test data
    customers_data = [
        Row(customer_id=1, age=25, customer_since_date=date(2023, 1, 1))
    ]
    customers_df = spark.createDataFrame(customers_data)
    
    transactions_data = [
        Row(transaction_id=1, customer_id=1, amount=100.0, 
            transaction_date=date.today() - timedelta(days=1))
    ]
    transactions_df = spark.createDataFrame(transactions_data)
    
    # Compute features
    features_df = compute_customer_features(transactions_df, customers_df)
    
    # Assertions
    assert features_df.count() == 1
    assert "customer_id" in features_df.columns
    assert "total_transactions_30d" in features_df.columns
    
    result = features_df.collect()[0]
    assert result.customer_id == 1
    assert result.total_transactions_30d == 1
    assert result.total_spend_30d == 100.0


def test_feature_validation_passes(spark):
    """Test feature validation with valid data."""
    from src.features.customer_features import validate_features
    
    valid_data = [
        Row(customer_id=1, age=30, customer_tenure_days=365,
            total_transactions_30d=5, total_spend_30d=500.0,
            avg_transaction_value_30d=100.0, days_since_last_transaction=2,
            is_high_value=0, is_frequent_buyer=1)
    ]
    features_df = spark.createDataFrame(valid_data)
    
    assert validate_features(features_df)


def test_feature_validation_fails_on_invalid_age(spark):
    """Test feature validation fails with invalid age."""
    from src.features.customer_features import validate_features
    
    invalid_data = [
        Row(customer_id=1, age=150, customer_tenure_days=365,
            total_transactions_30d=5, total_spend_30d=500.0,
            avg_transaction_value_30d=100.0, days_since_last_transaction=2,
            is_high_value=0, is_frequent_buyer=1)
    ]
    features_df = spark.createDataFrame(invalid_data)
    
    with pytest.raises(ValueError, match="Invalid age"):
        validate_features(features_df)