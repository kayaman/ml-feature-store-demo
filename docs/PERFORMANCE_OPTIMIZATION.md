# Performance Optimization for Feature Store

## Overview

This guide covers performance optimization techniques for scaling your Feature Store implementation from prototype to production with millions of features and real-time serving requirements.

## Feature Computation Performance

### 1. Delta Lake Optimization

```python
# Enable auto-optimize and auto-compaction
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

# Z-ordering for frequently filtered columns
spark.sql("""
    OPTIMIZE ml.churn_features.customer_features
    ZORDER BY (customer_id, days_since_last_transaction)
""")

# Liquid clustering (preferred for DBR 14+)
spark.sql("""
    ALTER TABLE ml.churn_features.customer_features
    CLUSTER BY (customer_id)
""")
```

### 2. Incremental Feature Computation

```python
"""
Process only changed data instead of full recomputation.
"""
from delta.tables import DeltaTable
from pyspark.sql.functions import col, current_timestamp

def compute_features_incremental(catalog: str, schema: str):
    """Compute features incrementally using CDC."""
    
    # Get last processed timestamp
    feature_table = f"{catalog}.{schema}.customer_features"
    
    try:
        last_update = spark.sql(f"""
            SELECT MAX(_update_timestamp) as last_ts
            FROM {feature_table}
        """).collect()[0].last_ts
    except:
        last_update = None
    
    # Read only new/updated transactions
    if last_update:
        new_transactions = spark.table(f"{catalog}.churn_raw.raw_transactions") \
            .filter(col("_change_timestamp") > last_update)
    else:
        # First run: process all
        new_transactions = spark.table(f"{catalog}.churn_raw.raw_transactions")
    
    print(f"Processing {new_transactions.count()} new/updated transactions")
    
    # Compute features only for affected customers
    affected_customers = new_transactions.select("customer_id").distinct()
    
    # Recompute features for these customers
    customers = spark.table(f"{catalog}.churn_raw.raw_customers") \
        .join(affected_customers, "customer_id")
    
    # Compute features
    features_df = compute_customer_features(new_transactions, customers)
    features_df = features_df.withColumn("_update_timestamp", current_timestamp())
    
    # Merge into feature store
    fe = FeatureEngineeringClient()
    fe.write_table(
        name=feature_table,
        df=features_df,
        mode="merge"
    )
    
    print(f"✅ Updated features for {features_df.count()} customers")
```

### 3. Photon Engine Acceleration

```yaml
# databricks.yml - Enable Photon
job_clusters:
  - job_cluster_key: feature_cluster
    new_cluster:
      spark_version: "16.4.x-photon-scala2.12"  # Photon-enabled
      node_type_id: "Standard_DS3_v2"
      num_workers: 4
      runtime_engine: "PHOTON"  # Explicit Photon
```

### 4. Partition Strategy

```python
# Partition feature tables by date for time-series features
spark.sql("""
    CREATE TABLE ml.churn_features.customer_features_partitioned (
        customer_id INT,
        feature_date DATE,
        -- other columns
    )
    USING DELTA
    PARTITIONED BY (feature_date)
    CLUSTER BY (customer_id)
""")

# Write with partitioning
features_df.write \
    .partitionBy("feature_date") \
    .mode("append") \
    .saveAsTable("ml.churn_features.customer_features_partitioned")
```

## Online Feature Serving Performance

### 1. Online Store Configuration

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

w = WorkspaceClient()

# Optimized online table with fast refresh
spec = OnlineTableSpec(
    primary_key_columns=["customer_id"],
    source_table_full_name=f"{catalog}.{schema}.customer_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({
        "triggered": {
            "refresh_interval": "5 minutes"  # Faster refresh for real-time needs
        }
    }),
    # Enable performance features
    timeseries_key="feature_timestamp",  # If time-series
    perform_full_copy=False  # Incremental only
)

online_table = w.online_tables.create(
    name=f"{catalog}.{schema}_online.customer_features",
    spec=spec
)
```

### 2. Feature Serving Endpoint Optimization

```python
# Create high-performance serving endpoint
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

endpoint_config = EndpointCoreConfigInput(
    name="churn-prediction-realtime",
    served_entities=[
        ServedEntityInput(
            entity_name=f"{catalog}.{schema}.churn_model",
            entity_version="1",
            workload_size="Small",  # Start small, scale as needed
            scale_to_zero_enabled=False  # Keep warm for low latency
        )
    ],
    # Traffic optimization
    traffic_config={
        "routes": [{
            "served_model_name": "churn_model-1",
            "traffic_percentage": 100
        }]
    }
)
```

### 3. Caching Strategy

```python
"""
Cache frequently accessed features in-memory.
"""
from functools import lru_cache
import redis

class FeatureCache:
    """Redis cache for hot features."""
    
    def __init__(self, redis_host: str, redis_port: int = 6379):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.ttl_seconds = 300  # 5 minutes
    
    def get_features(self, customer_id: int) -> dict:
        """Get features with cache."""
        cache_key = f"features:customer:{customer_id}"
        
        # Try cache first
        cached = self.redis_client.get(cache_key)
        if cached:
            return eval(cached)  # In production, use json.loads
        
        # Cache miss: fetch from feature store
        features = self._fetch_from_feature_store(customer_id)
        
        # Store in cache
        self.redis_client.setex(
            cache_key,
            self.ttl_seconds,
            str(features)
        )
        
        return features
    
    def _fetch_from_feature_store(self, customer_id: int) -> dict:
        """Fetch from Databricks online store."""
        # Implementation depends on your online store setup
        pass
```

## Model Training Performance

### 1. Distributed Training with Spark

```python
from sklearn.ensemble import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.feature import VectorAssembler

def train_distributed(training_df, feature_cols: list, label_col: str):
    """Train using Spark ML for large datasets."""
    
    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    
    training_assembled = assembler.transform(training_df)
    
    # Spark RandomForest (distributed)
    rf = SparkRF(
        featuresCol="features",
        labelCol=label_col,
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    
    # Train in parallel across cluster
    model = rf.fit(training_assembled)
    
    return model
```

### 2. Feature Store Training Set Caching

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create and cache training set
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churn_label"
)

# Cache for multiple training runs
training_df = training_set.load_df()
training_df.cache()  # Cache in memory
training_df.count()  # Materialize cache

# Now train multiple models without re-loading
for param in hyperparameters:
    model = train_model(training_df, param)
```

### 3. Hyperparameter Tuning at Scale

```python
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt import SparkTrials

def objective(params):
    """Objective function for hyperopt."""
    model = LogisticRegression(
        C=params['C'],
        max_iter=int(params['max_iter'])
    )
    
    # Train and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    return {'loss': -score, 'status': STATUS_OK}

# Distributed hyperparameter search
spark_trials = SparkTrials(parallelism=4)

best = fmin(
    fn=objective,
    space={
        'C': hp.loguniform('C', -5, 2),
        'max_iter': hp.quniform('max_iter', 100, 1000, 100)
    },
    algo=tpe.suggest,
    max_evals=100,
    trials=spark_trials
)
```

## Monitoring and Observability

### 1. Feature Computation Metrics

```python
"""
Track feature computation performance.
"""
import time
from pyspark.sql.functions import current_timestamp

class FeatureComputationMonitor:
    """Monitor feature computation performance."""
    
    def __init__(self, catalog: str, schema: str):
        self.catalog = catalog
        self.schema = schema
        self.metrics_table = f"{catalog}.{schema}.feature_computation_metrics"
        self._init_metrics_table()
    
    def _init_metrics_table(self):
        """Create metrics table if not exists."""
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.metrics_table} (
                run_id STRING,
                run_timestamp TIMESTAMP,
                num_customers_processed LONG,
                num_features_computed LONG,
                duration_seconds DOUBLE,
                rows_per_second DOUBLE,
                status STRING,
                error_message STRING
            )
            USING DELTA
        """)
    
    def log_computation(
        self,
        run_id: str,
        num_customers: int,
        num_features: int,
        duration: float,
        status: str = "success",
        error: str = None
    ):
        """Log computation metrics."""
        metrics_df = spark.createDataFrame([{
            'run_id': run_id,
            'run_timestamp': datetime.now(),
            'num_customers_processed': num_customers,
            'num_features_computed': num_features,
            'duration_seconds': duration,
            'rows_per_second': num_customers / duration if duration > 0 else 0,
            'status': status,
            'error_message': error
        }])
        
        metrics_df.write.mode("append").saveAsTable(self.metrics_table)

# Usage
monitor = FeatureComputationMonitor(catalog, schema)

start_time = time.time()
try:
    features_df = compute_customer_features(transactions_df, customers_df)
    duration = time.time() - start_time
    
    monitor.log_computation(
        run_id=run_id,
        num_customers=features_df.count(),
        num_features=len(features_df.columns),
        duration=duration,
        status="success"
    )
except Exception as e:
    duration = time.time() - start_time
    monitor.log_computation(
        run_id=run_id,
        num_customers=0,
        num_features=0,
        duration=duration,
        status="failed",
        error=str(e)
    )
    raise
```

### 2. Feature Drift Detection

```python
"""
Monitor feature distribution drift over time.
"""
from scipy.stats import ks_2samp
import numpy as np

class FeatureDriftMonitor:
    """Detect statistical drift in features."""
    
    def __init__(self, baseline_df):
        """Initialize with baseline feature distributions."""
        self.baseline_stats = self._compute_stats(baseline_df)
    
    def _compute_stats(self, df):
        """Compute distribution statistics."""
        stats = {}
        for col in df.columns:
            if col != 'customer_id':
                col_data = df.select(col).toPandas()[col]
                stats[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'p50': col_data.median(),
                    'p95': col_data.quantile(0.95)
                }
        return stats
    
    def detect_drift(self, current_df, threshold: float = 0.05):
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            current_df: Current features
            threshold: P-value threshold for drift detection
        
        Returns:
            Dict of drifted features with p-values
        """
        drifted_features = {}
        
        baseline_df = spark.table(self.baseline_table)
        
        for col in current_df.columns:
            if col == 'customer_id':
                continue
            
            baseline_values = baseline_df.select(col).toPandas()[col]
            current_values = current_df.select(col).toPandas()[col]
            
            # KS test
            statistic, pvalue = ks_2samp(baseline_values, current_values)
            
            if pvalue < threshold:
                drifted_features[col] = {
                    'pvalue': pvalue,
                    'ks_statistic': statistic,
                    'baseline_mean': baseline_values.mean(),
                    'current_mean': current_values.mean(),
                    'drift_pct': abs(
                        (current_values.mean() - baseline_values.mean()) 
                        / baseline_values.mean()
                    )
                }
        
        return drifted_features

# Usage
drift_monitor = FeatureDriftMonitor(baseline_features_df)
drift_results = drift_monitor.detect_drift(current_features_df)

if drift_results:
    print("⚠️ DRIFT DETECTED:")
    for feature, metrics in drift_results.items():
        print(f"  {feature}: {metrics['drift_pct']:.1%} drift (p={metrics['pvalue']:.4f})")
```

### 3. Real-time Performance Dashboard

```sql
-- Create dashboard queries for monitoring

-- 1. Feature Computation Performance (Last 7 days)
CREATE OR REPLACE VIEW feature_computation_performance AS
SELECT 
    DATE(run_timestamp) as run_date,
    COUNT(*) as num_runs,
    AVG(duration_seconds) as avg_duration_sec,
    AVG(rows_per_second) as avg_throughput,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs
FROM ml.churn_features.feature_computation_metrics
WHERE run_timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
GROUP BY DATE(run_timestamp)
ORDER BY run_date DESC;

-- 2. Feature Freshness
CREATE OR REPLACE VIEW feature_freshness AS
SELECT 
    MAX(days_since_last_transaction) as oldest_activity_days,
    COUNT(*) as total_customers,
    SUM(CASE WHEN days_since_last_transaction > 60 THEN 1 ELSE 0 END) as stale_features,
    SUM(CASE WHEN days_since_last_transaction > 60 THEN 1 ELSE 0 END) / COUNT(*) as stale_pct
FROM ml.churn_features.customer_features;

-- 3. Online Store Sync Status
SELECT 
    table_name,
    sync_status,
    last_sync_time,
    TIMESTAMPDIFF(MINUTE, last_sync_time, CURRENT_TIMESTAMP) as minutes_since_sync
FROM system.information_schema.online_tables
WHERE catalog_name = 'ml';
```

## Performance Benchmarks

### Target Metrics

| Operation | Target | Notes |
|-----------|--------|-------|
| Feature Computation | <5 min | 10k customers, 10 features |
| Incremental Update | <1 min | 1k changed customers |
| Training Set Creation | <30 sec | 10k samples with feature lookup |
| Batch Inference | <2 min | 10k predictions |
| Online Feature Lookup | <10ms p99 | Single customer lookup |
| Model Serving | <50ms p99 | Including feature lookup |

### Optimization Checklist

- [ ] Enable Photon engine for all compute clusters
- [ ] Configure auto-optimize and auto-compaction
- [ ] Implement incremental feature computation
- [ ] Use liquid clustering on feature tables
- [ ] Cache training sets during hyperparameter tuning
- [ ] Enable scale-to-zero for dev serving endpoints
- [ ] Disable scale-to-zero for production serving
- [ ] Implement feature caching for hot paths
- [ ] Set up online store with 5-minute refresh
- [ ] Monitor feature computation metrics
- [ ] Track feature drift weekly
- [ ] Optimize SQL queries with explain plans
- [ ] Partition large tables by date
- [ ] Use broadcast joins for small dimension tables

## Cost Optimization

### 1. Cluster Right-Sizing

```python
# Use smallest sufficient cluster
# Development: 2 workers
# Staging: 4 workers  
# Production: 8+ workers with autoscaling

job_clusters:
  - job_cluster_key: prod_cluster
    new_cluster:
      autoscale:
        min_workers: 2
        max_workers: 16  # Scale based on load
      spot_bid_price_percent: 80  # Use spot instances for cost savings
```

### 2. Scheduled Job Optimization

```yaml
# Run expensive jobs during off-peak hours
schedule:
  quartz_cron_expression: "0 0 2 * * ?"  # 2 AM UTC
  timezone_id: "UTC"
  pause_status: UNPAUSED
```

### 3. Storage Optimization

```python
# Vacuum old versions to save storage
spark.sql("""
    VACUUM ml.churn_features.customer_features
    RETAIN 168 HOURS  -- 7 days
""")

# Shallow clone for dev/test environments
spark.sql("""
    CREATE TABLE ml_dev.churn_features.customer_features
    SHALLOW CLONE ml.churn_features.customer_features
""")
```

## Troubleshooting Performance Issues

### Issue: Slow Feature Computation

**Diagnosis:**
```python
# Check query plan
features_df.explain(mode="cost")

# Identify shuffle partitions
spark.sql("SET spark.sql.shuffle.partitions")
```

**Solution:**
```python
# Optimize shuffle partitions based on data size
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Adjust based on data

# Enable adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

### Issue: Online Store Lag

**Diagnosis:**
```sql
SELECT 
    MAX(feature_timestamp) as latest_feature,
    CURRENT_TIMESTAMP as now,
    TIMESTAMPDIFF(MINUTE, MAX(feature_timestamp), CURRENT_TIMESTAMP) as lag_minutes
FROM ml.churn_features_online.customer_features;
```

**Solution:**
- Reduce refresh interval from 1 hour to 5-15 minutes
- Enable incremental-only updates
- Check source table update frequency

### Issue: High Model Serving Latency

**Diagnosis:**
```python
# Profile endpoint performance
import requests
import time

times = []
for _ in range(100):
    start = time.time()
    response = requests.post(endpoint_url, json=payload)
    times.append(time.time() - start)

print(f"P50: {np.percentile(times, 50)*1000:.2f}ms")
print(f"P95: {np.percentile(times, 95)*1000:.2f}ms")
print(f"P99: {np.percentile(times, 99)*1000:.2f}ms")
```

**Solution:**
- Disable scale-to-zero for production
- Increase workload size (Small → Medium)
- Implement feature caching layer
- Reduce number of features used