# Production Deployment Checklist & Best Practices

## Pre-Production Checklist

### Infrastructure & Security

- [ ] **Unity Catalog Setup**
  - [ ] Production catalog created with proper naming convention
  - [ ] Schemas organized by domain (raw, features, models, monitoring)
  - [ ] Access controls configured using groups, not individual users
  - [ ] Service principals created for automated workloads
  - [ ] Data lineage enabled and visible in Unity Catalog UI

- [ ] **Network & Security**
  - [ ] VNet injection configured (if required)
  - [ ] Private endpoints enabled for Azure storage
  - [ ] Secrets stored in Azure Key Vault
  - [ ] Databricks secret scopes configured
  - [ ] IP access lists configured (if required)
  - [ ] Audit logs enabled and forwarded to central SIEM

- [ ] **Compute Resources**
  - [ ] Production cluster pools created
  - [ ] Cluster policies enforced (instance types, auto-termination)
  - [ ] Photon engine enabled for performance
  - [ ] Spot instance usage evaluated for cost savings
  - [ ] Auto-scaling configured based on load patterns

### Data Quality & Validation

- [ ] **Synthetic Data Generation**
  - [ ] Data generation scripts tested with production-scale volumes
  - [ ] Data quality validation implemented
  - [ ] Referential integrity checks automated
  - [ ] Data lineage documented
  - [ ] Backup and recovery procedures tested

- [ ] **Feature Store**
  - [ ] Feature tables created with proper primary keys
  - [ ] Feature descriptions and metadata documented
  - [ ] Data types optimized for storage and performance
  - [ ] Partition strategy implemented for large tables
  - [ ] Liquid clustering enabled on feature tables
  - [ ] Delta table properties optimized (optimize write, auto-compact)

- [ ] **Data Validation**
  - [ ] Schema validation automated
  - [ ] Range checks implemented for all numeric features
  - [ ] Null value policies defined and enforced
  - [ ] Outlier detection configured
  - [ ] Data quality metrics tracked over time

### Model Development & Training

- [ ] **MLflow Integration**
  - [ ] Experiment tracking configured
  - [ ] Model registry set up with staging/production stages
  - [ ] Model versioning strategy defined
  - [ ] Artifact storage configured (DBFS or external)
  - [ ] Model signatures included for all models
  - [ ] Custom metrics logged for business KPIs

- [ ] **Feature Store Integration**
  - [ ] Training sets created with FeatureLookup
  - [ ] Feature store metadata packaged with models
  - [ ] Point-in-time correctness verified for time-series features
  - [ ] Feature lineage tracked from raw data to model

- [ ] **Model Validation**
  - [ ] Train/test split strategy documented
  - [ ] Cross-validation implemented for hyperparameter tuning
  - [ ] Holdout test set reserved for final evaluation
  - [ ] Model performance thresholds defined
  - [ ] A/B testing framework ready (if applicable)

### Serving Infrastructure

- [ ] **Batch Inference**
  - [ ] Batch inference notebooks tested at scale
  - [ ] Prediction storage tables created
  - [ ] Scheduling configured via Databricks Jobs
  - [ ] Error handling and retry logic implemented
  - [ ] Performance benchmarks established

- [ ] **Online Feature Store**
  - [ ] Online tables created and syncing
  - [ ] Refresh frequency optimized for use case
  - [ ] Online store performance tested (< 10ms p99)
  - [ ] Fallback logic implemented for sync failures
  - [ ] Cost monitoring enabled

- [ ] **Real-time Serving**
  - [ ] Model serving endpoint created
  - [ ] Endpoint scaling configured (min/max instances)
  - [ ] Health checks configured
  - [ ] Load testing completed
  - [ ] Latency SLAs defined and monitored
  - [ ] Rate limiting configured

### CI/CD Pipeline

- [ ] **GitHub Actions**
  - [ ] Dev, staging, prod workflows configured
  - [ ] Secrets managed securely (GitHub Secrets)
  - [ ] OAuth OIDC configured for production (no long-lived tokens)
  - [ ] Environment protection rules enabled
  - [ ] Manual approvals required for production
  - [ ] Rollback procedures documented

- [ ] **Databricks Asset Bundles**
  - [ ] Bundle configuration validated for all environments
  - [ ] Resource definitions complete (jobs, endpoints, tables)
  - [ ] Variables externalized for environment-specific config
  - [ ] Bundle deployment tested in dev/staging
  - [ ] Deployment logs captured and analyzed

- [ ] **Testing**
  - [ ] Unit tests achieving >80% code coverage
  - [ ] Integration tests running in CI pipeline
  - [ ] End-to-end tests validated in staging
  - [ ] Performance tests baseline established
  - [ ] Chaos engineering tests (optional but recommended)

### Monitoring & Observability

- [ ] **Data Quality Monitoring**
  - [ ] Feature completeness tracked
  - [ ] Feature range validation automated
  - [ ] Feature freshness monitored
  - [ ] Data drift detection configured
  - [ ] Anomaly detection implemented

- [ ] **Model Performance Monitoring**
  - [ ] Prediction logging enabled
  - [ ] Model accuracy tracked over time
  - [ ] Prediction drift detection configured
  - [ ] Business metric tracking (conversion, revenue, etc.)
  - [ ] Model retraining triggers defined

- [ ] **Operational Monitoring**
  - [ ] Job success/failure alerts configured
  - [ ] Compute resource utilization tracked
  - [ ] Cost monitoring dashboards created
  - [ ] Serving endpoint health monitored
  - [ ] Latency p95/p99 tracked

- [ ] **Alerting**
  - [ ] Slack integration configured
  - [ ] Email alerts set up for critical issues
  - [ ] PagerDuty integration (if applicable)
  - [ ] Alert escalation policies defined
  - [ ] On-call rotation documented

### Documentation

- [ ] **Architecture Documentation**
  - [ ] System architecture diagram created
  - [ ] Data flow diagrams documented
  - [ ] Feature engineering logic explained
  - [ ] Model training pipeline documented
  - [ ] Serving architecture described

- [ ] **Operational Runbooks**
  - [ ] Incident response procedures documented
  - [ ] Common troubleshooting scenarios covered
  - [ ] Rollback procedures clearly defined
  - [ ] Disaster recovery plan documented
  - [ ] Data backup/restore procedures tested

- [ ] **API Documentation**
  - [ ] Feature Store APIs documented
  - [ ] Model serving API documented with examples
  - [ ] Authentication and authorization explained
  - [ ] Rate limits and quotas documented
  - [ ] Error codes and handling described

### Compliance & Governance

- [ ] **Data Governance**
  - [ ] Data classification implemented (PII, sensitive, public)
  - [ ] Data retention policies defined and automated
  - [ ] Data access logged and auditable
  - [ ] GDPR/CCPA compliance verified (if applicable)
  - [ ] Data deletion procedures implemented

- [ ] **Model Governance**
  - [ ] Model approval process defined
  - [ ] Model risk assessment completed
  - [ ] Bias and fairness testing performed
  - [ ] Explainability requirements addressed
  - [ ] Model cards created for documentation

- [ ] **Audit & Compliance**
  - [ ] Audit log retention configured (90+ days)
  - [ ] Compliance reports automated
  - [ ] Change management process followed
  - [ ] Penetration testing completed (if required)
  - [ ] SOC 2 / ISO 27001 compliance verified

## Production Best Practices

### Feature Engineering

```python
# DO: Use descriptive feature names
features = {
    'customer_total_spend_30d': 1500.00,
    'customer_transaction_count_30d': 12,
    'customer_days_since_last_purchase': 5
}

# DON'T: Use cryptic abbreviations
features = {
    'cts30': 1500.00,
    'txc30': 12,
    'dslp': 5
}
```

### Error Handling

```python
# DO: Implement comprehensive error handling
try:
    features_df = compute_customer_features(transactions_df, customers_df)
    validate_features(features_df)
    
    fe.write_table(
        name=feature_table_name,
        df=features_df,
        mode="merge"
    )
    
    log_success(run_id, features_df.count())
    
except ValueError as e:
    log_error(run_id, "Validation failed", str(e))
    send_alert(f"Feature validation failed: {str(e)}")
    raise
    
except Exception as e:
    log_error(run_id, "Unexpected error", str(e))
    send_alert(f"Critical error in feature computation: {str(e)}")
    raise
```

### Logging

```python
# DO: Log at appropriate levels with context
import logging

logger = logging.getLogger(__name__)

logger.info(f"Starting feature computation for {num_customers} customers")
logger.debug(f"Using configuration: {config}")
logger.warning(f"Found {null_count} null values in age column")
logger.error(f"Feature computation failed: {error_message}")
logger.critical(f"Service unavailable: {service_name}")

# DON'T: Use print statements
print("Starting computation")
print(f"Error: {error}")
```

### Configuration Management

```python
# DO: Externalize configuration
from dataclasses import dataclass
from typing import Dict
import yaml

@dataclass
class FeatureStoreConfig:
    catalog: str
    schema: str
    feature_table: str
    refresh_interval_hours: int
    quality_thresholds: Dict[str, float]
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

# Usage
config = FeatureStoreConfig.from_yaml('config/prod.yaml')
```

### Testing Patterns

```python
# DO: Write comprehensive tests

def test_feature_computation_with_empty_transactions():
    """Test that feature computation handles empty transaction data."""
    customers_df = create_sample_customers(n=10)
    transactions_df = create_empty_dataframe(transaction_schema)
    
    features_df = compute_customer_features(transactions_df, customers_df)
    
    # All customers should have zero transaction features
    assert features_df.filter(col("total_transactions_30d") != 0).count() == 0
    assert features_df.filter(col("total_spend_30d") != 0).count() == 0


def test_feature_validation_catches_invalid_age():
    """Test that validation detects invalid age values."""
    features_df = create_features_with_invalid_age()
    
    with pytest.raises(ValueError, match="Invalid age"):
        validate_features(features_df)
```

### Performance Optimization

```python
# DO: Optimize Spark operations

# Cache intermediate results used multiple times
customers_df.cache()
customer_count = customers_df.count()  # Materialize cache

# Use broadcast for small tables
from pyspark.sql.functions import broadcast

features = large_table.join(
    broadcast(small_lookup_table),
    on="key"
)

# Repartition before expensive operations
transactions_df = transactions_df.repartition(200, "customer_id")
```

### Security

```python
# DO: Use secret scopes, never hardcode credentials

# ❌ NEVER do this
token = "dapi1234567890abcdef"
database_password = "MyP@ssw0rd!"

# ✅ Always do this
token = dbutils.secrets.get(scope="production", key="databricks-token")
db_password = dbutils.secrets.get(scope="production", key="db-password")
```

## Production Monitoring Dashboard Queries

### Feature Quality Dashboard

```sql
-- Feature Completeness Over Time
SELECT 
    DATE(computation_timestamp) as date,
    COUNT(*) as total_features,
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) / COUNT(*) as null_age_rate,
    SUM(CASE WHEN total_spend_30d IS NULL THEN 1 ELSE 0 END) / COUNT(*) as null_spend_rate,
    AVG(completeness_score) as avg_completeness
FROM ml.monitoring.feature_quality_metrics
WHERE computation_timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
GROUP BY DATE(computation_timestamp)
ORDER BY date DESC;

-- Feature Drift Detection
SELECT 
    feature_name,
    baseline_mean,
    current_mean,
    drift_percentage,
    ks_statistic,
    p_value,
    drift_detected
FROM ml.monitoring.feature_drift_metrics
WHERE check_timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
  AND drift_detected = TRUE
ORDER BY drift_percentage DESC;
```

### Model Performance Dashboard

```sql
-- Model Accuracy Trend
SELECT 
    DATE(prediction_date) as date,
    COUNT(*) as total_predictions,
    AVG(CASE WHEN prediction = actual_label THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(prediction_confidence) as avg_confidence,
    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) / COUNT(*) as positive_rate
FROM ml.monitoring.model_predictions
WHERE prediction_date >= CURRENT_DATE - INTERVAL 30 DAYS
GROUP BY DATE(prediction_date)
ORDER BY date DESC;

-- Prediction Latency
SELECT 
    HOUR(prediction_timestamp) as hour,
    AVG(inference_time_ms) as avg_latency_ms,
    PERCENTILE(inference_time_ms, 0.95) as p95_latency_ms,
    PERCENTILE(inference_time_ms, 0.99) as p99_latency_ms,
    MAX(inference_time_ms) as max_latency_ms
FROM ml.monitoring.inference_log
WHERE prediction_date = CURRENT_DATE
GROUP BY HOUR(prediction_timestamp)
ORDER BY hour;
```

### Operational Dashboard

```sql
-- Job Success Rate
SELECT 
    job_name,
    DATE(run_timestamp) as date,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful_runs,
    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) / COUNT(*) as success_rate,
    AVG(duration_minutes) as avg_duration_minutes
FROM ml.monitoring.job_runs
WHERE run_timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
GROUP BY job_name, DATE(run_timestamp)
ORDER BY date DESC, job_name;

-- Cost Monitoring
SELECT 
    DATE(usage_date) as date,
    cluster_name,
    SUM(dbu_consumed) as total_dbu,
    SUM(dbu_consumed * dbu_cost) as total_cost_usd,
    AVG(cluster_uptime_hours) as avg_uptime_hours
FROM ml.monitoring.compute_usage
WHERE usage_date >= CURRENT_DATE - INTERVAL 30 DAYS
GROUP BY DATE(usage_date), cluster_name
ORDER BY date DESC, total_cost_usd DESC;
```

## Troubleshooting Guide

### Issue: Feature Table Write Fails

**Symptoms:**
```
AnalysisException: Conflicting columns detected
```

**Diagnosis:**
```python
# Check schema compatibility
current_schema = spark.table(feature_table).schema
new_schema = features_df.schema

print("Current schema:", current_schema)
print("New schema:", new_schema)
```

**Resolution:**
```python
# Option 1: Evolve schema
fe.write_table(
    name=feature_table,
    df=features_df,
    mode="merge",
    schema_evolution_mode="overwrite"  # or "addNewColumns"
)

# Option 2: Explicitly match schema
features_df = features_df.select(*current_schema.fieldNames())
```

### Issue: High Serving Latency

**Symptoms:**
- p95 latency > 500ms
- Timeouts on prediction requests

**Diagnosis:**
```python
# Check online store sync status
online_status = spark.sql("""
    SELECT 
        MAX(last_sync_time) as last_sync,
        TIMESTAMPDIFF(MINUTE, MAX(last_sync_time), CURRENT_TIMESTAMP) as lag_minutes
    FROM system.information_schema.online_tables
    WHERE table_name = 'customer_features'
""").collect()[0]

print(f"Last sync: {online_status.last_sync}")
print(f"Lag: {online_status.lag_minutes} minutes")
```

**Resolution:**
```python
# 1. Reduce online store refresh interval
# 2. Increase serving endpoint compute
# 3. Implement feature caching
# 4. Reduce number of features used
```

### Issue: Memory Errors During Training

**Symptoms:**
```
java.lang.OutOfMemoryError: Java heap space
```

**Diagnosis:**
```python
# Check data size
print(f"Training data size: {training_df.count()} rows")
print(f"Memory usage: {training_df.rdd.map(lambda x: len(str(x))).sum()} bytes")
```

**Resolution:**
```python
# 1. Sample data for training
training_sample = training_df.sample(fraction=0.1, seed=42)

# 2. Use distributed algorithms
from pyspark.ml.classification import LogisticRegression as SparkLR

# 3. Increase driver memory
spark.conf.set("spark.driver.memory", "16g")
```

## Post-Deployment Tasks

### Week 1: Validation
- [ ] Monitor all alerts (should be minimal)
- [ ] Verify feature freshness daily
- [ ] Check model prediction quality
- [ ] Review cost consumption
- [ ] Validate backup procedures

### Week 2-4: Optimization
- [ ] Analyze performance bottlenecks
- [ ] Optimize slow queries
- [ ] Right-size compute resources
- [ ] Fine-tune alert thresholds
- [ ] Document learnings

### Monthly: Review & Improve
- [ ] Review incident reports
- [ ] Analyze cost trends
- [ ] Update documentation
- [ ] Conduct team retrospective
- [ ] Plan feature enhancements

---

**Remember:** Production is not a destination, it's a continuous journey of improvement!