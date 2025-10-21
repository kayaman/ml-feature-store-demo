# Azure Databricks Feature Store Production Template

Production-ready ML system with Feature Store, MLflow, and GitHub Actions CI/CD.

## Features

- ✅ Unity Catalog Feature Engineering
- ✅ MLflow 2.x integration
- ✅ Python 3.12 on DBR 16.4 LTS ML
- ✅ GitHub Actions CI/CD with OAuth OIDC
- ✅ Offline (batch) and online (real-time) serving
- ✅ Complete testing suite
- ✅ Multi-environment deployment (dev/staging/prod)

## Quick Start

### Prerequisites

1. Azure Databricks workspace with Unity Catalog enabled
2. DBR 16.4 LTS ML cluster
3. GitHub repository with Actions enabled
4. Databricks CLI installed locally

### Local Development
```bash
# Clone repository
git clone https://github.com/your-org/databricks-ml-feature-store.git
cd databricks-ml-feature-store

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/unit -v
```

### First-Time Setup

1. **Configure GitHub Secrets:**
   - `DATABRICKS_TOKEN` (dev/staging/prod)
   - `DATABRICKS_CLIENT_ID` (for production OAuth)
   - `SLACK_WEBHOOK_URL` (optional)

2. **Configure GitHub Variables:**
   - `DATABRICKS_HOST` per environment

3. **Deploy Infrastructure:**
```bash
# Authenticate
databricks auth login --host https://adb-dev-1234567.7.azuredatabricks.net

# Validate bundle
databricks bundle validate --target dev

# Deploy
databricks bundle deploy --target dev
```

4. **Run Setup Notebook:**
```bash
# Creates catalog, schema, and feature tables
databricks bundle run setup_job --target dev
```

### Development Workflow

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes to code
3. Run tests locally: `pytest tests/ -v`
4. Push and create PR: automatic deployment to dev
5. Merge to main: automatic deployment to staging
6. Manual approval: production deployment via GitHub Actions

## Project Structure

- `src/features/` - Feature engineering logic
- `src/models/` - Model training logic
- `notebooks/` - Databricks notebooks
- `tests/` - Unit and integration tests
- `.github/workflows/` - CI/CD pipelines
- `databricks.yml` - Asset Bundle configuration

## Feature Store Operations

### Compute and Write Features
```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
fe.write_table(
    name="ml.churn_features.customer_features",
    df=features_df,
    mode="merge"
)
```

### Train Model with Features
```python
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churn_label"
)

fe.log_model(
    model=model,
    artifact_path="model",
    flavor=mlflow.sklearn,
    training_set=training_set
)
```

### Batch Inference
```python
predictions = fe.score_batch(
    model_uri="models:/ml.churn_features.churn_model/1",
    df=batch_df  # Only needs primary keys
)
```

## Online Serving

Features are automatically published to Databricks Online Feature Store every hour. For real-time serving:

1. Deploy model to Model Serving endpoint
2. Send requests with only primary keys
3. Features retrieved automatically from online store

## Monitoring

- MLflow experiments: Track all training runs
- Feature lineage: View in Unity Catalog
- Job monitoring: Databricks Workflows UI
- Deployment status: GitHub Actions

## Troubleshooting

**Issue**: Bundle validation fails  
**Solution**: Check `databricks.yml` syntax and catalog/schema exist

**Issue**: Feature table not found  
**Solution**: Run setup notebook to create infrastructure

**Issue**: Authentication fails in CI/CD  
**Solution**: Verify GitHub secrets and OAuth configuration

## Resources

- [Databricks Feature Engineering Docs](https://docs.databricks.com/machine-learning/feature-store/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/)