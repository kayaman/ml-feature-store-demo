"""
Model training logic with Feature Store integration.
"""
import mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
import pandas as pd
from typing import Tuple, Dict
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup


class ChurnModel:
    """Churn prediction model with feature store integration."""
    
    def __init__(self, catalog: str, schema: str):
        self.catalog = catalog
        self.schema = schema
        self.fe = FeatureEngineeringClient()
        self.model = None
        
    def create_training_set(self, labels_df: pd.DataFrame) -> Tuple:
        """
        Create training set with automatic feature lookup.
        
        Args:
            labels_df: Spark DataFrame with customer_id and churn_label
            
        Returns:
            Tuple of (training_set object, loaded DataFrame)
        """
        # Define features to lookup from feature store
        feature_lookups = [
            FeatureLookup(
                table_name=f"{self.catalog}.{self.schema}.customer_features",
                feature_names=[
                    "age",
                    "customer_tenure_days",
                    "total_transactions_30d",
                    "total_spend_30d",
                    "avg_transaction_value_30d",
                    "days_since_last_transaction",
                    "is_high_value",
                    "is_frequent_buyer"
                ],
                lookup_key="customer_id"
            )
        ]
        
        # Create training set with feature lookups
        training_set = self.fe.create_training_set(
            df=labels_df,
            feature_lookups=feature_lookups,
            label="churn_label",
            exclude_columns=["customer_id"]
        )
        
        # Load as pandas DataFrame for sklearn
        training_df = training_set.load_df().toPandas()
        
        return training_set, training_df
    
    def train(self, training_set, training_df: pd.DataFrame, 
              experiment_name: str) -> Dict[str, float]:
        """
        Train logistic regression model and log to MLflow.
        
        Args:
            training_set: TrainingSet object from Feature Store
            training_df: Loaded pandas DataFrame
            experiment_name: MLflow experiment name
            
        Returns:
            Dictionary of evaluation metrics
        """
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="logistic_regression_churn") as run:
            # Split data
            X = training_df.drop("churn_label", axis=1)
            y = training_df["churn_label"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Log parameters
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("num_features", X_train.shape[1])
            mlflow.log_param("training_samples", len(X_train))
            
            # Train model
            self.model = LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_pred_proba)
            }
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            mlflow.log_dict(
                feature_importance.to_dict(orient='records'),
                "feature_importance.json"
            )
            
            # Log model with feature store metadata
            signature = infer_signature(X_train, y_train)
            
            self.fe.log_model(
                model=self.model,
                artifact_path="model",
                flavor=mlflow.sklearn,
                training_set=training_set,
                registered_model_name=f"{self.catalog}.{self.schema}.churn_model",
                signature=signature
            )
            
            print(f"Model logged to run: {run.info.run_id}")
            print(f"Metrics: {metrics}")
            
            return metrics