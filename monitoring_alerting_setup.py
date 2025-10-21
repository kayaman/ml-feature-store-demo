"""
Production Monitoring and Alerting System
Comprehensive monitoring for Feature Store, MLflow models, and serving endpoints.
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class DataQualityMonitor:
    """Monitor data quality metrics for feature store."""
    
    def __init__(self, catalog: str, schema: str):
        self.catalog = catalog
        self.schema = schema
        self.w = WorkspaceClient()
    
    def check_feature_completeness(self) -> Dict:
        """Check for null values and missing features."""
        query = f"""
        SELECT 
            COUNT(*) as total_rows,
            SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) as null_age,
            SUM(CASE WHEN total_spend_30d IS NULL THEN 1 ELSE 0 END) as null_spend,
            SUM(CASE WHEN days_since_last_transaction IS NULL THEN 1 ELSE 0 END) as null_recency
        FROM {self.catalog}.{self.schema}.customer_features
        """
        
        result = spark.sql(query).collect()[0]
        
        metrics = {
            'total_rows': result.total_rows,
            'null_age': result.null_age,
            'null_spend': result.null_spend,
            'null_recency': result.null_recency,
            'completeness_rate': 1 - (
                (result.null_age + result.null_spend + result.null_recency) 
                / (result.total_rows * 3)
            )
        }
        
        return metrics
    
    def check_feature_ranges(self) -> Dict:
        """Check if features are within expected ranges."""
        query = f"""
        SELECT 
            MIN(age) as min_age,
            MAX(age) as max_age,
            MIN(total_spend_30d) as min_spend,
            MAX(total_spend_30d) as max_spend,
            MIN(days_since_last_transaction) as min_recency,
            MAX(days_since_last_transaction) as max_recency,
            COUNT(CASE WHEN age < 18 OR age > 100 THEN 1 END) as invalid_age_count,
            COUNT(CASE WHEN total_spend_30d < 0 THEN 1 END) as negative_spend_count
        FROM {self.catalog}.{self.schema}.customer_features
        """
        
        result = spark.sql(query).collect()[0]
        
        return {
            'age_range': (result.min_age, result.max_age),
            'spend_range': (result.min_spend, result.max_spend),
            'recency_range': (result.min_recency, result.max_recency),
            'invalid_age_count': result.invalid_age_count,
            'negative_spend_count': result.negative_spend_count
        }
    
    def check_feature_freshness(self) -> Dict:
        """Check how fresh the features are."""
        query = f"""
        SELECT 
            MAX(days_since_last_transaction) as oldest_activity,
            AVG(days_since_last_transaction) as avg_recency,
            PERCENTILE(days_since_last_transaction, 0.95) as p95_recency,
            COUNT(CASE WHEN days_since_last_transaction > 60 THEN 1 END) as stale_count,
            COUNT(*) as total_count
        FROM {self.catalog}.{self.schema}.customer_features
        """
        
        result = spark.sql(query).collect()[0]
        
        return {
            'oldest_activity_days': result.oldest_activity,
            'avg_recency_days': result.avg_recency,
            'p95_recency_days': result.p95_recency,
            'stale_count': result.stale_count,
            'stale_percentage': result.stale_count / result.total_count
        }
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive data quality report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'completeness': self.check_feature_completeness(),
            'ranges': self.check_feature_ranges(),
            'freshness': self.check_feature_freshness()
        }


class ModelPerformanceMonitor:
    """Monitor ML model performance metrics."""
    
    def __init__(self, model_name: str, version: int):
        self.model_name = model_name
        self.version = version
    
    def get_prediction_metrics(self, start_date: str, end_date: str) -> Dict:
        """Get prediction performance metrics from inference logs."""
        query = f"""
        SELECT 
            COUNT(*) as total_predictions,
            AVG(prediction_confidence) as avg_confidence,
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as positive_predictions,
            AVG(inference_time_ms) as avg_latency_ms,
            PERCENTILE(inference_time_ms, 0.95) as p95_latency_ms,
            PERCENTILE(inference_time_ms, 0.99) as p99_latency_ms
        FROM ml.monitoring.inference_log
        WHERE model_name = '{self.model_name}'
          AND model_version = {self.version}
          AND prediction_date BETWEEN '{start_date}' AND '{end_date}'
        """
        
        result = spark.sql(query).collect()[0]
        
        return {
            'total_predictions': result.total_predictions,
            'avg_confidence': result.avg_confidence,
            'positive_rate': result.positive_predictions / result.total_predictions,
            'avg_latency_ms': result.avg_latency_ms,
            'p95_latency_ms': result.p95_latency_ms,
            'p99_latency_ms': result.p99_latency_ms
        }
    
    def detect_prediction_drift(self, baseline_period: int = 7, current_period: int = 1) -> Dict:
        """Detect drift in prediction distribution."""
        today = datetime.now().date()
        
        # Baseline period
        baseline_start = today - timedelta(days=baseline_period + current_period)
        baseline_end = today - timedelta(days=current_period)
        
        # Current period
        current_start = today - timedelta(days=current_period)
        current_end = today
        
        baseline_query = f"""
        SELECT AVG(CASE WHEN prediction = 1 THEN 1.0 ELSE 0.0 END) as positive_rate
        FROM ml.monitoring.inference_log
        WHERE prediction_date BETWEEN '{baseline_start}' AND '{baseline_end}'
        """
        
        current_query = f"""
        SELECT AVG(CASE WHEN prediction = 1 THEN 1.0 ELSE 0.0 END) as positive_rate
        FROM ml.monitoring.inference_log
        WHERE prediction_date BETWEEN '{current_start}' AND '{current_end}'
        """
        
        baseline_rate = spark.sql(baseline_query).collect()[0].positive_rate
        current_rate = spark.sql(current_query).collect()[0].positive_rate
        
        drift_pct = abs(current_rate - baseline_rate) / baseline_rate if baseline_rate > 0 else 0
        
        return {
            'baseline_positive_rate': baseline_rate,
            'current_positive_rate': current_rate,
            'drift_percentage': drift_pct,
            'drift_detected': drift_pct > 0.15  # 15% threshold
        }


class ServingEndpointMonitor:
    """Monitor model serving endpoint health and performance."""
    
    def __init__(self, endpoint_name: str, workspace_url: str, token: str):
        self.endpoint_name = endpoint_name
        self.workspace_url = workspace_url
        self.token = token
        self.api_url = f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}"
    
    def get_endpoint_status(self) -> Dict:
        """Get current endpoint status."""
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(self.api_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'state': data['state']['ready'],
                'config_update': data['state'].get('config_update'),
                'pending_config': data.get('pending_config')
            }
        else:
            return {'error': f"Status code: {response.status_code}"}
    
    def get_endpoint_metrics(self) -> Dict:
        """Get endpoint performance metrics."""
        headers = {"Authorization": f"Bearer {self.token}"}
        metrics_url = f"{self.api_url}/metrics"
        
        response = requests.get(metrics_url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'requests_per_minute': data.get('requests_per_minute', 0),
                'avg_latency_ms': data.get('avg_latency_ms', 0),
                'error_rate': data.get('error_rate', 0),
                'p95_latency_ms': data.get('p95_latency_ms', 0)
            }
        else:
            return {'error': f"Status code: {response.status_code}"}
    
    def health_check(self) -> bool:
        """Perform health check on endpoint."""
        status = self.get_endpoint_status()
        metrics = self.get_endpoint_metrics()
        
        # Check criteria
        is_ready = status.get('state') == 'READY'
        error_rate_ok = metrics.get('error_rate', 1.0) < 0.05  # < 5% errors
        latency_ok = metrics.get('p95_latency_ms', 1000) < 500  # < 500ms p95
        
        return is_ready and error_rate_ok and latency_ok


class AlertManager:
    """Manage alerts for various monitoring conditions."""
    
    def __init__(self, slack_webhook: Optional[str] = None, email_config: Optional[Dict] = None):
        self.slack_webhook = slack_webhook
        self.email_config = email_config
        self.alerts_sent = []
    
    def send_slack_alert(self, message: str, severity: str = "warning"):
        """Send alert to Slack."""
        if not self.slack_webhook:
            print(f"Slack webhook not configured. Alert: {message}")
            return
        
        color_map = {
            'critical': '#FF0000',
            'warning': '#FFA500',
            'info': '#00FF00'
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(severity, '#FFA500'),
                "title": f"🚨 Feature Store Alert - {severity.upper()}",
                "text": message,
                "footer": "Databricks Feature Store Monitor",
                "ts": int(datetime.now().timestamp())
            }]
        }
        
        try:
            response = requests.post(self.slack_webhook, json=payload)
            if response.status_code == 200:
                self.alerts_sent.append({
                    'timestamp': datetime.now(),
                    'channel': 'slack',
                    'message': message,
                    'severity': severity
                })
                print(f"✅ Slack alert sent: {message}")
            else:
                print(f"❌ Failed to send Slack alert: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending Slack alert: {str(e)}")
    
    def send_email_alert(self, subject: str, message: str, recipients: List[str]):
        """Send alert via email."""
        if not self.email_config:
            print(f"Email not configured. Alert: {message}")
            return
        
        msg = MIMEMultipart()
        msg['From'] = self.email_config['from_email']
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject
        
        body = f"""
        <html>
        <body>
        <h2>Feature Store Alert</h2>
        <p>{message}</p>
        <hr>
        <p><small>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        try:
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            self.alerts_sent.append({
                'timestamp': datetime.now(),
                'channel': 'email',
                'message': message,
                'recipients': recipients
            })
            print(f"✅ Email alert sent to {recipients}")
        except Exception as e:
            print(f"❌ Error sending email alert: {str(e)}")
    
    def get_alert_history(self) -> List[Dict]:
        """Get history of sent alerts."""
        return self.alerts_sent


class MonitoringOrchestrator:
    """Orchestrate all monitoring checks and alerting."""
    
    def __init__(
        self,
        catalog: str,
        schema: str,
        model_name: str,
        model_version: int,
        endpoint_name: str,
        workspace_url: str,
        token: str,
        alert_manager: AlertManager
    ):
        self.data_quality_monitor = DataQualityMonitor(catalog, schema)
        self.model_monitor = ModelPerformanceMonitor(model_name, model_version)
        self.endpoint_monitor = ServingEndpointMonitor(endpoint_name, workspace_url, token)
        self.alert_manager = alert_manager
    
    def run_all_checks(self) -> Dict:
        """Run all monitoring checks and send alerts if needed."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'alerts_triggered': []
        }
        
        # 1. Data Quality Checks
        print("Running data quality checks...")
        quality_report = self.data_quality_monitor.generate_quality_report()
        results['checks']['data_quality'] = quality_report
        
        # Alert on data quality issues
        completeness = quality_report['completeness']['completeness_rate']
        if completeness < 0.95:
            alert_msg = f"Data completeness dropped to {completeness:.1%}. Expected > 95%"
            self.alert_manager.send_slack_alert(alert_msg, severity='warning')
            results['alerts_triggered'].append(alert_msg)
        
        freshness = quality_report['freshness']
        if freshness['stale_percentage'] > 0.20:
            alert_msg = f"Feature staleness at {freshness['stale_percentage']:.1%}. Expected < 20%"
            self.alert_manager.send_slack_alert(alert_msg, severity='warning')
            results['alerts_triggered'].append(alert_msg)
        
        # 2. Model Performance Checks
        print("Checking model performance...")
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        
        try:
            prediction_metrics = self.model_monitor.get_prediction_metrics(
                start_date=str(week_ago),
                end_date=str(today)
            )
            results['checks']['model_performance'] = prediction_metrics
            
            # Alert on high latency
            if prediction_metrics['p95_latency_ms'] > 500:
                alert_msg = f"Model p95 latency at {prediction_metrics['p95_latency_ms']:.0f}ms. Expected < 500ms"
                self.alert_manager.send_slack_alert(alert_msg, severity='warning')
                results['alerts_triggered'].append(alert_msg)
        except Exception as e:
            print(f"Error checking model performance: {str(e)}")
            results['checks']['model_performance'] = {'error': str(e)}
        
        # 3. Prediction Drift Check
        print("Detecting prediction drift...")
        try:
            drift_metrics = self.model_monitor.detect_prediction_drift()
            results['checks']['prediction_drift'] = drift_metrics
            
            if drift_metrics['drift_detected']:
                alert_msg = (
                    f"Prediction drift detected: {drift_metrics['drift_percentage']:.1%} change. "
                    f"Baseline: {drift_metrics['baseline_positive_rate']:.2%}, "
                    f"Current: {drift_metrics['current_positive_rate']:.2%}"
                )
                self.alert_manager.send_slack_alert(alert_msg, severity='critical')
                results['alerts_triggered'].append(alert_msg)
        except Exception as e:
            print(f"Error detecting drift: {str(e)}")
            results['checks']['prediction_drift'] = {'error': str(e)}
        
        # 4. Endpoint Health Check
        print("Checking endpoint health...")
        try:
            endpoint_healthy = self.endpoint_monitor.health_check()
            endpoint_status = self.endpoint_monitor.get_endpoint_status()
            endpoint_metrics = self.endpoint_monitor.get_endpoint_metrics()
            
            results['checks']['endpoint'] = {
                'healthy': endpoint_healthy,
                'status': endpoint_status,
                'metrics': endpoint_metrics
            }
            
            if not endpoint_healthy:
                alert_msg = f"Serving endpoint unhealthy. Status: {endpoint_status}"
                self.alert_manager.send_slack_alert(alert_msg, severity='critical')
                results['alerts_triggered'].append(alert_msg)
        except Exception as e:
            print(f"Error checking endpoint: {str(e)}")
            results['checks']['endpoint'] = {'error': str(e)}
        
        # Summary
        print(f"\n{'='*60}")
        print("MONITORING SUMMARY")
        print(f"{'='*60}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Alerts triggered: {len(results['alerts_triggered'])}")
        if results['alerts_triggered']:
            for alert in results['alerts_triggered']:
                print(f"  ⚠️ {alert}")
        else:
            print("  ✅ All checks passed")
        print(f"{'='*60}\n")
        
        return results
    
    def continuous_monitoring(self, interval_minutes: int = 15):
        """Run monitoring checks continuously."""
        import time
        
        print(f"Starting continuous monitoring (every {interval_minutes} minutes)...")
        
        while True:
            try:
                self.run_all_checks()
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying


def create_monitoring_dashboard(catalog: str, schema: str):
    """Create SQL dashboards for monitoring."""
    
    dashboard_queries = {
        'feature_quality': f"""
            SELECT 
                CURRENT_DATE as check_date,
                COUNT(*) as total_features,
                SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) as null_age_count,
                SUM(CASE WHEN total_spend_30d IS NULL THEN 1 ELSE 0 END) as null_spend_count,
                AVG(total_spend_30d) as avg_spend,
                PERCENTILE(days_since_last_transaction, 0.95) as p95_recency
            FROM {catalog}.{schema}.customer_features
        """,
        
        'model_performance_7d': f"""
            SELECT 
                DATE(prediction_timestamp) as prediction_date,
                COUNT(*) as num_predictions,
                AVG(prediction_confidence) as avg_confidence,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) / COUNT(*) as positive_rate,
                AVG(inference_time_ms) as avg_latency_ms
            FROM ml.monitoring.inference_log
            WHERE prediction_timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
            GROUP BY DATE(prediction_timestamp)
            ORDER BY prediction_date DESC
        """,
        
        'feature_computation_health': f"""
            SELECT 
                DATE(run_timestamp) as run_date,
                COUNT(*) as num_runs,
                AVG(duration_seconds) as avg_duration_sec,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                MAX(rows_per_second) as max_throughput
            FROM {catalog}.{schema}.feature_computation_metrics
            WHERE run_timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
            GROUP BY DATE(run_timestamp)
            ORDER BY run_date DESC
        """
    }
    
    return dashboard_queries


# Example usage
if __name__ == "__main__":
    # Configuration
    CATALOG = "ml"
    SCHEMA = "churn_features"
    MODEL_NAME = "ml.churn_features.churn_model"
    MODEL_VERSION = 1
    ENDPOINT_NAME = "churn-prediction-realtime"
    WORKSPACE_URL = "https://adb-1234567.7.azuredatabricks.net"
    TOKEN = dbutils.secrets.get(scope="feature-store", key="databricks-token")
    SLACK_WEBHOOK = dbutils.secrets.get(scope="feature-store", key="slack-webhook")
    
    # Initialize alert manager
    alert_manager = AlertManager(slack_webhook=SLACK_WEBHOOK)
    
    # Initialize orchestrator
    orchestrator = MonitoringOrchestrator(
        catalog=CATALOG,
        schema=SCHEMA,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        endpoint_name=ENDPOINT_NAME,
        workspace_url=WORKSPACE_URL,
        token=TOKEN,
        alert_manager=alert_manager
    )
    
    # Run checks
    results = orchestrator.run_all_checks()
    
    # Print summary
    print(json.dumps(results, indent=2, default=str))