"""
Synthetic Data Generator for Feature Store Application
Generates realistic customer, transaction, and churn data with proper correlations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    num_customers: int = 10000
    avg_transactions_per_customer: int = 15
    date_range_days: int = 365
    churn_rate: float = 0.15
    seed: int = 42


class SyntheticDataGenerator:
    """Generate synthetic data for churn prediction model."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        np.random.seed(config.seed)
        self.today = datetime.now().date()
        
    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete synthetic dataset.
        
        Returns:
            Tuple of (customers_df, transactions_df, labels_df)
        """
        print(f"Generating synthetic data for {self.config.num_customers} customers...")
        
        customers_df = self._generate_customers()
        transactions_df = self._generate_transactions(customers_df)
        labels_df = self._generate_churn_labels(customers_df, transactions_df)
        
        print(f"✅ Generated {len(customers_df)} customers")
        print(f"✅ Generated {len(transactions_df)} transactions")
        print(f"✅ Generated {len(labels_df)} labels (churn rate: {labels_df['churn_label'].mean():.2%})")
        
        return customers_df, transactions_df, labels_df
    
    def _generate_customers(self) -> pd.DataFrame:
        """Generate customer demographic data with realistic distributions."""
        num_customers = self.config.num_customers
        
        # Customer segments with different characteristics
        segments = np.random.choice(
            ['young_active', 'mid_stable', 'senior_declining', 'new_customer'],
            size=num_customers,
            p=[0.25, 0.40, 0.20, 0.15]
        )
        
        customers = []
        for customer_id in range(1, num_customers + 1):
            segment = segments[customer_id - 1]
            
            # Age distribution by segment
            if segment == 'young_active':
                age = np.random.normal(28, 5)
            elif segment == 'mid_stable':
                age = np.random.normal(42, 8)
            elif segment == 'senior_declining':
                age = np.random.normal(58, 7)
            else:  # new_customer
                age = np.random.normal(35, 10)
            
            age = int(np.clip(age, 18, 80))
            
            # Customer since date (tenure) by segment
            if segment == 'new_customer':
                days_back = np.random.randint(1, 90)  # 0-3 months
            elif segment == 'young_active':
                days_back = np.random.randint(180, 730)  # 6 months - 2 years
            elif segment == 'mid_stable':
                days_back = np.random.randint(730, 1825)  # 2-5 years
            else:  # senior_declining
                days_back = np.random.randint(1095, 2555)  # 3-7 years
            
            customer_since_date = self.today - timedelta(days=days_back)
            
            customers.append({
                'customer_id': customer_id,
                'age': age,
                'customer_since_date': customer_since_date,
                'segment': segment
            })
        
        return pd.DataFrame(customers)
    
    def _generate_transactions(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate transaction data with realistic patterns and correlations."""
        transactions = []
        transaction_id = 1
        
        for _, customer in customers_df.iterrows():
            segment = customer['segment']
            customer_id = customer['customer_id']
            customer_since = customer['customer_since_date']
            
            # Number of transactions varies by segment
            if segment == 'young_active':
                num_txns = np.random.poisson(self.config.avg_transactions_per_customer * 1.5)
            elif segment == 'mid_stable':
                num_txns = np.random.poisson(self.config.avg_transactions_per_customer)
            elif segment == 'senior_declining':
                num_txns = np.random.poisson(self.config.avg_transactions_per_customer * 0.5)
            else:  # new_customer
                num_txns = np.random.poisson(self.config.avg_transactions_per_customer * 0.7)
            
            # Some customers have no recent transactions (churned)
            is_churned = np.random.random() < 0.15
            
            for _ in range(num_txns):
                # Transaction date distribution
                if is_churned:
                    # Churned customers: transactions only in the past (>60 days ago)
                    days_ago = np.random.randint(60, self.config.date_range_days)
                else:
                    # Active customers: more recent transactions
                    days_ago = int(np.random.exponential(30))
                    days_ago = min(days_ago, self.config.date_range_days)
                
                transaction_date = self.today - timedelta(days=days_ago)
                
                # Don't create transactions before customer joined
                if transaction_date < customer_since:
                    continue
                
                # Transaction amount varies by segment
                if segment == 'young_active':
                    base_amount = 150
                    std = 80
                elif segment == 'mid_stable':
                    base_amount = 250
                    std = 120
                elif segment == 'senior_declining':
                    base_amount = 180
                    std = 90
                else:  # new_customer
                    base_amount = 120
                    std = 60
                
                amount = np.random.gamma(2, base_amount / 2)
                amount = max(10, min(amount, 5000))  # Clip to reasonable range
                
                transactions.append({
                    'transaction_id': transaction_id,
                    'customer_id': customer_id,
                    'amount': round(amount, 2),
                    'transaction_date': transaction_date
                })
                
                transaction_id += 1
        
        return pd.DataFrame(transactions)
    
    def _generate_churn_labels(self, customers_df: pd.DataFrame, 
                                transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate churn labels based on realistic business logic.
        
        Churn indicators:
        - No transactions in last 60 days
        - Low total spend
        - Declining transaction frequency
        - Short tenure
        """
        labels = []
        
        # Calculate recent activity for each customer
        recent_cutoff = self.today - timedelta(days=60)
        recent_txns = transactions_df[
            transactions_df['transaction_date'] >= recent_cutoff
        ]
        recent_activity = recent_txns.groupby('customer_id').agg({
            'transaction_id': 'count',
            'amount': 'sum'
        }).rename(columns={'transaction_id': 'recent_txn_count', 'amount': 'recent_spend'})
        
        for _, customer in customers_df.iterrows():
            customer_id = customer['customer_id']
            segment = customer['segment']
            
            # Get customer's recent activity
            if customer_id in recent_activity.index:
                recent_txn_count = recent_activity.loc[customer_id, 'recent_txn_count']
                recent_spend = recent_activity.loc[customer_id, 'recent_spend']
            else:
                recent_txn_count = 0
                recent_spend = 0
            
            # Calculate churn probability based on multiple factors
            churn_prob = 0.1  # Base rate
            
            # Factor 1: No recent activity (strongest indicator)
            if recent_txn_count == 0:
                churn_prob += 0.6
            elif recent_txn_count < 2:
                churn_prob += 0.3
            
            # Factor 2: Low spending
            if recent_spend < 100:
                churn_prob += 0.2
            
            # Factor 3: Segment-specific patterns
            if segment == 'senior_declining':
                churn_prob += 0.15
            elif segment == 'new_customer':
                churn_prob += 0.10
            elif segment == 'young_active':
                churn_prob -= 0.15
            
            # Factor 4: Very short tenure (< 30 days)
            tenure_days = (self.today - customer['customer_since_date']).days
            if tenure_days < 30:
                churn_prob += 0.20
            
            # Clip probability
            churn_prob = np.clip(churn_prob, 0, 0.95)
            
            # Generate label with some randomness
            churn_label = 1 if np.random.random() < churn_prob else 0
            
            labels.append({
                'customer_id': customer_id,
                'churn_label': churn_label,
                'churn_probability': round(churn_prob, 3)  # True probability (not for training)
            })
        
        return pd.DataFrame(labels)
    
    def add_data_quality_issues(self, customers_df: pd.DataFrame, 
                                transactions_df: pd.DataFrame,
                                corruption_rate: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Add realistic data quality issues for testing validation.
        
        Args:
            corruption_rate: Fraction of data to corrupt (default 2%)
        """
        customers_corrupted = customers_df.copy()
        transactions_corrupted = transactions_df.copy()
        
        n_customers = len(customers_corrupted)
        n_transactions = len(transactions_corrupted)
        
        # Introduce nulls in age
        null_idx = np.random.choice(n_customers, size=int(n_customers * corruption_rate * 0.5), replace=False)
        customers_corrupted.loc[null_idx, 'age'] = np.nan
        
        # Introduce invalid ages
        invalid_idx = np.random.choice(n_customers, size=int(n_customers * corruption_rate * 0.5), replace=False)
        customers_corrupted.loc[invalid_idx, 'age'] = np.random.choice([150, -5, 999])
        
        # Introduce negative transaction amounts
        neg_idx = np.random.choice(n_transactions, size=int(n_transactions * corruption_rate), replace=False)
        transactions_corrupted.loc[neg_idx, 'amount'] = -abs(transactions_corrupted.loc[neg_idx, 'amount'])
        
        print(f"Added data quality issues: {corruption_rate:.1%} corruption rate")
        
        return customers_corrupted, transactions_corrupted
    
    def generate_time_series(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-series transaction data for trend analysis.
        Creates daily transaction summaries.
        """
        date_range = pd.date_range(
            end=self.today,
            periods=self.config.date_range_days,
            freq='D'
        )
        
        time_series = []
        for date in date_range:
            # Overall trend: slightly increasing over time
            days_from_start = (date.date() - date_range[0].date()).days
            trend_factor = 1 + (days_from_start / self.config.date_range_days) * 0.2
            
            # Weekly seasonality
            day_of_week = date.dayofweek
            if day_of_week in [5, 6]:  # Weekend
                seasonal_factor = 1.3
            else:
                seasonal_factor = 1.0
            
            # Base daily transactions
            base_txns = 100 * trend_factor * seasonal_factor
            daily_txns = int(np.random.poisson(base_txns))
            
            # Daily revenue
            avg_txn_value = 200
            daily_revenue = daily_txns * avg_txn_value * np.random.uniform(0.8, 1.2)
            
            time_series.append({
                'date': date.date(),
                'transaction_count': daily_txns,
                'total_revenue': round(daily_revenue, 2),
                'avg_transaction_value': round(daily_revenue / max(daily_txns, 1), 2)
            })
        
        return pd.DataFrame(time_series)


def generate_dataset(size: str = 'small') -> Dict[str, pd.DataFrame]:
    """
    Convenience function to generate datasets of different sizes.
    
    Args:
        size: 'small' (1k customers), 'medium' (10k), or 'large' (100k)
    
    Returns:
        Dictionary with all generated dataframes
    """
    size_configs = {
        'small': DataConfig(num_customers=1_000, avg_transactions_per_customer=10),
        'medium': DataConfig(num_customers=10_000, avg_transactions_per_customer=15),
        'large': DataConfig(num_customers=100_000, avg_transactions_per_customer=20)
    }
    
    config = size_configs.get(size, size_configs['small'])
    generator = SyntheticDataGenerator(config)
    
    customers_df, transactions_df, labels_df = generator.generate_all()
    time_series_df = generator.generate_time_series(customers_df)
    
    return {
        'customers': customers_df,
        'transactions': transactions_df,
        'labels': labels_df,
        'time_series': time_series_df
    }


def main():
    """Example usage of synthetic data generator."""
    # Generate medium-sized dataset
    generator = SyntheticDataGenerator(DataConfig(num_customers=5000))
    customers_df, transactions_df, labels_df = generator.generate_all()
    
    # Display summary statistics
    print("\n" + "="*50)
    print("CUSTOMER SUMMARY")
    print("="*50)
    print(customers_df.describe())
    print(f"\nSegment distribution:\n{customers_df['segment'].value_counts()}")
    
    print("\n" + "="*50)
    print("TRANSACTION SUMMARY")
    print("="*50)
    print(transactions_df.describe())
    print(f"\nTotal revenue: ${transactions_df['amount'].sum():,.2f}")
    
    print("\n" + "="*50)
    print("CHURN LABEL SUMMARY")
    print("="*50)
    print(f"Churn rate: {labels_df['churn_label'].mean():.2%}")
    print(f"Total churned: {labels_df['churn_label'].sum()}")
    
    # Generate corrupted data
    print("\n" + "="*50)
    print("GENERATING CORRUPTED DATA")
    print("="*50)
    customers_bad, transactions_bad = generator.add_data_quality_issues(
        customers_df, transactions_df, corruption_rate=0.05
    )
    
    # Show data quality issues
    print(f"Null ages: {customers_bad['age'].isna().sum()}")
    print(f"Invalid ages: {((customers_bad['age'] < 18) | (customers_bad['age'] > 100)).sum()}")
    print(f"Negative amounts: {(transactions_bad['amount'] < 0).sum()}")


if __name__ == "__main__":
    main()