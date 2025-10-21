"""
Advanced Synthetic Data Generation Patterns
Includes streaming data, multi-product catalogs, seasonal patterns, and A/B testing datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class AdvancedDataConfig:
    """Advanced configuration with multi-product support."""
    num_customers: int = 10000
    num_products: int = 50
    num_categories: int = 10
    date_range_days: int = 365
    seasonality_factor: float = 0.3
    promotion_probability: float = 0.15
    seed: int = 42


class AdvancedSyntheticDataGenerator:
    """Generate realistic multi-product, multi-channel synthetic data."""
    
    def __init__(self, config: AdvancedDataConfig):
        self.config = config
        np.random.seed(config.seed)
        self.today = datetime.now().date()
        
    def generate_product_catalog(self) -> pd.DataFrame:
        """Generate product catalog with categories and pricing."""
        products = []
        
        categories = [
            'Electronics', 'Clothing', 'Home & Garden', 'Sports',
            'Books', 'Toys', 'Beauty', 'Food', 'Automotive', 'Health'
        ][:self.config.num_categories]
        
        for product_id in range(1, self.config.num_products + 1):
            category = np.random.choice(categories)
            
            # Price varies by category
            category_price_ranges = {
                'Electronics': (50, 1000),
                'Clothing': (20, 200),
                'Home & Garden': (30, 500),
                'Sports': (25, 300),
                'Books': (10, 50),
                'Toys': (15, 100),
                'Beauty': (10, 150),
                'Food': (5, 50),
                'Automotive': (20, 500),
                'Health': (15, 200)
            }
            
            min_price, max_price = category_price_ranges.get(category, (10, 100))
            base_price = np.random.uniform(min_price, max_price)
            
            products.append({
                'product_id': product_id,
                'product_name': f'{category}_Product_{product_id}',
                'category': category,
                'base_price': round(base_price, 2),
                'cost': round(base_price * 0.6, 2),  # 40% margin
                'stock_quantity': np.random.randint(10, 1000)
            })
        
        return pd.DataFrame(products)
    
    def generate_customer_segments_advanced(self) -> pd.DataFrame:
        """Generate customers with detailed behavioral segments."""
        customers = []
        
        segments = {
            'premium': {'ratio': 0.10, 'age_mean': 45, 'income_mean': 100000},
            'frequent': {'ratio': 0.20, 'age_mean': 35, 'income_mean': 70000},
            'occasional': {'ratio': 0.40, 'age_mean': 40, 'income_mean': 60000},
            'bargain': {'ratio': 0.20, 'age_mean': 50, 'income_mean': 50000},
            'new': {'ratio': 0.10, 'age_mean': 30, 'income_mean': 55000}
        }
        
        # Calculate customers per segment
        segment_counts = {
            seg: int(self.config.num_customers * props['ratio'])
            for seg, props in segments.items()
        }
        
        customer_id = 1
        for segment, count in segment_counts.items():
            props = segments[segment]
            
            for _ in range(count):
                # Age with segment-specific distribution
                age = int(np.clip(
                    np.random.normal(props['age_mean'], 8), 
                    18, 75
                ))
                
                # Income
                income = int(np.clip(
                    np.random.normal(props['income_mean'], 15000),
                    30000, 200000
                ))
                
                # Tenure
                if segment == 'new':
                    days_back = np.random.randint(1, 90)
                elif segment == 'premium':
                    days_back = np.random.randint(730, 2555)
                else:
                    days_back = np.random.randint(180, 1095)
                
                customer_since = self.today - timedelta(days=days_back)
                
                # Location (simplified)
                cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
                city = np.random.choice(cities)
                
                # Channel preference
                channels = ['web', 'mobile', 'store']
                if segment == 'new' or segment == 'frequent':
                    channel_prefs = [0.5, 0.4, 0.1]
                elif segment == 'premium':
                    channel_prefs = [0.4, 0.3, 0.3]
                else:
                    channel_prefs = [0.3, 0.3, 0.4]
                
                preferred_channel = np.random.choice(channels, p=channel_prefs)
                
                customers.append({
                    'customer_id': customer_id,
                    'age': age,
                    'income': income,
                    'city': city,
                    'customer_since_date': customer_since,
                    'segment': segment,
                    'preferred_channel': preferred_channel,
                    'email_subscriber': np.random.random() < 0.6
                })
                
                customer_id += 1
        
        return pd.DataFrame(customers)
    
    def generate_transactions_with_seasonality(
        self, 
        customers_df: pd.DataFrame, 
        products_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate transactions with seasonal patterns and promotions."""
        transactions = []
        transaction_id = 1
        
        for _, customer in customers_df.iterrows():
            segment = customer['segment']
            customer_since = customer['customer_since_date']
            preferred_channel = customer['preferred_channel']
            
            # Transaction frequency by segment
            frequency_map = {
                'premium': 25,
                'frequent': 18,
                'occasional': 8,
                'bargain': 12,
                'new': 5
            }
            base_txn_count = frequency_map[segment]
            num_txns = max(0, int(np.random.poisson(base_txn_count)))
            
            # Determine if customer is churning
            is_churning = np.random.random() < 0.15
            
            for txn_num in range(num_txns):
                # Transaction date with seasonality
                if is_churning and txn_num > num_txns * 0.6:
                    # Churning customers: older transactions
                    days_ago = np.random.randint(60, self.config.date_range_days)
                else:
                    # Active customers: recent transactions
                    days_ago = int(np.random.exponential(30))
                    days_ago = min(days_ago, self.config.date_range_days)
                
                txn_date = self.today - timedelta(days=days_ago)
                
                # Don't create transactions before customer joined
                if txn_date < customer_since:
                    continue
                
                # Seasonal adjustment
                month = txn_date.month
                seasonal_multiplier = self._get_seasonal_multiplier(month)
                
                # Promotion (Black Friday, holidays)
                is_promotion = self._is_promotion_day(txn_date)
                promotion_discount = 0.2 if is_promotion else 0.0
                
                # Select products for this transaction
                num_items = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
                
                # Product selection varies by segment
                if segment == 'premium':
                    # Premium customers buy expensive items
                    product_probs = products_df['base_price'] / products_df['base_price'].sum()
                else:
                    # Other segments more random
                    product_probs = None
                
                selected_products = np.random.choice(
                    products_df['product_id'].values,
                    size=min(num_items, len(products_df)),
                    replace=False,
                    p=product_probs
                )
                
                # Calculate transaction total
                total_amount = 0
                for product_id in selected_products:
                    product = products_df[products_df['product_id'] == product_id].iloc[0]
                    base_price = product['base_price']
                    
                    # Apply seasonal and promotion adjustments
                    price = base_price * seasonal_multiplier * (1 - promotion_discount)
                    quantity = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
                    
                    item_total = price * quantity
                    total_amount += item_total
                
                # Channel might differ from preferred occasionally
                if np.random.random() < 0.2:
                    channel = np.random.choice(['web', 'mobile', 'store'])
                else:
                    channel = preferred_channel
                
                transactions.append({
                    'transaction_id': transaction_id,
                    'customer_id': customer['customer_id'],
                    'transaction_date': txn_date,
                    'amount': round(total_amount, 2),
                    'num_items': len(selected_products),
                    'channel': channel,
                    'is_promotion': is_promotion,
                    'discount_applied': promotion_discount
                })
                
                transaction_id += 1
        
        return pd.DataFrame(transactions)
    
    def _get_seasonal_multiplier(self, month: int) -> float:
        """Calculate seasonal multiplier for a given month."""
        # Higher spending in Nov-Dec (holidays), lower in Jan-Feb
        seasonal_pattern = {
            1: 0.8, 2: 0.85, 3: 0.95, 4: 1.0,
            5: 1.0, 6: 0.95, 7: 1.05, 8: 1.0,
            9: 0.95, 10: 1.1, 11: 1.3, 12: 1.4
        }
        base = seasonal_pattern.get(month, 1.0)
        # Add some randomness
        return base * np.random.uniform(0.95, 1.05)
    
    def _is_promotion_day(self, date: datetime.date) -> bool:
        """Check if date is a promotion day."""
        # Black Friday (4th Friday of November)
        if date.month == 11 and date.weekday() == 4 and 22 <= date.day <= 28:
            return True
        
        # Cyber Monday
        if date.month == 11 and date.weekday() == 0 and 25 <= date.day <= 30:
            return True
        
        # Christmas week
        if date.month == 12 and 20 <= date.day <= 26:
            return True
        
        # Random promotion days
        return np.random.random() < self.config.promotion_probability
    
    def generate_customer_events(
        self, 
        customers_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate customer engagement events (web visits, email opens, etc.)."""
        events = []
        event_id = 1
        
        event_types = ['page_view', 'email_open', 'email_click', 'search', 'cart_add']
        
        for _, customer in customers_df.iterrows():
            # Number of events varies by segment
            if customer['segment'] == 'premium':
                num_events = np.random.poisson(50)
            elif customer['segment'] == 'frequent':
                num_events = np.random.poisson(40)
            else:
                num_events = np.random.poisson(20)
            
            for _ in range(num_events):
                days_ago = int(np.random.exponential(15))
                days_ago = min(days_ago, self.config.date_range_days)
                
                event_date = self.today - timedelta(days=days_ago)
                
                # Event type distribution
                if customer['email_subscriber']:
                    event_probs = [0.5, 0.2, 0.1, 0.15, 0.05]
                else:
                    event_probs = [0.6, 0.0, 0.0, 0.3, 0.1]
                
                event_type = np.random.choice(event_types, p=event_probs)
                
                events.append({
                    'event_id': event_id,
                    'customer_id': customer['customer_id'],
                    'event_date': event_date,
                    'event_type': event_type,
                    'channel': customer['preferred_channel']
                })
                
                event_id += 1
        
        return pd.DataFrame(events)
    
    def generate_streaming_data(
        self,
        customers_df: pd.DataFrame,
        num_events: int = 1000
    ) -> pd.DataFrame:
        """Generate real-time streaming events for online feature updates."""
        events = []
        
        for _ in range(num_events):
            customer = customers_df.sample(1).iloc[0]
            
            # Recent events only (last 24 hours)
            minutes_ago = np.random.randint(0, 1440)
            event_timestamp = datetime.now() - timedelta(minutes=minutes_ago)
            
            event_types = ['page_view', 'product_view', 'add_to_cart', 'purchase']
            event_probs = [0.6, 0.25, 0.1, 0.05]
            
            event_type = np.random.choice(event_types, p=event_probs)
            
            events.append({
                'event_id': f'stream_{_}',
                'customer_id': customer['customer_id'],
                'timestamp': event_timestamp,
                'event_type': event_type,
                'session_id': f'session_{np.random.randint(1000, 9999)}'
            })
        
        return pd.DataFrame(events)


class ABTestingDataGenerator:
    """Generate synthetic data for A/B testing scenarios."""
    
    def __init__(self, num_customers: int = 10000, seed: int = 42):
        self.num_customers = num_customers
        np.random.seed(seed)
    
    def generate_ab_test_data(
        self,
        control_conversion_rate: float = 0.10,
        treatment_lift: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate A/B test data with known treatment effect.
        
        Args:
            control_conversion_rate: Baseline conversion rate
            treatment_lift: Relative lift from treatment (0.15 = 15% lift)
        
        Returns:
            Tuple of (assignments_df, outcomes_df)
        """
        # Random assignment
        assignments = []
        outcomes = []
        
        for customer_id in range(1, self.num_customers + 1):
            # 50/50 split
            variant = np.random.choice(['control', 'treatment'])
            assignment_date = datetime.now().date() - timedelta(
                days=np.random.randint(1, 30)
            )
            
            assignments.append({
                'customer_id': customer_id,
                'variant': variant,
                'assignment_date': assignment_date
            })
            
            # Outcome based on variant
            if variant == 'control':
                conversion_prob = control_conversion_rate
            else:
                conversion_prob = control_conversion_rate * (1 + treatment_lift)
            
            converted = np.random.random() < conversion_prob
            
            if converted:
                # Generate conversion value
                if variant == 'treatment':
                    avg_value = 150  # Treatment might also increase AOV
                else:
                    avg_value = 130
                
                value = np.random.gamma(2, avg_value / 2)
                
                outcomes.append({
                    'customer_id': customer_id,
                    'converted': 1,
                    'conversion_value': round(value, 2),
                    'conversion_date': assignment_date + timedelta(
                        days=np.random.randint(0, 7)
                    )
                })
            else:
                outcomes.append({
                    'customer_id': customer_id,
                    'converted': 0,
                    'conversion_value': 0.0,
                    'conversion_date': None
                })
        
        return pd.DataFrame(assignments), pd.DataFrame(outcomes)


def generate_complete_ecommerce_dataset(
    config: AdvancedDataConfig
) -> Dict[str, pd.DataFrame]:
    """Generate complete e-commerce dataset with all tables."""
    print(f"Generating comprehensive e-commerce dataset...")
    
    generator = AdvancedSyntheticDataGenerator(config)
    
    # Generate all data
    products_df = generator.generate_product_catalog()
    customers_df = generator.generate_customer_segments_advanced()
    transactions_df = generator.generate_transactions_with_seasonality(
        customers_df, products_df
    )
    events_df = generator.generate_customer_events(customers_df)
    streaming_df = generator.generate_streaming_data(customers_df)
    
    print(f"✅ Generated {len(customers_df):,} customers")
    print(f"✅ Generated {len(products_df):,} products")
    print(f"✅ Generated {len(transactions_df):,} transactions")
    print(f"✅ Generated {len(events_df):,} events")
    print(f"✅ Generated {len(streaming_df):,} streaming events")
    
    return {
        'customers': customers_df,
        'products': products_df,
        'transactions': transactions_df,
        'events': events_df,
        'streaming': streaming_df
    }


def main():
    """Example usage of advanced data generation."""
    
    # Generate comprehensive dataset
    config = AdvancedDataConfig(
        num_customers=5000,
        num_products=100,
        num_categories=10,
        date_range_days=365
    )
    
    datasets = generate_complete_ecommerce_dataset(config)
    
    # Display summaries
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print("\nCustomer Segments:")
    print(datasets['customers']['segment'].value_counts())
    
    print("\nProduct Categories:")
    print(datasets['products']['category'].value_counts())
    
    print("\nTransaction Channels:")
    print(datasets['transactions']['channel'].value_counts())
    
    print("\nPromotional Transactions:")
    promo_pct = datasets['transactions']['is_promotion'].mean()
    print(f"  {promo_pct:.1%} of transactions had promotions")
    
    # A/B Test example
    print("\n" + "="*60)
    print("A/B TEST DATA GENERATION")
    print("="*60)
    
    ab_generator = ABTestingDataGenerator(num_customers=2000)
    assignments, outcomes = ab_generator.generate_ab_test_data(
        control_conversion_rate=0.10,
        treatment_lift=0.20
    )
    
    # Analyze results
    results = assignments.merge(outcomes, on='customer_id')
    
    control_rate = results[results['variant'] == 'control']['converted'].mean()
    treatment_rate = results[results['variant'] == 'treatment']['converted'].mean()
    
    print(f"\nControl conversion rate: {control_rate:.2%}")
    print(f"Treatment conversion rate: {treatment_rate:.2%}")
    print(f"Observed lift: {(treatment_rate/control_rate - 1):.1%}")


if __name__ == "__main__":
    main()