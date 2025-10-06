"""
Real-World Use Cases for PostgreSQL Schema Generator
===================================================

This module demonstrates practical use cases for the schema generator
in various business scenarios including e-commerce, financial services,
SaaS platforms, and data analytics pipelines.
"""

import json
import logging
from datetime import datetime, timedelta
from .schema_generator_client import SchemaGenerator, ConnectionConfig, create_dimension_metadata, create_fact_metadata

logger = logging.getLogger(__name__)


class ECommerceDataWarehouse:
    """
    E-commerce dimensional model use case
    
    Demonstrates creating a complete e-commerce data warehouse with:
    - Customer dimensions (SCD2 for tracking changes)
    - Product catalog dimensions
    - Order fact tables
    - Inventory snapshots
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
        self.schema = "ecommerce_dw"
    
    def create_customer_dimension(self):
        """Create customer dimension with SCD2 to track changes"""
        return create_dimension_metadata(
            name="dim_customers",
            grain="One row per customer per time period",
            primary_keys=["customer_id"],
            physical_columns=[
                {"name": "customer_id", "type": "text"},
                {"name": "email", "type": "text"},
                {"name": "first_name", "type": "text"},
                {"name": "last_name", "type": "text"},
                {"name": "registration_date", "type": "date"},
                {"name": "customer_segment", "type": "text"},
                {"name": "lifetime_value", "type": "numeric(10,2)"}
            ],
            scd_type="SCD2"
        )
    
    def create_product_dimension(self):
        """Create product dimension with full catalog info"""
        return create_dimension_metadata(
            name="dim_products",
            grain="One row per product",
            primary_keys=["product_id"],
            physical_columns=[
                {"name": "product_id", "type": "text"},
                {"name": "sku", "type": "text"},
                {"name": "product_name", "type": "text"},
                {"name": "category", "type": "text"},
                {"name": "subcategory", "type": "text"},
                {"name": "brand", "type": "text"},
                {"name": "unit_cost", "type": "numeric(10,2)"},
                {"name": "list_price", "type": "numeric(10,2)"}
            ],
            scd_type="SCD1"
        )
    
    def create_date_dimension(self):
        """Create date dimension for time-based analysis"""
        return create_dimension_metadata(
            name="dim_dates",
            grain="One row per date",
            primary_keys=["date_key"],
            physical_columns=[
                {"name": "date_key", "type": "date"},
                {"name": "year", "type": "integer"},
                {"name": "quarter", "type": "integer"},
                {"name": "month", "type": "integer"},
                {"name": "week", "type": "integer"},
                {"name": "day_of_week", "type": "integer"},
                {"name": "is_weekend", "type": "boolean"},
                {"name": "is_holiday", "type": "boolean"}
            ]
        )
    
    def create_orders_fact(self):
        """Create orders fact table"""
        return create_fact_metadata(
            name="fact_orders",
            grain="One row per order",
            dimension_references=[
                {"dimension": "dim_customers", "fk_column": "customer_sk"},
                {"dimension": "dim_dates", "fk_column": "order_date_sk"}
            ],
            measures=[
                {"name": "order_amount", "type": "numeric(10,2)", "additivity": "additive"},
                {"name": "tax_amount", "type": "numeric(10,2)", "additivity": "additive"},
                {"name": "shipping_amount", "type": "numeric(10,2)", "additivity": "additive"},
                {"name": "discount_amount", "type": "numeric(10,2)", "additivity": "additive"},
                {"name": "item_count", "type": "integer", "additivity": "additive"}
            ],
            physical_columns=[
                {"name": "order_id", "type": "text"},
                {"name": "order_timestamp", "type": "timestamptz"}
            ]
        )
    
    def create_order_items_fact(self):
        """Create order items fact table for line-level analysis"""
        return create_fact_metadata(
            name="fact_order_items",
            grain="One row per order line item",
            dimension_references=[
                {"dimension": "dim_customers", "fk_column": "customer_sk"},
                {"dimension": "dim_products", "fk_column": "product_sk"},
                {"dimension": "dim_dates", "fk_column": "order_date_sk"}
            ],
            measures=[
                {"name": "quantity", "type": "integer", "additivity": "additive"},
                {"name": "unit_price", "type": "numeric(10,2)", "additivity": "non_additive"},
                {"name": "line_total", "type": "numeric(10,2)", "additivity": "additive"},
                {"name": "unit_cost", "type": "numeric(10,2)", "additivity": "non_additive"},
                {"name": "profit_margin", "type": "numeric(10,2)", "additivity": "additive"}
            ],
            physical_columns=[
                {"name": "order_id", "type": "text"},
                {"name": "line_number", "type": "integer"}
            ]
        )
    
    def deploy_complete_model(self, execute: bool = False):
        """Deploy the complete e-commerce data warehouse"""
        logger.info("🛒 Deploying E-commerce Data Warehouse...")
        
        # Create pipeline metadata
        pipeline = {
            "schema": self.schema,
            "tables": [
                self.create_date_dimension(),
                self.create_customer_dimension(),
                self.create_product_dimension(),
                self.create_orders_fact(),
                self.create_order_items_fact()
            ]
        }
        
        # Generate and optionally execute
        ddl, dml = self.generator.create_dimensional_model(pipeline, execute=execute)
        
        logger.info("📊 E-commerce Data Warehouse ready!")
        return ddl, dml


class SaaSMetricsDataMart:
    """
    SaaS business metrics use case
    
    Demonstrates creating metrics for:
    - Customer acquisition and churn
    - Subscription revenue tracking
    - Feature usage analytics
    - Support ticket analysis
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
        self.schema = "saas_metrics"
    
    def create_subscription_fact(self):
        """Monthly subscription snapshots for MRR analysis"""
        return create_fact_metadata(
            name="fact_subscription_monthly",
            grain="One row per subscription per month",
            dimension_references=[
                {"dimension": "dim_customers", "fk_column": "customer_sk"},
                {"dimension": "dim_plans", "fk_column": "plan_sk"},
                {"dimension": "dim_dates", "fk_column": "month_sk"}
            ],
            measures=[
                {"name": "mrr", "type": "numeric(10,2)", "additivity": "additive"},
                {"name": "seats", "type": "integer", "additivity": "additive"},
                {"name": "is_active", "type": "boolean", "additivity": "semi_additive"},
                {"name": "months_since_signup", "type": "integer", "additivity": "non_additive"}
            ]
        )
    
    def create_feature_usage_fact(self):
        """Daily feature usage for product analytics"""
        return create_fact_metadata(
            name="fact_feature_usage_daily",
            grain="One row per customer per feature per day",
            dimension_references=[
                {"dimension": "dim_customers", "fk_column": "customer_sk"},
                {"dimension": "dim_features", "fk_column": "feature_sk"},
                {"dimension": "dim_dates", "fk_column": "date_sk"}
            ],
            measures=[
                {"name": "usage_count", "type": "integer", "additivity": "additive"},
                {"name": "session_duration_seconds", "type": "integer", "additivity": "additive"},
                {"name": "unique_users", "type": "integer", "additivity": "semi_additive"}
            ]
        )
    
    def deploy_saas_metrics(self, execute: bool = False):
        """Deploy SaaS metrics data mart"""
        logger.info("📈 Deploying SaaS Metrics Data Mart...")
        
        pipeline = {
            "schema": self.schema,
            "tables": [
                create_dimension_metadata("dim_plans", "One row per subscription plan", ["plan_id"]),
                create_dimension_metadata("dim_features", "One row per product feature", ["feature_id"]),
                self.create_subscription_fact(),
                self.create_feature_usage_fact()
            ]
        }
        
        return self.generator.create_dimensional_model(pipeline, execute=execute)


class FinancialDataPipeline:
    """
    Financial services use case
    
    Demonstrates:
    - Account transaction processing
    - Risk metrics calculation
    - Regulatory reporting tables
    - Customer portfolio analysis
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
        self.schema = "financial_dw"
    
    def create_account_dimension(self):
        """Customer accounts with SCD2 for regulatory compliance"""
        return create_dimension_metadata(
            name="dim_accounts",
            grain="One row per account per time period",
            primary_keys=["account_number"],
            physical_columns=[
                {"name": "account_number", "type": "text"},
                {"name": "account_type", "type": "text"},
                {"name": "customer_id", "type": "text"},
                {"name": "branch_code", "type": "text"},
                {"name": "currency_code", "type": "text"},
                {"name": "status", "type": "text"},
                {"name": "opened_date", "type": "date"},
                {"name": "risk_rating", "type": "text"}
            ],
            scd_type="SCD2"  # Track all changes for compliance
        )
    
    def create_transactions_fact(self):
        """Daily transaction summaries"""
        return create_fact_metadata(
            name="fact_transactions_daily",
            grain="One row per account per day",
            dimension_references=[
                {"dimension": "dim_accounts", "fk_column": "account_sk"},
                {"dimension": "dim_dates", "fk_column": "date_sk"}
            ],
            measures=[
                {"name": "transaction_count", "type": "integer", "additivity": "additive"},
                {"name": "debit_amount", "type": "numeric(15,2)", "additivity": "additive"},
                {"name": "credit_amount", "type": "numeric(15,2)", "additivity": "additive"},
                {"name": "balance_end_of_day", "type": "numeric(15,2)", "additivity": "semi_additive"},
                {"name": "overdraft_amount", "type": "numeric(15,2)", "additivity": "semi_additive"}
            ]
        )
    
    def create_risk_metrics_fact(self):
        """Monthly risk metrics for portfolio analysis"""
        return create_fact_metadata(
            name="fact_risk_metrics_monthly",
            grain="One row per account per month",
            dimension_references=[
                {"dimension": "dim_accounts", "fk_column": "account_sk"},
                {"dimension": "dim_dates", "fk_column": "month_sk"}
            ],
            measures=[
                {"name": "credit_exposure", "type": "numeric(15,2)", "additivity": "additive"},
                {"name": "var_95", "type": "numeric(15,2)", "additivity": "non_additive"},
                {"name": "expected_loss", "type": "numeric(15,2)", "additivity": "additive"},
                {"name": "days_past_due", "type": "integer", "additivity": "non_additive"}
            ]
        )


class DataAnalyticsPlatform:
    """
    Data analytics platform use case
    
    Demonstrates:
    - Event tracking and user behavior
    - A/B testing results
    - Performance monitoring
    - Content engagement metrics
    """
    
    def __init__(self, generator: SchemaGenerator):
        self.generator = generator
        self.schema = "analytics_platform"
    
    def create_events_fact(self):
        """User events for behavioral analysis"""
        return create_fact_metadata(
            name="fact_user_events",
            grain="One row per user event",
            dimension_references=[
                {"dimension": "dim_users", "fk_column": "user_sk"},
                {"dimension": "dim_sessions", "fk_column": "session_sk"},
                {"dimension": "dim_pages", "fk_column": "page_sk"}
            ],
            measures=[
                {"name": "session_duration", "type": "integer", "additivity": "additive"},
                {"name": "page_views", "type": "integer", "additivity": "additive"},
                {"name": "bounce_rate", "type": "numeric(5,4)", "additivity": "non_additive"}
            ],
            physical_columns=[
                {"name": "event_type", "type": "text"},
                {"name": "event_timestamp", "type": "timestamptz"},
                {"name": "user_agent", "type": "text"},
                {"name": "ip_address", "type": "inet"}
            ]
        )
    
    def create_ab_test_results_fact(self):
        """A/B testing results"""
        return create_fact_metadata(
            name="fact_ab_test_results",
            grain="One row per user per test variant",
            dimension_references=[
                {"dimension": "dim_users", "fk_column": "user_sk"},
                {"dimension": "dim_ab_tests", "fk_column": "test_sk"}
            ],
            measures=[
                {"name": "conversion_flag", "type": "boolean", "additivity": "additive"},
                {"name": "revenue_impact", "type": "numeric(10,2)", "additivity": "additive"},
                {"name": "engagement_score", "type": "numeric(5,2)", "additivity": "non_additive"}
            ],
            physical_columns=[
                {"name": "test_variant", "type": "text"},
                {"name": "enrollment_date", "type": "date"},
                {"name": "conversion_date", "type": "date"}
            ]
        )


def main():
    """
    Demonstrate all use cases
    """
    # Configuration
    config = ConnectionConfig(
        host="localhost",
        database="postgres", 
        username="postgres",
        password="postgres"
    )
    
    generator = SchemaGenerator(config)
    
    # E-commerce Data Warehouse
    print("🛒 E-COMMERCE DATA WAREHOUSE")
    print("=" * 50)
    ecommerce = ECommerceDataWarehouse(generator)
    ddl, dml = ecommerce.deploy_complete_model(execute=False)
    print("✅ E-commerce model generated")
    print(f"DDL length: {len(ddl)} characters")
    print(f"DML length: {len(dml)} characters\n")
    
    # SaaS Metrics
    print("📈 SAAS METRICS DATA MART")
    print("=" * 50)
    saas = SaaSMetricsDataMart(generator)
    ddl, dml = saas.deploy_saas_metrics(execute=False)
    print("✅ SaaS metrics model generated")
    print(f"DDL length: {len(ddl)} characters")
    print(f"DML length: {len(dml)} characters\n")
    
    # Financial Services
    print("🏦 FINANCIAL DATA PIPELINE")
    print("=" * 50)
    financial = FinancialDataPipeline(generator)
    
    # Generate individual components
    account_dim = financial.create_account_dimension()
    transactions_fact = financial.create_transactions_fact()
    
    print(f"✅ Account dimension: {account_dim['name']}")
    print(f"✅ Transactions fact: {transactions_fact['name']}")
    
    # Validate metadata
    try:
        generator.validate_metadata(account_dim)
        generator.validate_metadata(transactions_fact)
        print("✅ Financial model metadata validated\n")
    except Exception as e:
        print(f"❌ Validation error: {e}\n")
    
    # Analytics Platform
    print("📊 DATA ANALYTICS PLATFORM")
    print("=" * 50)
    analytics = DataAnalyticsPlatform(generator)
    
    events_fact = analytics.create_events_fact()
    ab_test_fact = analytics.create_ab_test_results_fact()
    
    print(f"✅ Events fact: {events_fact['name']}")
    print(f"✅ A/B test fact: {ab_test_fact['name']}")
    
    # Generate DDL for specific table
    try:
        ddl = generator.generate_table_ddl(events_fact, "analytics_platform")
        print("✅ Events table DDL generated")
        print(f"DDL preview:\n{ddl[:200]}...\n")
    except Exception as e:
        print(f"❌ DDL generation error: {e}\n")


if __name__ == "__main__":
    main()