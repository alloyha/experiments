"""
Apache Spark Processing Engine Implementation
===========================================

Implementation of Spark processing engine for large-scale batch processing.
"""

from typing import Dict, List, Any
from .abstract_engine import ProcessingEngineBackend
from ..storage.abstract_backend import StorageType


class SparkProcessingEngine(ProcessingEngineBackend):
    """Apache Spark distributed processing engine"""
    
    def generate_query(self, table_metadata: Dict[str, Any], 
                      operation: str, **kwargs) -> str:
        """Generate Spark SQL/PySpark code"""
        
        if operation == 'batch_etl':
            table_name = table_metadata['name']
            entity_type = table_metadata.get('entity_type', 'dimension')
            
            if entity_type in ['fact', 'transaction_fact']:
                return f"""
# Spark Batch ETL for {table_name}
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Initialize Spark session with Iceberg support
spark = SparkSession.builder \\
    .appName("{table_name}_batch_etl") \\
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \\
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \\
    .config("spark.sql.catalog.spark_catalog.type", "hive") \\
    .getOrCreate()

# Read from staging with schema validation
staging_df = spark.read \\
    .option("mergeSchema", "true") \\
    .format("parquet") \\
    .load("s3://data-lake/staging/{table_name}/")

# Data quality checks and transformations
processed_df = (staging_df
    .filter(col("amount").isNotNull() & (col("amount") > 0))
    .withColumn("partition_date", date_format(col("transaction_timestamp"), "yyyy-MM-dd"))
    .withColumn("transaction_hour", hour(col("transaction_timestamp")))
    .withColumn("created_at", current_timestamp())
    .withColumn("etl_batch_id", lit("{{{{ batch_id }}}}"))
    .dropDuplicates(["transaction_id"])
)

# Write to Iceberg table with partitioning and optimization
(processed_df
    .write
    .format("iceberg")
    .mode("append")
    .partitionBy("partition_date")
    .option("fanout-enabled", "true")
    .saveAsTable("analytics.{table_name}")
)

# Update table statistics
spark.sql("CALL system.rewrite_data_files('analytics.{table_name}')")
spark.sql("CALL system.rewrite_manifests('analytics.{table_name}')")

print(f"Successfully processed {{processed_df.count()}} records for {table_name}")
"""
            
        elif operation == 'scd2_processing':
            return f"""
# Spark SCD2 Processing for {table_metadata['name']}
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("scd2_processing").getOrCreate()

# Read current dimension data
current_df = spark.read.format("iceberg").load("analytics.{table_metadata['name']}")

# Read new source data
new_df = spark.read.format("parquet").load("staging/{table_metadata['name']}/")

# Get current active records
current_active = current_df.filter(col("is_current") == True)

# Detect changes using hash comparison
current_with_hash = current_active.withColumn(
    "record_hash", 
    sha2(concat_ws("|", *[col(c) for c in new_df.columns if c != "updated_at"]), 256)
)

new_with_hash = new_df.withColumn(
    "record_hash",
    sha2(concat_ws("|", *[col(c) for c in new_df.columns if c != "updated_at"]), 256)
)

# Find records that need to be expired (changed or deleted)
to_expire = (current_with_hash
    .join(new_with_hash, ["customer_id"], "left_anti")
    .unionByName(
        current_with_hash.join(
            new_with_hash.select("customer_id", "record_hash"), 
            ["customer_id"], "inner"
        ).filter(
            current_with_hash.record_hash != new_with_hash.record_hash
        )
    )
    .withColumn("valid_to", current_timestamp())
    .withColumn("is_current", lit(False))
    .withColumn("updated_at", current_timestamp())
    .drop("record_hash")
)

# Find new and changed records
window_spec = Window.partitionBy("customer_id").orderBy(desc("updated_at"))
new_records = (new_with_hash
    .withColumn("customer_sk", 
        monotonically_increasing_id() + lit(1000000))  # Ensure unique SKs
    .withColumn("valid_from", current_timestamp())
    .withColumn("valid_to", lit(None).cast("timestamp"))
    .withColumn("is_current", lit(True))
    .withColumn("created_at", current_timestamp())
    .withColumn("updated_at", current_timestamp())
    .drop("record_hash")
)

# Combine all records: unchanged + expired + new
final_df = (
    current_df.filter(col("is_current") == False)  # Keep historical records
    .unionByName(to_expire)  # Add expired records
    .unionByName(new_records)  # Add new/changed records
)

# Write back to Iceberg table
(final_df
    .write
    .format("iceberg")
    .mode("overwrite")
    .saveAsTable("analytics.{table_metadata['name']}")
)

print(f"SCD2 processing complete for {table_metadata['name']}")
"""
        
        elif operation == 'streaming_ingestion':
            return f"""
# Spark Streaming Ingestion for {table_metadata['name']}
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder \\
    .appName("{table_metadata['name']}_streaming") \\
    .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoints/") \\
    .getOrCreate()

# Define schema for incoming data
schema = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("customer_id", StringType(), False), 
    StructField("amount", DecimalType(18,2), False),
    StructField("transaction_timestamp", TimestampType(), False)
])

# Read from Kafka stream
stream_df = (spark
    .readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", "{table_metadata['name']}_events")
    .option("startingOffsets", "latest")
    .load()
)

# Parse JSON and apply transformations
processed_stream = (stream_df
    .select(from_json(col("value").cast("string"), schema).alias("data"))
    .select("data.*")
    .withColumn("partition_date", date_format(col("transaction_timestamp"), "yyyy-MM-dd"))
    .withColumn("created_at", current_timestamp())
    .withWatermark("transaction_timestamp", "10 minutes")
)

# Write to Iceberg table
query = (processed_stream
    .writeStream
    .format("iceberg")
    .outputMode("append")
    .option("checkpointLocation", f"/tmp/checkpoints/{table_metadata['name']}")
    .option("path", f"s3://data-lake/analytics/{table_metadata['name']}")
    .trigger(processingTime="30 seconds")
    .start()
)

query.awaitTermination()
"""
        
        return f"# Spark {operation} for {table_metadata.get('name', 'unknown')}"
    
    def supports_storage_backend(self, storage_type: StorageType) -> bool:
        """Spark supports many storage backends"""
        supported = {
            StorageType.ICEBERG,
            StorageType.POSTGRESQL,  # Through JDBC
            StorageType.CLICKHOUSE,  # Through JDBC
            # Note: Would support Delta Lake, Parquet files, etc.
        }
        return storage_type in supported
    
    def estimate_query_performance(self, query: str, 
                                 table_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate Spark performance"""
        entity_type = table_metadata.get('entity_type', 'dimension')
        estimated_size = table_metadata.get('estimated_size', 'medium')
        
        is_streaming = 'readStream' in query or 'writeStream' in query
        has_joins = 'join' in query.lower()
        has_aggregations = any(func in query.lower() for func in ['groupby', 'agg', 'count'])
        
        if is_streaming:
            return {
                'complexity': 'high',
                'estimated_execution_time': 'continuous',
                'scalability': 'excellent',
                'parallelism': 'high',
                'optimal_for': 'streaming_processing',
                'throughput': 'millions of events/second'
            }
        elif entity_type in ['fact', 'transaction_fact'] and estimated_size in ['large', 'very_large']:
            return {
                'complexity': 'medium',
                'estimated_execution_time': '5-30 minutes',
                'scalability': 'excellent',
                'parallelism': 'high', 
                'optimal_for': 'large_batch_processing',
                'throughput': 'GB/second processing'
            }
        else:
            return {
                'complexity': 'low',
                'estimated_execution_time': '1-5 minutes',
                'scalability': 'good',
                'parallelism': 'medium',
                'optimal_for': 'medium_batch_processing'
            }
    
    def get_optimal_query_patterns(self) -> List[str]:
        """Get optimal patterns for Spark"""
        return [
            'large_batch_etl',
            'streaming_processing',
            'complex_transformations',
            'data_quality_checks',
            'scd2_processing',
            'machine_learning_pipelines',
            'aggregation_heavy_workloads',
            'distributed_joins',
            'parquet_processing'
        ]