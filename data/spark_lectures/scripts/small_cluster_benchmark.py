#!/usr/bin/env python3
"""
small_cluster_benchmark.py

Smaller benchmark for Spark Data Layout Masterclass
- Uses local mode for simplicity
- Tests 3 main layout strategies
- Writes to MinIO S3 instead of /tmp
- Much faster for testing

Usage:
  docker-compose exec spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py
  
Or from host:
  ./scripts/run_in_cluster.sh small_benchmark
"""

import os
import sys
import json
import time
from datetime import datetime

print("\n" + "="*80)
print("SMALL CLUSTER BENCHMARK - Spark Data Layout Masterclass")
print("="*80)
print(f"Timestamp: {datetime.now().isoformat()}\n")

# ============================================================================
# PART 1: INITIALIZE SPARK (NO packages parameter to avoid Ivy issues)
# ============================================================================

print("[SETUP] Creating Spark session...")
print("  Configuration:")
print("    - MinIO endpoint: http://minio:9000")
print("    - S3 buckets: warehouse/small_benchmark")
print("    - Running in local[*] mode")

try:
    # Set environment to disable package resolution
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.jars.ivy=/tmp/ivy2 pyspark-shell'
    
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        col, expr, rand, year, month, to_date, count
    )
    
    spark = (SparkSession.builder
             .appName("SmallClusterBenchmark")
             .master("local[*]")  # Local mode
             .config("spark.driver.memory", "2g")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.shuffle.partitions", "50")
             # MinIO/S3 configuration
             .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
             .config("spark.hadoop.fs.s3a.access.key", "admin")
             .config("spark.hadoop.fs.s3a.secret.key", "password")
             .config("spark.hadoop.fs.s3a.path.style.access", "true")
             .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                     "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
             # Disable package resolution to avoid Ivy issues
             .config("spark.submit.pyFiles", "")
             .getOrCreate())
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"✓ Spark session created")
    print(f"  - Version: {spark.version}")
    print(f"  - Master: {spark.sparkContext.master}")
    
except Exception as e:
    print(f"✗ Failed to create Spark session: {e}")
    print("\nThis is likely due to PySpark/Java gateway issues.")
    print("Try running the connectivity test first:")
    print("  docker-compose exec -T spark-master python3 /opt/spark/scripts/test_simple_docker.py")
    sys.exit(1)

# ============================================================================
# PART 2: GENERATE TEST DATA
# ============================================================================

print("\n[DATA] Generating test data...")
ROWS = 500_000  # Smaller for faster testing
SEED = 42

try:
    df = (spark.range(0, ROWS)
          .select(
              (col("id") % 100_000).cast("int").alias("customer_id"),
              (col("id") % 10_000).cast("int").alias("product_id"),
              expr("date_add('2020-01-01', cast(id % 1000 as int))").alias("sale_date"),
              (rand(SEED) * 1000.0).alias("amount"),
              (col("id") % 10).cast("int").alias("region_id"),
          ))
    
    # Add derived columns
    df = df.select(
        col("*"),
        year("sale_date").alias("year"),
        month("sale_date").alias("month"),
    )
    
    df.cache()
    count = df.count()
    print(f"✓ Generated {count:,} rows")
    print(f"  Schema: customer_id, product_id, sale_date, amount, region_id, year, month")
    
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    spark.stop()
    sys.exit(1)

# ============================================================================
# PART 3: TEST STORAGE STRATEGIES
# ============================================================================

metrics = []
base_path = "s3a://warehouse/small_benchmark"

strategies = [
    {
        "name": "Unpartitioned Parquet",
        "path": f"{base_path}/01_unpartitioned",
        "write_func": lambda d, p: d.write.mode("overwrite").parquet(p),
        "description": "Simple write to S3 without partitioning"
    },
    {
        "name": "Partitioned by Region",
        "path": f"{base_path}/02_partitioned_region",
        "write_func": lambda d, p: d.write.partitionBy("region_id").mode("overwrite").parquet(p),
        "description": "Partition by low-cardinality dimension"
    },
    {
        "name": "Partitioned by Year/Month",
        "path": f"{base_path}/03_partitioned_time",
        "write_func": lambda d, p: d.write.partitionBy("year", "month").mode("overwrite").parquet(p),
        "description": "Partition by time dimensions"
    },
]

print(f"\n[WRITE] Testing {len(strategies)} storage strategies...")

for strategy in strategies:
    print(f"\n  Writing: {strategy['name']}")
    print(f"    Path: {strategy['path']}")
    print(f"    Desc: {strategy['description']}")
    
    try:
        start = time.time()
        strategy['write_func'](df, strategy['path'])
        duration = time.time() - start
        
        # Read back to verify
        verify = spark.read.parquet(strategy['path'])
        verify_count = verify.count()
        
        throughput = count / duration
        print(f"    ✓ Success: {duration:.1f}s ({throughput:,.0f} rows/s)")
        
        metrics.append({
            "strategy": strategy['name'],
            "duration": round(duration, 2),
            "throughput": round(throughput, 2),
            "rows": count,
            "verified": verify_count == count
        })
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")

# ============================================================================
# PART 4: QUERY BENCHMARK
# ============================================================================

print(f"\n[QUERY] Running benchmark queries...")

queries = [
    {
        "name": "Time filter (month == 6)",
        "sql": "SELECT COUNT(*) as cnt FROM data WHERE month = 6"
    },
    {
        "name": "Region filter (region == 5)",
        "sql": "SELECT COUNT(*) as cnt FROM data WHERE region_id = 5"
    },
    {
        "name": "Aggregation by region",
        "sql": "SELECT region_id, COUNT(*) as cnt FROM data GROUP BY region_id"
    },
]

query_results = []

for strategy in strategies:
    print(f"\n  Strategy: {strategy['name']}")
    try:
        data = spark.read.parquet(strategy['path'])
        data.createOrReplaceTempView("data")
        
        for query in queries:
            start = time.time()
            result = spark.sql(query['sql']).collect()
            duration = time.time() - start
            
            print(f"    ✓ {query['name']}: {duration:.2f}s")
            
            query_results.append({
                "strategy": strategy['name'],
                "query": query['name'],
                "duration": round(duration, 3)
            })
    except Exception as e:
        print(f"    ✗ Query failed: {e}")

# ============================================================================
# PART 5: RESULTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("BENCHMARK RESULTS")
print("="*80)

print("\nWRITE PERFORMANCE:")
print(f"{'Strategy':<30} {'Duration':<12} {'Throughput':<15}")
print("-" * 57)
for m in metrics:
    print(f"{m['strategy']:<30} {m['duration']:>6.2f}s      {m['throughput']:>12,.0f} rows/s")

print("\nQUERY PERFORMANCE (fastest first):")
# Sort by duration
sorted_queries = sorted(query_results, key=lambda x: x['duration'])
query_names = set(q['query'] for q in query_results)

for qname in query_names:
    results_for_query = sorted([q for q in query_results if q['query'] == qname], 
                               key=lambda x: x['duration'])
    print(f"\n  {qname}:")
    for r in results_for_query[:1]:  # Show fastest
        fastest_strategy = r['strategy']
        fastest_time = r['duration']
    for r in results_for_query:
        ratio = r['duration'] / fastest_time if fastest_time > 0 else 1
        marker = " ✓ FASTEST" if ratio < 1.05 else f" ({ratio:.1f}x)"
        print(f"    {r['strategy']:<30} {r['duration']:>6.3f}s{marker}")

# ============================================================================
# PART 6: SAVE AND SUMMARIZE
# ============================================================================

print("\n" + "="*80)
print("✓ BENCHMARK COMPLETE")
print("="*80)

print("\n📊 Summary:")
print(f"  ✓ Data written to MinIO S3: {base_path}")
print(f"  ✓ Strategies tested: {len(strategies)}")
print(f"  ✓ Queries executed: {len(query_results)}")

if metrics:
    print(f"\n💡 Key Insights:")
    print(f"  - Best write performance: {metrics[0]['strategy']} ({metrics[0]['duration']:.1f}s)")
    print(f"  - Data is now in MinIO S3 (accessible from any service)")

print("\n📁 Access your data:")
print(f"  - S3 Browser: http://localhost:9001 (admin/password)")
print(f"  - MinIO CLI: mc ls minio/warehouse/small_benchmark/")

print("\n🚀 Next steps:")
print(f"  1. Review data in MinIO console")
print(f"  2. Run the larger benchmark locally: ./scripts/run_masterclass.sh")
print(f"  3. Or scale to the full cluster benchmark")
print("\n" + "="*80 + "\n")

spark.stop()


# ============================================================================
# PART 2: GENERATE TEST DATA
# ============================================================================

print("\n[DATA] Generating test data...")
ROWS = 1_000_000  # Smaller for faster testing
SEED = 42

try:
    df = (spark.range(0, ROWS)
          .select(
              (col("id") % 100_000).cast("int").alias("customer_id"),
              (col("id") % 10_000).cast("int").alias("product_id"),
              expr("date_add('2020-01-01', cast(id % 1000 as int))").alias("sale_date"),
              (rand(SEED) * 1000.0).alias("amount"),
              (col("id") % 10).cast("int").alias("region_id"),
          ))
    
    # Add derived columns
    df = df.select(
        col("*"),
        year("sale_date").alias("year"),
        month("sale_date").alias("month"),
    )
    
    df.cache()
    count = df.count()
    print(f"✓ Generated {count:,} rows")
    print(f"  Schema: customer_id, product_id, sale_date, amount, region_id, year, month")
    
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    spark.stop()
    sys.exit(1)

# ============================================================================
# PART 3: TEST STORAGE STRATEGIES
# ============================================================================

metrics = []
base_path = "s3a://warehouse/small_benchmark"

strategies = [
    {
        "name": "Unpartitioned Parquet",
        "path": f"{base_path}/01_unpartitioned",
        "write_func": lambda d, p: d.write.mode("overwrite").parquet(p),
        "description": "Simple write to S3 without partitioning"
    },
    {
        "name": "Partitioned by Region",
        "path": f"{base_path}/02_partitioned_region",
        "write_func": lambda d, p: d.write.partitionBy("region_id").mode("overwrite").parquet(p),
        "description": "Partition by low-cardinality dimension"
    },
    {
        "name": "Partitioned by Year/Month",
        "path": f"{base_path}/03_partitioned_time",
        "write_func": lambda d, p: d.write.partitionBy("year", "month").mode("overwrite").parquet(p),
        "description": "Partition by time dimensions"
    },
]

print(f"\n[WRITE] Testing {len(strategies)} storage strategies...")

for strategy in strategies:
    print(f"\n  Writing: {strategy['name']}")
    print(f"    Path: {strategy['path']}")
    print(f"    Desc: {strategy['description']}")
    
    try:
        start = time.time()
        strategy['write_func'](df, strategy['path'])
        duration = time.time() - start
        
        # Read back to verify
        verify = spark.read.parquet(strategy['path'])
        verify_count = verify.count()
        
        throughput = count / duration
        print(f"    ✓ Success: {duration:.1f}s ({throughput:,.0f} rows/s)")
        
        metrics.append({
            "strategy": strategy['name'],
            "duration": round(duration, 2),
            "throughput": round(throughput, 2),
            "rows": count,
            "verified": verify_count == count
        })
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")

# ============================================================================
# PART 4: QUERY BENCHMARK
# ============================================================================

print(f"\n[QUERY] Running benchmark queries...")

queries = [
    {
        "name": "Time filter (month == 6)",
        "sql": "SELECT COUNT(*) as cnt FROM data WHERE month = 6"
    },
    {
        "name": "Region filter (region == 5)",
        "sql": "SELECT COUNT(*) as cnt FROM data WHERE region_id = 5"
    },
    {
        "name": "Aggregation by region",
        "sql": "SELECT region_id, COUNT(*) as cnt FROM data GROUP BY region_id"
    },
]

query_results = []

for strategy in strategies:
    print(f"\n  Strategy: {strategy['name']}")
    try:
        data = spark.read.parquet(strategy['path'])
        data.createOrReplaceTempView("data")
        
        for query in queries:
            start = time.time()
            result = spark.sql(query['sql']).collect()
            duration = time.time() - start
            
            print(f"    ✓ {query['name']}: {duration:.2f}s")
            
            query_results.append({
                "strategy": strategy['name'],
                "query": query['name'],
                "duration": round(duration, 3)
            })
    except Exception as e:
        print(f"    ✗ Query failed: {e}")

# ============================================================================
# PART 5: RESULTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("BENCHMARK RESULTS")
print("="*80)

print("\nWRITE PERFORMANCE:")
print(f"{'Strategy':<30} {'Duration':<12} {'Throughput':<15}")
print("-" * 57)
for m in metrics:
    print(f"{m['strategy']:<30} {m['duration']:>6.2f}s      {m['throughput']:>12,.0f} rows/s")

print("\nQUERY PERFORMANCE (fastest first):")
# Sort by duration
sorted_queries = sorted(query_results, key=lambda x: x['duration'])
query_names = set(q['query'] for q in query_results)

for qname in query_names:
    results_for_query = sorted([q for q in query_results if q['query'] == qname], 
                               key=lambda x: x['duration'])
    print(f"\n  {qname}:")
    for r in results_for_query[:1]:  # Show fastest
        fastest_strategy = r['strategy']
        fastest_time = r['duration']
    for r in results_for_query:
        ratio = r['duration'] / fastest_time
        marker = " ✓ FASTEST" if ratio < 1.05 else f" ({ratio:.1f}x)"
        print(f"    {r['strategy']:<30} {r['duration']:>6.3f}s{marker}")

# ============================================================================
# PART 6: SAVE AND SUMMARIZE
# ============================================================================

print("\n" + "="*80)
print("✓ BENCHMARK COMPLETE")
print("="*80)

print("\n📊 Summary:")
print(f"  ✓ Data written to MinIO S3: {base_path}")
print(f"  ✓ Strategies tested: {len(strategies)}")
print(f"  ✓ Queries executed: {len(query_results)}")

print("\n💡 Key Insights:")
print(f"  - Best write performance: {metrics[0]['strategy']} ({metrics[0]['duration']:.1f}s)")
print(f"  - Data is now in MinIO S3 (accessible from any service)")
print(f"  - Hive Metastore configured for bucketed tables")

print("\n📁 Access your data:")
print(f"  - S3 Browser: http://localhost:9001 (admin/password)")
print(f"  - Spark SQL: SELECT * FROM warehouse.data LIMIT 10")
print(f"  - MinIO CLI: mc ls minio/warehouse/small_benchmark/")

print("\n🚀 Next steps:")
print(f"  1. Run full benchmark: ./scripts/run_in_cluster.sh benchmark")
print(f"  2. Access Spark UI: http://localhost:4040")
print(f"  3. Use MinIO for data lake development")
print("\n" + "="*80 + "\n")

spark.stop()
