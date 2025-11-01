"""
spark_layout_masterclass.py

PySpark Masterclass: Comprehensive benchmark comparing data layout strategies
with detailed analysis, metrics collection, and educational insights.

SETUP REQUIREMENTS:
1. Java 8 or 11 installed: sudo apt-get install openjdk-11-jdk
2. Set JAVA_HOME: export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
3. PySpark installed: pip install pyspark
4. Optional: pip install psutil delta-spark

TROUBLESHOOTING:
- If "Java gateway process exited" error:
  * Check Java is installed: java -version
  * Set JAVA_HOME environment variable
  * Try: export SPARK_LOCAL_IP=127.0.0.1
  
- If memory errors:
  * Reduce Config.ROWS to 1_000_000
  * Reduce SHUFFLE_PARTITIONS to 50
  * Close other applications

Usage:
  python3 spark_layout_masterclass.py
  
  # Or with spark-submit:
  spark-submit --master local[4] --driver-memory 4g spark_layout_masterclass.py

Author: Educational benchmark for understanding Spark data layouts
"""

import time
import os
import shutil
import json
import sys
from datetime import datetime

# Check Java before importing PySpark
def check_java():
    """Check if Java is available"""
    try:
        import subprocess
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0 or result.stderr:
            java_version = result.stderr.split('\n')[0] if result.stderr else "unknown"
            print(f"✓ Java found: {java_version}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    print("\n" + "="*80)
    print("❌ ERROR: Java not found!")
    print("="*80)
    print("\nPySpark requires Java 8 or 11. Please install:")
    print("\nOn Ubuntu/Debian:")
    print("  sudo apt-get update")
    print("  sudo apt-get install openjdk-11-jdk")
    print("\nOn macOS:")
    print("  brew install openjdk@11")
    print("\nThen set JAVA_HOME:")
    print("  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64")
    print("\nOr find your Java location with:")
    print("  update-alternatives --config java  # Linux")
    print("  /usr/libexec/java_home             # macOS")
    print("\n" + "="*80)
    return False

if not check_java():
    sys.exit(1)

# Set local IP to avoid hostname resolution issues
os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    expr, col, floor, rand, year, month, dayofmonth, 
    to_date, lit, count, avg, sum as spark_sum, min as spark_min, 
    max as spark_max, when
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, DateType, TimestampType
)

# Check for psutil (optional, for memory detection)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("ℹ️  psutil not available - using default memory settings")
    print("   Install with: pip install psutil\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data generation
    ROWS = 5_000_000            # Start smaller, increase after testing
    SEED = 42
    
    # Output paths
    BASE_DIR = "/tmp/spark_masterclass"
    UNPART_DIR = os.path.join(BASE_DIR, "unpartitioned")
    PART_YEAR_MONTH_DIR = os.path.join(BASE_DIR, "partitioned_year_month")
    PART_REGION_DIR = os.path.join(BASE_DIR, "partitioned_region")
    SORTED_DIR = os.path.join(BASE_DIR, "sorted_within_partitions")
    BUCKETED_DB = "masterclass_db"
    BUCKETED_TABLE = "sales_bucketed"
    DELTA_DIR = os.path.join(BASE_DIR, "delta_sales")
    DELTA_ZORDER_DIR = os.path.join(BASE_DIR, "delta_zorder")
    
    # Layout parameters
    NUM_BUCKETS = 8
    TARGET_FILE_SIZE_MB = 128  # Target Parquet file size
    
    # Spark tuning (more conservative for local mode)
    SHUFFLE_PARTITIONS = 100   # Reduced for local execution
    
    # Benchmark control
    WARMUP_RUNS = 0            # Skip warmup for faster iteration
    TIMED_RUNS = 2             # Reduced for faster completion
    
    # Resource control
    ENABLE_DELTA = True        # Set to False to skip Delta tests
    SHOW_EXPLAINS = False      # Set to True for detailed explain plans

config = Config()

# ============================================================================
# HELPER CLASSES
# ============================================================================

class BenchmarkMetrics:
    """Collect and display benchmark metrics"""
    
    def __init__(self):
        self.results = []
        
    def record(self, category, operation, duration, row_count=None, notes=""):
        """Record a benchmark result"""
        self.results.append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "operation": operation,
            "duration_sec": round(duration, 3),
            "row_count": row_count,
            "rows_per_sec": round(row_count / duration, 2) if row_count and duration > 0 else None,
            "notes": notes
        })
        
    def print_summary(self):
        """Print formatted summary of all results"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        for cat, results in categories.items():
            print(f"\n{cat.upper()}")
            print("-" * 80)
            for r in results:
                throughput = f" ({r['rows_per_sec']:,.0f} rows/s)" if r['rows_per_sec'] else ""
                count_str = f" [{r['row_count']:,} rows]" if r['row_count'] else ""
                print(f"  {r['operation']:50s} {r['duration_sec']:8.3f}s{throughput}{count_str}")
                if r['notes']:
                    print(f"    ↳ {r['notes']}")
        
        print("\n" + "="*80)
        
    def save_json(self, filepath):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nMetrics saved to: {filepath}")

class FileAnalyzer:
    """Analyze directory structure and file statistics"""
    
    @staticmethod
    def analyze_directory(path, label):
        """Analyze Parquet directory structure"""
        if not os.path.exists(path):
            print(f"\n[{label}] Path does not exist: {path}")
            return
            
        print(f"\n{'='*60}")
        print(f"FILE ANALYSIS: {label}")
        print(f"{'='*60}")
        print(f"Path: {path}")
        
        total_size = 0
        file_count = 0
        file_sizes = []
        partitions = set()
        
        for root, dirs, files in os.walk(path):
            # Track partition directories
            rel_path = os.path.relpath(root, path)
            if rel_path != '.' and '=' in rel_path:
                partitions.add(rel_path)
            
            for file in files:
                if file.endswith('.parquet'):
                    filepath = os.path.join(root, file)
                    size = os.path.getsize(filepath)
                    total_size += size
                    file_sizes.append(size)
                    file_count += 1
        
        if file_count == 0:
            print("No Parquet files found")
            return
        
        avg_size_mb = (sum(file_sizes) / len(file_sizes)) / (1024**2)
        min_size_mb = min(file_sizes) / (1024**2)
        max_size_mb = max(file_sizes) / (1024**2)
        total_size_mb = total_size / (1024**2)
        
        print(f"Total size: {total_size_mb:,.2f} MB")
        print(f"File count: {file_count:,}")
        print(f"Partitions: {len(partitions) if partitions else 'none'}")
        print(f"Avg file size: {avg_size_mb:.2f} MB")
        print(f"Min file size: {min_size_mb:.2f} MB")
        print(f"Max file size: {max_size_mb:.2f} MB")
        
        # Small files warning
        if avg_size_mb < 64 and file_count > 100:
            print("⚠️  WARNING: Small files problem detected!")
            print("   Consider coalescing or increasing target file size")
        
        if len(partitions) > 1000:
            print("⚠️  WARNING: Too many partitions detected!")
            print("   Consider using fewer partition columns or bucketing")

# ============================================================================
# SPARK SESSION SETUP
# ============================================================================

def create_spark_session():
    """Create optimized Spark session with detailed configs"""
    print("Initializing Spark Session...")
    print(f"Target configuration:")
    print(f"  - Shuffle partitions: {config.SHUFFLE_PARTITIONS}")
    print(f"  - Adaptive Query Execution: enabled")
    print(f"  - Adaptive Coalesce Partitions: enabled")
    
    # Detect available memory
    driver_memory = 4  # Default
    if HAS_PSUTIL:
        try:
            available_gb = psutil.virtual_memory().available / (1024**3)
            driver_memory = min(int(available_gb * 0.5), 8)  # Use max 50% or 8GB
            print(f"  - Driver memory: {driver_memory}g (detected from system)")
        except Exception:
            print(f"  - Driver memory: {driver_memory}g (default)")
    else:
        print(f"  - Driver memory: {driver_memory}g (default)")
    
    spark = (SparkSession.builder
             .appName("SparkLayoutMasterclass")
             # Memory settings
             .config("spark.driver.memory", f"{driver_memory}g")
             .config("spark.executor.memory", f"{driver_memory}g")
             # Adaptive Query Execution
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
             .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
             # Shuffle settings
             .config("spark.sql.shuffle.partitions", str(config.SHUFFLE_PARTITIONS))
             # File size optimization
             .config("spark.sql.files.maxPartitionBytes", f"{config.TARGET_FILE_SIZE_MB}MB")
             # Performance tuning
             .config("spark.sql.parquet.filterPushdown", "true")
             .config("spark.sql.parquet.mergeSchema", "false")
             # Reduce verbosity
             .config("spark.ui.showConsoleProgress", "false")
             # Enable Hive for bucketing
             .enableHiveSupport()
             .getOrCreate())
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_sales_data(spark, num_rows):
    """Generate realistic sales data with multiple dimensions"""
    print(f"\nGenerating {num_rows:,} rows of sales data...")
    print("Schema: customer_id, product_id, sale_date, amount, region, category, channel")
    
    df = (spark.range(0, num_rows)
          .select(
              # Customer dimension (high cardinality)
              (col("id") % 1_000_000).alias("customer_id"),
              
              # Product dimension (medium cardinality)
              (col("id") % 50_000).alias("product_id"),
              
              # Date dimension (1500 days = ~4 years)
              expr("date_add('2020-01-01', cast(id % 1500 as int))").alias("sale_date"),
              
              # Amount (continuous)
              (rand(config.SEED) * 1000.0).alias("amount"),
              
              # Region (low cardinality - good for bucketing)
              (col("id") % 10).cast("int").alias("region_id"),
              
              # Category (low cardinality)
              (col("id") % 5).cast("int").alias("category_id"),
              
              # Channel (very low cardinality)
              (col("id") % 3).cast("int").alias("channel_id")
          ))
    
    # Add human-readable dimensions
    df = (df
          .withColumn("region", 
                     when(col("region_id") == 0, "US")
                     .when(col("region_id") == 1, "BR")
                     .when(col("region_id") == 2, "DE")
                     .when(col("region_id") == 3, "UK")
                     .when(col("region_id") == 4, "FR")
                     .when(col("region_id") == 5, "JP")
                     .when(col("region_id") == 6, "AU")
                     .when(col("region_id") == 7, "CA")
                     .when(col("region_id") == 8, "IN")
                     .otherwise("CN"))
          .withColumn("category",
                     when(col("category_id") == 0, "Electronics")
                     .when(col("category_id") == 1, "Clothing")
                     .when(col("category_id") == 2, "Food")
                     .when(col("category_id") == 3, "Books")
                     .otherwise("Home"))
          .withColumn("channel",
                     when(col("channel_id") == 0, "Online")
                     .when(col("channel_id") == 1, "Store")
                     .otherwise("Mobile"))
          .withColumn("year", year(col("sale_date")))
          .withColumn("month", month(col("sale_date")))
          .withColumn("day", dayofmonth(col("sale_date"))))
    
    return df

# ============================================================================
# LAYOUT STRATEGIES
# ============================================================================

def write_unpartitioned(df, metrics):
    """Strategy 1: Unpartitioned - baseline approach"""
    print("\n" + "="*80)
    print("STRATEGY 1: UNPARTITIONED PARQUET")
    print("="*80)
    print("Description: Simple write with no partitioning or bucketing")
    print("Use case: Small datasets, no common filter patterns")
    print("Pros: Simplest approach, fewer files")
    print("Cons: Must scan all data for every query")
    
    # Coalesce to control file count
    num_files = max(1, config.ROWS // 1_000_000)  # ~1M rows per file
    df_coalesced = df.coalesce(num_files)
    
    t0 = time.time()
    df_coalesced.write.mode("overwrite").parquet(config.UNPART_DIR)
    duration = time.time() - t0
    
    metrics.record("write", "unpartitioned", duration, config.ROWS)
    print(f"✓ Written in {duration:.2f}s")
    FileAnalyzer.analyze_directory(config.UNPART_DIR, "Unpartitioned")

def write_partitioned_year_month(df, metrics):
    """Strategy 2: Partitioned by year/month - time-based partitioning"""
    print("\n" + "="*80)
    print("STRATEGY 2: PARTITIONED BY YEAR/MONTH")
    print("="*80)
    print("Description: Partitioned directory structure by time dimensions")
    print("Use case: Time-series data, queries filtered by date ranges")
    print("Pros: Excellent partition pruning for date filters")
    print("Cons: Can create many small files if data skewed")
    
    t0 = time.time()
    df.write.mode("overwrite").partitionBy("year", "month").parquet(config.PART_YEAR_MONTH_DIR)
    duration = time.time() - t0
    
    metrics.record("write", "partitioned_year_month", duration, config.ROWS)
    print(f"✓ Written in {duration:.2f}s")
    FileAnalyzer.analyze_directory(config.PART_YEAR_MONTH_DIR, "Partitioned (year/month)")

def write_partitioned_region(df, metrics):
    """Strategy 3: Partitioned by region - dimension-based partitioning"""
    print("\n" + "="*80)
    print("STRATEGY 3: PARTITIONED BY REGION")
    print("="*80)
    print("Description: Partitioned by low-cardinality dimension (region)")
    print("Use case: Queries typically filtered by specific dimension")
    print("Pros: Perfect for region-specific queries, balanced partitions")
    print("Cons: No help for non-region queries")
    
    t0 = time.time()
    df.write.mode("overwrite").partitionBy("region").parquet(config.PART_REGION_DIR)
    duration = time.time() - t0
    
    metrics.record("write", "partitioned_region", duration, config.ROWS)
    print(f"✓ Written in {duration:.2f}s")
    FileAnalyzer.analyze_directory(config.PART_REGION_DIR, "Partitioned (region)")

def write_sorted_within_partitions(df, metrics):
    """Strategy 4: Sorted within partitions - optimized for range scans"""
    print("\n" + "="*80)
    print("STRATEGY 4: SORTED WITHIN PARTITIONS")
    print("="*80)
    print("Description: Partitioned + sortWithinPartitions for better statistics")
    print("Use case: Range queries on sorted columns (dates, IDs)")
    print("Pros: Better min/max stats, improved compression, faster range scans")
    print("Cons: Slower writes due to sorting")
    
    t0 = time.time()
    # Repartition by partition columns, then sort
    df_sorted = (df
                 .repartition("year", "month")
                 .sortWithinPartitions("region", "sale_date", "customer_id"))
    df_sorted.write.mode("overwrite").partitionBy("year", "month").parquet(config.SORTED_DIR)
    duration = time.time() - t0
    
    metrics.record("write", "sorted_within_partitions", duration, config.ROWS, 
                  "Sorted by: region, sale_date, customer_id")
    print(f"✓ Written in {duration:.2f}s")
    FileAnalyzer.analyze_directory(config.SORTED_DIR, "Sorted within partitions")

def write_bucketed_table(spark, df, metrics):
    """Strategy 5: Bucketed table - optimized for joins"""
    print("\n" + "="*80)
    print("STRATEGY 5: BUCKETED TABLE")
    print("="*80)
    print("Description: Hash bucketing for sort-merge bucket joins")
    print("Use case: Tables frequently joined on bucketed column")
    print("Pros: Avoids shuffle in joins when both sides bucketed identically")
    print("Cons: Requires Hive metastore, fixed bucket count, slower writes")
    
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {config.BUCKETED_DB}")
    spark.sql(f"USE {config.BUCKETED_DB}")
    
    t0 = time.time()
    (df.repartition(200)  # Pre-shuffle for write parallelism
       .write
       .format("parquet")
       .bucketBy(config.NUM_BUCKETS, "region")
       .sortBy("sale_date", "customer_id")
       .mode("overwrite")
       .saveAsTable(config.BUCKETED_TABLE))
    duration = time.time() - t0
    
    metrics.record("write", "bucketed_table", duration, config.ROWS,
                  f"{config.NUM_BUCKETS} buckets on region, sorted by sale_date")
    print(f"✓ Written in {duration:.2f}s")
    
    # Get table location
    table_loc = spark.sql(f"DESCRIBE EXTENDED {config.BUCKETED_TABLE}").filter("col_name = 'Location'").collect()[0][1]
    FileAnalyzer.analyze_directory(table_loc, "Bucketed table")

def write_delta_tables(spark, df, metrics):
    """Strategy 6 & 7: Delta Lake with and without Z-ordering"""
    if not config.ENABLE_DELTA:
        print("\n⚠️  Delta Lake tests disabled in config - skipping")
        return
        
    try:
        import delta
        print("\n" + "="*80)
        print("STRATEGY 6: DELTA LAKE")
        print("="*80)
        print("Description: Delta Lake format with ACID transactions")
        print("Use case: Need ACID, time travel, schema evolution")
        print("Pros: ACID, DML operations, statistics, optimization")
        print("Cons: Overhead vs plain Parquet")
        
        # Basic Delta write
        t0 = time.time()
        df.write.format("delta").mode("overwrite").partitionBy("year", "month").save(config.DELTA_DIR)
        duration = time.time() - t0
        metrics.record("write", "delta_basic", duration, config.ROWS)
        print(f"✓ Written in {duration:.2f}s")
        FileAnalyzer.analyze_directory(config.DELTA_DIR, "Delta Lake")
        
        # Z-ORDER (multi-dimensional clustering)
        print("\n" + "="*80)
        print("STRATEGY 7: DELTA LAKE WITH Z-ORDER")
        print("="*80)
        print("Description: Delta with Z-ORDER clustering")
        print("Use case: Multiple filter dimensions in queries")
        print("Pros: Excellent for multi-column filters, better than single sort")
        print("Cons: Expensive OPTIMIZE operation")
        
        t0 = time.time()
        df.write.format("delta").mode("overwrite").save(config.DELTA_ZORDER_DIR)
        
        # Z-ORDER optimization
        from delta.tables import DeltaTable
        delta_table = DeltaTable.forPath(spark, config.DELTA_ZORDER_DIR)
        delta_table.optimize().executeZOrderBy("region", "sale_date")
        duration = time.time() - t0
        
        metrics.record("write", "delta_with_zorder", duration, config.ROWS,
                      "Z-ordered by: region, sale_date")
        print(f"✓ Written and optimized in {duration:.2f}s")
        FileAnalyzer.analyze_directory(config.DELTA_ZORDER_DIR, "Delta with Z-ORDER")
        
    except ImportError:
        print("\n⚠️  Delta Lake not available - skipping Delta strategies")
        print("   To enable: pip install delta-spark")
    except Exception as e:
        print(f"\n⚠️  Delta Lake error: {e}")
        print("   Continuing with other strategies...")

# ============================================================================
# BENCHMARK QUERIES
# ============================================================================

def benchmark_query(spark, query_func, query_name, metrics, warmup=True):
    """Execute query multiple times and record metrics"""
    
    try:
        if warmup and config.WARMUP_RUNS > 0:
            for i in range(config.WARMUP_RUNS):
                spark.catalog.clearCache()
                query_func()  # Warmup run
        
        durations = []
        row_counts = []
        
        for i in range(config.TIMED_RUNS):
            spark.catalog.clearCache()
            t0 = time.time()
            result = query_func()
            count = result if isinstance(result, int) else result.count()
            duration = time.time() - t0
            durations.append(duration)
            row_counts.append(count)
        
        avg_duration = sum(durations) / len(durations)
        avg_count = sum(row_counts) / len(row_counts)
        
        metrics.record("query", query_name, avg_duration, int(avg_count))
        
        return avg_duration, int(avg_count)
    
    except Exception as e:
        print(f"⚠️  Query {query_name} failed: {e}")
        metrics.record("query", query_name, 0, 0, f"FAILED: {str(e)[:50]}")
        return 0, 0

def run_benchmarks(spark, metrics):
    """Execute comprehensive benchmark suite"""
    print("\n" + "="*80)
    print("RUNNING BENCHMARK QUERIES")
    print("="*80)
    print(f"Configuration: {config.WARMUP_RUNS} warmup + {config.TIMED_RUNS} timed runs")
    
    try:
        # Load all datasets
        print("\n📂 Loading datasets...")
        unpart = spark.read.parquet(config.UNPART_DIR)
        part_ym = spark.read.parquet(config.PART_YEAR_MONTH_DIR)
        part_region = spark.read.parquet(config.PART_REGION_DIR)
        sorted_df = spark.read.parquet(config.SORTED_DIR)
        bucketed = spark.table(f"{config.BUCKETED_DB}.{config.BUCKETED_TABLE}")
        print("✓ All datasets loaded")
        
        # Query 1: Time-based filter (tests partition pruning)
        print("\n" + "-"*80)
        print("QUERY 1: Time-based filter (month = 6)")
        print("-"*80)
        print("Expected: Partitioned by year/month should be fastest")
        
        if config.SHOW_EXPLAINS:
            print("\n📋 Explain (partitioned):")
            part_ym.filter("month = 6").explain()
        
        def q1_unpart(): return unpart.filter("month = 6").count()
        def q1_part_ym(): return part_ym.filter("month = 6").count()
        def q1_part_region(): return part_region.filter("month = 6").count()
        def q1_sorted(): return sorted_df.filter("month = 6").count()
        def q1_bucketed(): return bucketed.filter("month = 6").count()
        
        benchmark_query(spark, q1_unpart, "Q1_unpartitioned", metrics)
        benchmark_query(spark, q1_part_ym, "Q1_partitioned_year_month", metrics)
        benchmark_query(spark, q1_part_region, "Q1_partitioned_region", metrics)
        benchmark_query(spark, q1_sorted, "Q1_sorted", metrics)
        benchmark_query(spark, q1_bucketed, "Q1_bucketed", metrics)
        
        # Query 2: Dimension filter (tests bucketing/region partitioning)
        print("\n" + "-"*80)
        print("QUERY 2: Dimension filter (region = 'BR')")
        print("-"*80)
        print("Expected: Partitioned by region should be fastest")
        
        def q2_unpart(): return unpart.filter("region = 'BR'").count()
        def q2_part_ym(): return part_ym.filter("region = 'BR'").count()
        def q2_part_region(): return part_region.filter("region = 'BR'").count()
        def q2_sorted(): return sorted_df.filter("region = 'BR'").count()
        def q2_bucketed(): return bucketed.filter("region = 'BR'").count()
        
        benchmark_query(spark, q2_unpart, "Q2_unpartitioned", metrics)
        benchmark_query(spark, q2_part_ym, "Q2_partitioned_year_month", metrics)
        benchmark_query(spark, q2_part_region, "Q2_partitioned_region", metrics)
        benchmark_query(spark, q2_sorted, "Q2_sorted", metrics)
        benchmark_query(spark, q2_bucketed, "Q2_bucketed", metrics)
        
        # Query 3: Range scan (tests sorted data)
        print("\n" + "-"*80)
        print("QUERY 3: Date range scan")
        print("-"*80)
        print("Expected: Sorted data should perform well due to statistics")
        
        def q3_unpart(): 
            return unpart.filter("sale_date BETWEEN DATE('2021-01-01') AND DATE('2021-03-31')").count()
        def q3_part_ym(): 
            return part_ym.filter("sale_date BETWEEN DATE('2021-01-01') AND DATE('2021-03-31')").count()
        def q3_sorted(): 
            return sorted_df.filter("sale_date BETWEEN DATE('2021-01-01') AND DATE('2021-03-31')").count()
        
        benchmark_query(spark, q3_unpart, "Q3_unpartitioned", metrics)
        benchmark_query(spark, q3_part_ym, "Q3_partitioned_year_month", metrics)
        benchmark_query(spark, q3_sorted, "Q3_sorted", metrics)
        
        # Query 4: Multi-dimension filter
        print("\n" + "-"*80)
        print("QUERY 4: Multi-dimension filter")
        print("-"*80)
        
        def q4_unpart():
            return unpart.filter("region = 'US' AND year = 2021 AND category = 'Electronics'").count()
        def q4_sorted():
            return sorted_df.filter("region = 'US' AND year = 2021 AND category = 'Electronics'").count()
        
        benchmark_query(spark, q4_unpart, "Q4_unpartitioned", metrics)
        benchmark_query(spark, q4_sorted, "Q4_sorted", metrics)
        
        # Query 5: Aggregation (tests data locality)
        print("\n" + "-"*80)
        print("QUERY 5: Aggregation by region")
        print("-"*80)
        
        def q5_unpart():
            return unpart.groupBy("region").agg(spark_sum("amount").alias("total")).count()
        def q5_bucketed():
            return bucketed.groupBy("region").agg(spark_sum("amount").alias("total")).count()
        
        benchmark_query(spark, q5_unpart, "Q5_agg_unpartitioned", metrics)
        benchmark_query(spark, q5_bucketed, "Q5_agg_bucketed", metrics)
        
        # Query 6: Join (tests bucketing advantage)
        print("\n" + "-"*80)
        print("QUERY 6: Join on bucketed column")
        print("-"*80)
        print("Expected: Bucketed table should avoid shuffle")
        
        # Create dimension table
        regions_data = [
            ("US", "United States", "North America"),
            ("BR", "Brazil", "South America"),
            ("DE", "Germany", "Europe"),
            ("UK", "United Kingdom", "Europe"),
            ("FR", "France", "Europe"),
            ("JP", "Japan", "Asia"),
            ("AU", "Australia", "Oceania"),
            ("CA", "Canada", "North America"),
            ("IN", "India", "Asia"),
            ("CN", "China", "Asia")
        ]
        regions = spark.createDataFrame(regions_data, ["region", "country_name", "continent"])
        
        # Non-bucketed join (will shuffle)
        def q6_unbucketed():
            return unpart.join(regions, on="region", how="inner").filter("year = 2021").count()
        
        # Bucketed join (no shuffle if dimension also bucketed)
        regions_bucketed = regions.repartition(config.NUM_BUCKETS, "region")
        def q6_bucketed():
            return bucketed.join(regions_bucketed, on="region", how="inner").filter("year = 2021").count()
        
        benchmark_query(spark, q6_unbucketed, "Q6_join_unbucketed", metrics)
        
        if config.SHOW_EXPLAINS:
            print("\n📋 Bucketed join physical plan:")
            bucketed.join(regions_bucketed, on="region", how="inner").filter("year = 2021").explain()
        
        benchmark_query(spark, q6_bucketed, "Q6_join_bucketed", metrics)
        
        print("\n✓ All benchmark queries completed")
        
    except Exception as e:
        print(f"\n⚠️  Benchmark suite encountered an error: {e}")
        print("Continuing with results collected so far...")

# ============================================================================
# EDUCATIONAL INSIGHTS
# ============================================================================

def print_insights():
    """Print key takeaways and best practices"""
    print("\n" + "="*80)
    print("KEY INSIGHTS & BEST PRACTICES")
    print("="*80)
    
    insights = [
        ("Partition Strategy", [
            "Use time-based partitioning (year/month/day) for time-series data",
            "Partition by low-cardinality columns queried frequently",
            "Avoid partitioning by high-cardinality columns (creates too many partitions)",
            "Target: 1GB+ per partition, avoid <100MB partitions",
            "Typical sweet spot: 10-1000 partitions"
        ]),
        ("Bucketing", [
            "Use for tables frequently joined on the same column",
            "Both tables must have same bucket count and column",
            "Excellent for dimension tables and fact-dimension joins",
            "Requires Hive metastore (managed tables)",
            "Bucket count should match or be multiple of parallelism"
        ]),
        ("Sorting", [
            "sortWithinPartitions improves min/max statistics",
            "Helps with: range scans, data skipping, compression",
            "Sort by most selective filter columns first",
            "Consider Z-ordering (Delta) for multi-column filters"
        ]),
        ("File Sizing", [
            "Target: 128MB-1GB per file for balanced parallelism",
            "Small files (<64MB): increase coalesce/repartition",
            "Large files (>1GB): may limit parallelism",
            "Use maxPartitionBytes to control read parallelism"
        ]),
        ("Anti-Patterns", [
            "⚠️ Over-partitioning: >10,000 partitions = metadata overhead",
            "⚠️ Small files: thousands of tiny files = slow queries",
            "⚠️ High-cardinality partitioning: e.g., user_id, timestamp",
            "⚠️ Not coalescing before write: creates too many files",
            "⚠️ Ignoring data skew in bucketing"
        ]),
        ("When to Use What", [
            "Unpartitioned: Small data (<1GB), no filter patterns",
            "Partitioned: Time-series, clear filter dimensions",
            "Bucketed: Frequent joins, star schema",
            "Sorted: Range queries, high selectivity filters",
            "Delta + Z-ORDER: Complex queries, multiple filter dimensions"
        ])
    ]
    
    for title, points in insights:
        print(f"\n📚 {title}")
        print("-" * 60)
        for point in points:
            print(f"  • {point}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    print("="*80)
    print("PYSPARK LAYOUT STRATEGY MASTERCLASS")
    print("="*80)
    print(f"Dataset size: {config.ROWS:,} rows")
    print(f"Output directory: {config.BASE_DIR}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize
    metrics = BenchmarkMetrics()
    
    # Clean previous runs
    if os.path.exists(config.BASE_DIR):
        print(f"\n🧹 Cleaning previous output: {config.BASE_DIR}")
        shutil.rmtree(config.BASE_DIR)
    os.makedirs(config.BASE_DIR, exist_ok=True)
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Generate data
        print("\n" + "="*80)
        print("PHASE 1: DATA GENERATION")
        print("="*80)
        t0 = time.time()
        df = generate_sales_data(spark, config.ROWS)
        
        # Cache for reuse across writes
        df.cache()
        row_count = df.count()
        duration = time.time() - t0
        
        metrics.record("setup", "data_generation", duration, row_count)
        print(f"✓ Generated and cached {row_count:,} rows in {duration:.2f}s")
        
        # Show sample
        print("\n📊 Sample data:")
        df.show(5, truncate=False)
        
        print("\n📊 Data distribution by region:")
        try:
            df.groupBy("region").count().orderBy("region").show(truncate=False)
        except Exception as e:
            print(f"⚠️  Could not compute region distribution: {e}")
        
        print("\n📊 Data distribution by time (showing first 12 months):")
        try:
            df.groupBy("year", "month").count().orderBy("year", "month").show(12, truncate=False)
        except Exception as e:
            print(f"⚠️  Could not compute time distribution: {e}")
        
        # Write all layouts
        print("\n" + "="*80)
        print("PHASE 2: WRITING DATA LAYOUTS")
        print("="*80)
        
        write_unpartitioned(df, metrics)
        write_partitioned_year_month(df, metrics)
        write_partitioned_region(df, metrics)
        write_sorted_within_partitions(df, metrics)
        write_bucketed_table(spark, df, metrics)
        write_delta_tables(spark, df, metrics)
        
        # Unpersist cached data
        df.unpersist()
        
        # Run benchmarks
        print("\n" + "="*80)
        print("PHASE 3: QUERY BENCHMARKS")
        print("="*80)
        run_benchmarks(spark, metrics)
        
        # Display results
        metrics.print_summary()
        
        # Save metrics
        metrics_file = os.path.join(config.BASE_DIR, "benchmark_results.json")
        metrics.save_json(metrics_file)
        
        # Educational insights
        print_insights()
        
        # Final summary
        print("\n" + "="*80)
        print("MASTERCLASS COMPLETE")
        print("="*80)
        print(f"📁 All data written to: {config.BASE_DIR}")
        print(f"📊 Metrics saved to: {metrics_file}")
        print("\n🎓 Next steps:")
        print("  1. Review the benchmark results above")
        print("  2. Experiment with different ROWS sizes")
        print("  3. Check explain() plans for different queries")
        print("  4. Examine Spark UI for shuffle/scan metrics")
        print("  5. Try your own query patterns")
        
        print("\n💡 Pro tips:")
        print("  • Open Spark UI (usually http://localhost:4040)")
        print("  • Look for 'PartitionFilters' in query plans")
        print("  • Monitor 'data size' vs 'scan size' ratios")
        print("  • Check for shuffle in joins (SortMergeJoin vs BucketedJoin)")
        
    finally:
        spark.stop()
        print("\n✓ Spark session stopped")

if __name__ == "__main__":
    main()


# ============================================================================
# ADDITIONAL UTILITIES FOR INTERACTIVE EXPLORATION
# ============================================================================

def interactive_query_analyzer(spark, layout_name, query):
    """
    Helper function for interactive query analysis
    
    Usage in PySpark shell:
        df = spark.read.parquet("/tmp/spark_masterclass/partitioned_year_month")
        interactive_query_analyzer(spark, "partitioned", df.filter("region = 'US'"))
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING QUERY: {layout_name}")
    print(f"{'='*60}")
    
    # Physical plan
    print("\n📋 Physical Plan:")
    query.explain(mode="formatted")
    
    # Execution
    print("\n⏱️  Executing query...")
    t0 = time.time()
    result = query.count()
    duration = time.time() - t0
    
    print(f"✓ Result: {result:,} rows in {duration:.3f}s")
    print(f"  Throughput: {result/duration:,.0f} rows/sec")
    
    return result

def compare_explains(spark, queries_dict):
    """
    Compare explain plans for multiple queries
    
    Usage:
        queries = {
            "unpartitioned": unpart_df.filter("month = 6"),
            "partitioned": part_df.filter("month = 6")
        }
        compare_explains(spark, queries)
    """
    print("\n" + "="*80)
    print("EXPLAIN PLAN COMPARISON")
    print("="*80)
    
    for name, query in queries_dict.items():
        print(f"\n{'─'*60}")
        print(f"Plan: {name}")
        print(f"{'─'*60}")
        query.explain(mode="simple")

def analyze_data_skew(df, column):
    """
    Analyze data distribution/skew for a column
    
    Usage:
        analyze_data_skew(df, "region")
    """
    print(f"\n{'='*60}")
    print(f"DATA SKEW ANALYSIS: {column}")
    print(f"{'='*60}")
    
    distribution = (df.groupBy(column)
                    .count()
                    .orderBy(col("count").desc())
                    .collect())
    
    total = sum([r['count'] for r in distribution])
    
    print(f"\nTotal rows: {total:,}")
    print(f"Distinct values: {len(distribution):,}")
    print(f"\nTop 10 values by frequency:")
    print(f"{'Value':<20} {'Count':>15} {'%':>8}")
    print("-" * 50)
    
    for i, row in enumerate(distribution[:10], 1):
        value = row[column]
        count = row['count']
        pct = 100.0 * count / total
        print(f"{str(value):<20} {count:>15,} {pct:>7.2f}%")
    
    # Skew warning
    if distribution:
        max_pct = 100.0 * distribution[0]['count'] / total
        if max_pct > 50:
            print(f"\n⚠️  WARNING: Severe skew detected!")
            print(f"   Top value represents {max_pct:.1f}% of data")
            print(f"   Consider salting or different partitioning strategy")
        elif max_pct > 25:
            print(f"\n⚠️  Moderate skew detected ({max_pct:.1f}%)")

# Example queries for experimentation
EXAMPLE_QUERIES = """
# ============================================================================
# EXAMPLE QUERIES FOR EXPERIMENTATION
# ============================================================================

After running the masterclass, try these queries to explore different scenarios:

# 1. Partition pruning effectiveness
spark.read.parquet("/tmp/spark_masterclass/partitioned_year_month") \\
    .filter("year = 2021 AND month = 6") \\
    .explain()

# 2. Predicate pushdown (check PartitionFilters vs PushedFilters)
spark.read.parquet("/tmp/spark_masterclass/partitioned_year_month") \\
    .filter("year = 2021 AND amount > 500") \\
    .explain()

# 3. Bucketed join (look for SortMergeJoin without Exchange)
bucketed = spark.table("masterclass_db.sales_bucketed")
regions = spark.createDataFrame([("US",), ("BR",)], ["region"]) \\
    .repartition(8, "region")
bucketed.join(regions, "region").explain()

# 4. Aggregation with bucketing (pre-partitioned for groupBy)
bucketed.groupBy("region").agg(sum("amount")).explain()

# 5. Complex multi-filter query
spark.read.parquet("/tmp/spark_masterclass/sorted_within_partitions") \\
    .filter("region IN ('US', 'BR', 'DE')") \\
    .filter("sale_date BETWEEN '2021-01-01' AND '2021-12-31'") \\
    .filter("category = 'Electronics'") \\
    .filter("amount > 100") \\
    .explain()

# 6. Check file-level statistics
spark.read.parquet("/tmp/spark_masterclass/sorted_within_partitions") \\
    .filter("sale_date > '2023-01-01'") \\
    .explain("cost")

# 7. Analyze shuffle impact
unpart = spark.read.parquet("/tmp/spark_masterclass/unpartitioned")
unpart.repartition("region").write.mode("overwrite").parquet("/tmp/test_shuffle")
# Check Spark UI for shuffle read/write metrics

# 8. Compare scan metrics in Spark UI
# Run both and compare "number of output rows" and "scan time"
spark.read.parquet("/tmp/spark_masterclass/unpartitioned") \\
    .filter("month = 6").count()
spark.read.parquet("/tmp/spark_masterclass/partitioned_year_month") \\
    .filter("month = 6").count()

"""

def print_example_queries():
    """Print example queries for users to experiment with"""
    print(EXAMPLE_QUERIES)