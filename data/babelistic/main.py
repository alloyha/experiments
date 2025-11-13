"""
spark_layout_benchmark.py

PySpark benchmark: compare unpartitioned, partitioned, sortWithinPartitions, and bucketed layouts.
Adjust ROWS and PARALLELISM for your environment.

Usage:
  spark-submit --master local[8] spark_layout_benchmark.py
or run from a PySpark shell.
"""

import time
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, floor, rand, year, month, to_date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# CONFIG
ROWS = 5_000_000           # change to e.g. 50_000_000 for heavier IO
OUT_DIR = "/tmp/spark_bench_sales"
PART_DIR = os.path.join(OUT_DIR, "parquet_partitioned")
UNPART_DIR = os.path.join(OUT_DIR, "parquet_unpartitioned")
SORTED_DIR = os.path.join(OUT_DIR, "parquet_sorted")
BUCKETED_DB = "bench_db"
BUCKETED_TABLE = "bucketed_sales"
BUCKETS = 8
SEED = 42

# Create/cleanup
if os.path.exists(OUT_DIR):
    print("Cleaning output dir:", OUT_DIR)
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

spark = (SparkSession.builder
         .appName("LayoutBenchmark")
         .config("spark.sql.shuffle.partitions", "200")   # tune for your cluster
         .config("spark.sql.adaptive.enabled", "true")
         .enableHiveSupport()                            # needed for bucketBy/saveAsTable
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

# Generate synthetic dataset
print("Generating dataset with", ROWS, "rows (this is lazy until we action).")
df = (spark.range(0, ROWS)
      .select(
          (col("id") % 1000000).alias("customer_id"),
          (col("id") % 10000).alias("product_id"),
          ( (expr("date_add('2020-01-01', cast(id % 1500 as int))")) ).alias("sale_date"),
          ( (rand(SEED) * 1000.0) ).alias("amount"),
          ( (col("id") % 10).cast("int") ).alias("region_id"),
      ))

# add human-friendly columns
df = df.withColumn("region", expr("CASE WHEN region_id = 0 THEN 'US' WHEN region_id = 1 THEN 'BR' WHEN region_id = 2 THEN 'DE' ELSE 'AP' END")) \
       .withColumn("year", year(col("sale_date"))) \
       .withColumn("month", month(col("sale_date")))

# Materialize a sample count to force generation
print("Materializing base DF...")
_ = df.count()

# 1) Unpartitioned Parquet
print("\n--- Writing unpartitioned parquet ---")
t0 = time.time()
df.write.mode("overwrite").parquet(UNPART_DIR)
t1 = time.time()
print("Wrote unpartitioned parquet in %.2fs" % (t1 - t0))

# 2) Partitioned Parquet by year/month
print("\n--- Writing partitioned parquet (year/month) ---")
t0 = time.time()
df.write.mode("overwrite").partitionBy("year", "month").parquet(PART_DIR)
t1 = time.time()
print("Wrote partitioned parquet in %.2fs" % (t1 - t0))

# 3) Parquet sorted within files (sortWithinPartitions) and partitioned
print("\n--- Writing partitioned + sortWithinPartitions parquet (region, sale_date) ---")
t0 = time.time()
df_sorted = df.repartition("year", "month").sortWithinPartitions("region", "sale_date")
df_sorted.write.mode("overwrite").partitionBy("year", "month").parquet(SORTED_DIR)
t1 = time.time()
print("Wrote partitioned+sorted parquet in %.2fs" % (t1 - t0))

# 4) Bucketed table (requires Hive metastore); writing as managed table
print("\n--- Creating bucketed table (bucketBy + sortBy) ---")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {BUCKETED_DB}")
spark.sql(f"USE {BUCKETED_DB}")
# Convert df to a temp view and then saveAsTable with bucketBy
(df.repartition(200)   # control parallelism
   .write
   .format("parquet")
   .bucketBy(BUCKETS, "region")
   .sortBy("sale_date")
   .mode("overwrite")
   .saveAsTable(BUCKETED_TABLE)
)
print("Bucketed table created in DB:", BUCKETED_DB, "table:", BUCKETED_TABLE)

# Optional Delta: Check availability and write a Delta table + attempt Z-order (if delta available)
HAS_DELTA = False
try:
    import delta
    HAS_DELTA = True
except Exception:
    print("Delta not available in this Spark environment; skipping delta section (fine).")

if HAS_DELTA:
    delta_dir = os.path.join(OUT_DIR, "delta_sales")
    df.write.format("delta").mode("overwrite").save(delta_dir)
    # Note: z-order requires Delta optimize support (databricks or proper delta optimize jar)
    print("Delta table written (skipping optimize if not available).")

# Read / Query definitions
def timed_count(read_df, label):
    spark.catalog.clearCache()
    start = time.time()
    c = read_df.count()   # action that forces read
    end = time.time()
    print("%-40s -> count=%d time=%.3fs" % (label, c, end-start))

print("\n=== BENCHMARK QUERIES ===")

# Query 1: filter by month = 2 (should hit partition pruning for partitioned datasets)
print("\n--- Query: month = 2 ---")
unpart = spark.read.parquet(UNPART_DIR)
part = spark.read.parquet(PART_DIR)
sortedp = spark.read.parquet(SORTED_DIR)
bucketed = spark.table(f"{BUCKETED_DB}.{BUCKETED_TABLE}")

print("\nExplain (unpartitioned):")
unpart.filter("month = 2").explain(True)
print("\nExplain (partitioned):")
part.filter("month = 2").explain(True)
print("\nExplain (sorted partitioned):")
sortedp.filter("month = 2").explain(True)
print("\nExplain (bucketed table):")
bucketed.filter("month = 2").explain(True)

timed_count(unpart.filter("month = 2"), "unpartitioned.filter(month=2)")
timed_count(part.filter("month = 2"), "partitioned.filter(month=2)")
timed_count(sortedp.filter("month = 2"), "sorted_partitioned.filter(month=2)")
timed_count(bucketed.filter("month = 2"), "bucketed.filter(month=2)")

# Query 2: filter by region = 'BR' AND sale_date range (range scan)
print("\n--- Query: region = 'BR' AND sale_date BETWEEN ... ---")
q_unpart = unpart.filter("region = 'BR' AND sale_date BETWEEN DATE('2020-02-01') AND DATE('2020-05-01')")
q_part = part.filter("region = 'BR' AND sale_date BETWEEN DATE('2020-02-01') AND DATE('2020-05-01')")
q_sorted = sortedp.filter("region = 'BR' AND sale_date BETWEEN DATE('2020-02-01') AND DATE('2020-05-01')")
q_bucketed = bucketed.filter("region = 'BR' AND sale_date BETWEEN DATE('2020-02-01') AND DATE('2020-05-01')")

print("\nExplain (unpartitioned range):")
q_unpart.explain(True)
print("\nExplain (partitioned range):")
q_part.explain(True)
print("\nExplain (sorted partitioned range):")
q_sorted.explain(True)
print("\nExplain (bucketed range):")
q_bucketed.explain(True)

timed_count(q_unpart, "unpartitioned.range")
timed_count(q_part, "partitioned.range")
timed_count(q_sorted, "sorted_partitioned.range")
timed_count(q_bucketed, "bucketed.range")

# Query 3: join on region (to show bucketing join benefit when both sides bucketed)
print("\n--- Query: join with small dimension on region (simulate) ---")
dim = spark.createDataFrame([("US", "United States"), ("BR", "Brazil"), ("DE", "Germany"), ("AP", "AsiaPac")], ["region", "region_name"])
dim = dim.repartition(BUCKETS, "region")  # simulate the dimension being bucketed similarly

big = spark.table(f"{BUCKETED_DB}.{BUCKETED_TABLE}")
join_df = big.join(dim, on="region", how="inner").select("region", "region_name", "sale_date").where("month=2")

print("\nExplain join (bucketed hint):")
join_df.explain(True)

t0 = time.time()
cnt = join_df.count()
t1 = time.time()
print("Join count=%d time=%.3fs" % (cnt, t1-t0))

# Clean up or leave data for inspection
print("\nBenchmark complete. Files at:", OUT_DIR)
spark.stop()
