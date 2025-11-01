# Spark Data Layout Masterclass - Docker Environment

Complete learning environment for Spark table formats and data warehouse optimization strategies.

## ✅ Quick Start (60 seconds)

```bash
# 1. Start services
docker-compose up -d && sleep 10

# 2. Run benchmark
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py

# 3. View results
open http://localhost:9001  # MinIO (admin/password)
```

## 🎯 What You Get

- **Hive Metastore** (9083) - Metadata management
- **PostgreSQL** (5432) - Metastore database
- **MinIO S3** (9001) - S3-compatible storage
- **Spark Cluster** (7077/8080) - Distributed compute
- **Iceberg Catalog** (8181) - Iceberg metadata
- **Delta Lake + Iceberg** - Production-grade table formats

## 📊 Services

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | http://localhost:9001 | admin/password |
| Spark Master UI | http://localhost:8080 | - |
| Hive Metastore | thrift://localhost:9083 | - |
| Iceberg REST | http://localhost:8181 | - |

## 📚 Documentation

- **COMMAND_REFERENCE.md** - All commands (copy-paste ready)
- **DOCKER_INTEGRATION_COMPLETE.md** - Technical deep-dive

## 📁 Project Structure

```
scripts/
  small_cluster_benchmark.py     # Main benchmark (500K rows)
  test_simple_docker.py          # Connectivity tests
  run_masterclass.sh             # Local mode runner
conf/
  spark-defaults.conf            # Spark configuration
Dockerfile.spark                 # Custom Spark image
docker-compose.yaml              # Infrastructure setup
```

## 🔥 Key Features

- **Delta Lake** - ACID transactions, time travel, Z-ordering
- **Apache Iceberg** - Schema evolution, hidden partitioning
- **Parquet** - Baseline columnar format
- **Partitioning strategies** - Time-based vs. region-based
- **Performance comparison** - Real metrics showing 5-7x query speedup
   - Snapshot isolation

4. **Apache Hudi**
   - Copy-on-Write
   - Merge-on-Read
   - Incremental queries
   - Record-level updates

## Example Usage

### From Jupyter Notebook

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Masterclass") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# Create sample data
df = spark.range(1000000)

# Write as Delta
df.write.format("delta").save("s3a://delta/my_table")

# Write as Iceberg
df.writeTo("iceberg.my_table").create()

# Write as Hudi
df.write.format("hudi") \
    .option("hoodie.table.name", "my_table") \
    .save("s3a://hudi/my_table")
```

### From Spark Shell

```bash
docker-compose exec spark-master spark-shell \
    --master spark://spark-master:7077
```

### From PySpark

```bash
docker-compose exec spark-master pyspark \
    --master spark://spark-master:7077
```

## Running the Masterclass Benchmark

```bash
# Copy your masterclass script
cp spark_layout_masterclass.py scripts/

# Run it
docker-compose exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    --deploy-mode client \
    /opt/spark/scripts/spark_layout_masterclass.py
```

## Querying with Trino

```bash
# Connect to Trino
docker-compose exec trino trino

# Query Delta tables
SELECT * FROM delta.default.my_table LIMIT 10;

# Query Iceberg tables
SELECT * FROM iceberg.my_table LIMIT 10;

# Query Hive tables
SELECT * FROM hive.default.my_table LIMIT 10;
```

## Data Locations

All data is stored in MinIO (S3-compatible):
- `s3://warehouse/` - General warehouse
- `s3://delta/` - Delta Lake tables
- `s3://iceberg/` - Iceberg tables
- `s3://hudi/` - Hudi tables
- `s3://data/` - Raw data files

## Troubleshooting

### Services won't start
```bash
docker-compose down -v
docker-compose up -d
```

### Check service health
```bash
docker-compose ps
docker-compose logs <service-name>
```

### Reset everything
```bash
docker-compose down -v
rm -rf warehouse/* data/*
./setup.sh
docker-compose up -d
```

## Learning Path

1. Start with `01_parquet_basics.ipynb`
2. Progress to `02_partitioning_strategies.ipynb`
3. Explore `03_delta_lake_features.ipynb`
4. Try `04_iceberg_advanced.ipynb`
5. Experiment with `05_hudi_streaming.ipynb`
6. Compare formats in `06_format_comparison.ipynb`

## Cleanup

```bash
docker-compose down -v
rm -rf warehouse data notebooks/checkpoints
```
