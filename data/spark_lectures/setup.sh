#!/bin/bash

# Comprehensive Spark Data Layout Masterclass - Setup Script
# This script creates the necessary directory structure and files

set -e

echo "========================================="
echo "Spark Data Layout Masterclass Setup"
echo "========================================="

# Create directory structure
echo "Creating directory structure..."
mkdir -p conf/trino/catalog
mkdir -p warehouse/{delta,iceberg,hudi,parquet}
mkdir -p notebooks
mkdir -p data
mkdir -p scripts

echo "✓ Directories created"

# Create Trino catalog configurations
echo "Creating Trino catalog configurations..."

# Delta Lake catalog
cat > conf/trino/catalog/delta.properties << 'EOF'
connector.name=delta_lake
hive.metastore.uri=thrift://hive-metastore:9083
hive.s3.endpoint=http://minio:9000
hive.s3.path-style-access=true
hive.s3.aws-access-key=admin
hive.s3.aws-secret-key=password
delta.enable-non-concurrent-writes=true
EOF

# Iceberg catalog
cat > conf/trino/catalog/iceberg.properties << 'EOF'
connector.name=iceberg
iceberg.catalog.type=rest
iceberg.rest-catalog.uri=http://iceberg-rest:8181
hive.s3.endpoint=http://minio:9000
hive.s3.path-style-access=true
hive.s3.aws-access-key=admin
hive.s3.aws-secret-key=password
EOF

# Hive catalog
cat > conf/trino/catalog/hive.properties << 'EOF'
connector.name=hive
hive.metastore.uri=thrift://hive-metastore:9083
hive.s3.endpoint=http://minio:9000
hive.s3.path-style-access=true
hive.s3.aws-access-key=admin
hive.s3.aws-secret-key=password
hive.non-managed-table-writes-enabled=true
hive.allow-drop-table=true
EOF

echo "✓ Trino catalogs configured"

# Create README
cat > README.md << 'EOF'
# Spark Data Layout Masterclass - Docker Environment

Complete environment for learning Spark table formats and optimization strategies.

## Quick Start

```bash
# 1. Setup directories and configs
./setup.sh

# 2. Start all services
docker-compose up -d

# 3. Wait for services to be ready (2-3 minutes)
docker-compose logs -f

# 4. Access Jupyter Lab
open http://localhost:8888  # Token: admin
```

## Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Jupyter Lab | http://localhost:8888 | Token: `admin` |
| Spark Master UI | http://localhost:8080 | - |
| Spark Worker 1 | http://localhost:8081 | - |
| Spark Worker 2 | http://localhost:8082 | - |
| MinIO Console | http://localhost:9001 | admin/password |
| Trino UI | http://localhost:8085 | - |
| Hive Metastore | thrift://localhost:9083 | - |

## Table Formats Included

1. **Parquet** (baseline)
   - Unpartitioned
   - Partitioned
   - Bucketed
   - Sorted

2. **Delta Lake**
   - ACID transactions
   - Time travel
   - Z-ordering
   - OPTIMIZE/VACUUM

3. **Apache Iceberg**
   - Hidden partitioning
   - Schema evolution
   - Time travel
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
EOF

echo "✓ README created"

# Create sample notebooks
echo "Creating sample notebooks..."

# Notebook 1: Parquet Basics
cat > notebooks/01_parquet_basics.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Parquet Basics and Layout Strategies\n",
    "\n",
    "This notebook demonstrates fundamental Parquet optimization techniques."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.sql import SparkSession\n",
    "from pyspark.sql.functions import *\n",
    "\n",
    "spark = SparkSession.builder \\\n",
    "    .appName(\"Parquet Basics\") \\\n",
    "    .master(\"spark://spark-master:7077\") \\\n",
    "    .getOrCreate()\n",
    "\n",
    "print(f\"Spark {spark.version} ready!\")\n",
    "print(f\"Master: {spark.sparkContext.master}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate sample sales data\n",
    "df = spark.range(0, 1_000_000) \\\n",
    "    .select(\n",
    "        (col(\"id\") % 100000).alias(\"customer_id\"),\n",
    "        (col(\"id\") % 10000).alias(\"product_id\"),\n",
    "        expr(\"date_add('2024-01-01', cast(id % 365 as int))\").alias(\"sale_date\"),\n",
    "        (rand() * 1000).alias(\"amount\"),\n",
    "        (col(\"id\") % 5).cast(\"int\").alias(\"region_id\")\n",
    "    ) \\\n",
    "    .withColumn(\"region\", \n",
    "        when(col(\"region_id\") == 0, \"US\")\n",
    "        .when(col(\"region_id\") == 1, \"EU\")\n",
    "        .when(col(\"region_id\") == 2, \"ASIA\")\n",
    "        .when(col(\"region_id\") == 3, \"LATAM\")\n",
    "        .otherwise(\"OTHER\")\n",
    "    ) \\\n",
    "    .withColumn(\"year\", year(\"sale_date\")) \\\n",
    "    .withColumn(\"month\", month(\"sale_date\"))\n",
    "\n",
    "df.show(5)\n",
    "print(f\"Generated {df.count():,} rows\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Write unpartitioned Parquet\n",
    "df.write.mode(\"overwrite\").parquet(\"s3a://warehouse/sales_unpartitioned\")\n",
    "print(\"✓ Unpartitioned written\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Write partitioned by year/month\n",
    "df.write.mode(\"overwrite\") \\\n",
    "    .partitionBy(\"year\", \"month\") \\\n",
    "    .parquet(\"s3a://warehouse/sales_partitioned\")\n",
    "print(\"✓ Partitioned written\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare query performance\n",
    "import time\n",
    "\n",
    "# Unpartitioned query\n",
    "unpart = spark.read.parquet(\"s3a://warehouse/sales_unpartitioned\")\n",
    "t0 = time.time()\n",
    "count_unpart = unpart.filter(\"month = 6\").count()\n",
    "t_unpart = time.time() - t0\n",
    "\n",
    "# Partitioned query\n",
    "part = spark.read.parquet(\"s3a://warehouse/sales_partitioned\")\n",
    "t0 = time.time()\n",
    "count_part = part.filter(\"month = 6\").count()\n",
    "t_part = time.time() - t0\n",
    "\n",
    "print(f\"Unpartitioned: {count_unpart:,} rows in {t_unpart:.3f}s\")\n",
    "print(f\"Partitioned:   {count_part:,} rows in {t_part:.3f}s\")\n",
    "print(f\"Speedup: {t_unpart/t_part:.2f}x\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Show physical plans\n",
    "print(\"\\nUnpartitioned plan:\")\n",
    "unpart.filter(\"month = 6\").explain()\n",
    "\n",
    "print(\"\\nPartitioned plan:\")\n",
    "part.filter(\"month = 6\").explain()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "✓ Sample notebooks created"

# Create quick start script
cat > scripts/run_masterclass.sh << 'EOF'
#!/bin/bash
# Run the masterclass benchmark in the cluster

docker-compose exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    --deploy-mode client \
    --driver-memory 4g \
    --executor-memory 4g \
    --conf spark.sql.adaptive.enabled=true \
    /opt/spark/scripts/spark_layout_masterclass.py "$@"
EOF

chmod +x scripts/run_masterclass.sh

echo "✓ Run script created"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. docker-compose up -d"
echo "2. Wait 2-3 minutes for services to start"
echo "3. Open http://localhost:8888 (token: admin)"
echo "4. Check http://localhost:9001 for MinIO console"
echo ""
echo "See README.md for detailed instructions"
echo "========================================="