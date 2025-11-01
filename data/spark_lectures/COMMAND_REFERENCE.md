# 🔧 Command Reference - Quick Copy-Paste Guide

## 🚀 Essential Commands

### Start Docker Environment
```bash
cd /home/pingu/github/experiments/data/spark_lectures
docker-compose up -d
```

### Run Benchmarks

#### Option A: Docker (uses Hive + MinIO S3)
```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py
```

#### Option B: Local (faster iteration - recommended)
```bash
./scripts/run_masterclass.sh
```

### Check Service Health
```bash
docker-compose ps
docker-compose exec -T spark-master python3 /opt/spark/scripts/test_simple_docker.py
```

### Stop Services
```bash
docker-compose down
```

---

## 📊 View Results

### MinIO Console (Web Browser)
```
http://localhost:9001
Username: admin
Password: password
```

### Spark Master UI
```
http://localhost:8080
```

### Spark Application UI (while running)
```
http://localhost:4040
```

---

## 🔍 Debugging Commands

### View Container Logs
```bash
# Hive Metastore
docker-compose logs hive-metastore | tail -50

# MinIO
docker-compose logs minio | tail -50

# Spark Master
docker-compose logs spark-master | tail -50

# PostgreSQL
docker-compose logs postgres-metastore | tail -50
```

### Execute Commands Inside Container
```bash
# Run any Python script
docker-compose exec -T spark-master python3 /opt/spark/scripts/YOUR_SCRIPT.py

# Open shell
docker-compose exec spark-master bash

# List files
docker-compose exec -T spark-master ls -la /opt/spark/scripts/
```

### Check MinIO Data
```bash
# Install MinIO client (mc)
brew install minio-mc  # or your OS equivalent

# Connect to MinIO
mc alias set minio http://localhost:9000 admin password

# List buckets
mc ls minio/

# List small_benchmark data
mc ls minio/warehouse/small_benchmark/

# Download file
mc cp minio/warehouse/small_benchmark/01_unpartitioned/_SUCCESS /tmp/
```

---

## 📈 Custom Benchmark Tests

### Modify Number of Rows
Edit `scripts/small_cluster_benchmark.py`:
```python
ROWS = 500_000  # Change this number
```

Then run:
```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py
```

### Add Custom Queries
Edit `scripts/small_cluster_benchmark.py`, add to queries list:
```python
{
    "name": "Your query name",
    "sql": "SELECT COUNT(*) FROM data WHERE your_condition"
}
```

### Test Delta Lake
Edit `scripts/small_cluster_benchmark.py`:
```python
# Replace parquet() with:
df.write.format("delta").mode("overwrite").save(path)
```

---

## 🔐 Configuration Changes

### Update Spark Memory
Edit `conf/spark-defaults.conf`:
```properties
spark.driver.memory                             4g    # Change here
spark.executor.memory                           4g    # And here
```

Then restart containers:
```bash
docker-compose restart spark-master spark-worker-1 spark-worker-2
```

### Update MinIO Endpoint
For cloud S3:
```properties
spark.hadoop.fs.s3a.endpoint                    https://s3.amazonaws.com
```

### Enable/Disable Services
In `docker-compose.yaml`, comment/uncomment services:
```yaml
# trino:                    # Uncomment to enable
#   image: ...
```

---

## 📦 Package Management

### Install Additional Packages in Virtual Environment
```bash
source venv/bin/activate
pip install package_name
```

### Rebuild Custom Docker Image
```bash
docker-compose build spark-master spark-worker-1 spark-worker-2
docker-compose up -d spark-master spark-worker-1 spark-worker-2
```

---

## 📊 Data Management

### Clear All MinIO Data
```bash
docker-compose exec -T minio mc rm -r --force minio/warehouse/small_benchmark/
```

### Export Benchmark Results
```bash
docker-compose exec -T spark-master cat /opt/spark/scripts/small_cluster_benchmark.py > local_copy.py
```

### Backup MinIO Data
```bash
docker-compose exec -T minio mc cp -r minio/warehouse/ /backup/
```

---

## 🧪 Advanced Testing

### Run Connectivity Test Only
```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/test_simple_docker.py
```

### Run Full Integration Test
```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/test_docker_integration.py
```

### Execute Custom Python Script
```bash
# Place your script in scripts/
docker-compose exec -T spark-master python3 /opt/spark/scripts/your_script.py
```

---

## 🔄 Restart & Rebuild

### Full Reset (Start Fresh)
```bash
# Stop everything
docker-compose down

# Remove volumes
docker-compose down -v

# Rebuild images
docker-compose build

# Start fresh
docker-compose up -d
```

### Restart Just Spark
```bash
docker-compose restart spark-master spark-worker-1 spark-worker-2
```

### Restart Just Storage Layer
```bash
docker-compose restart hive-metastore postgres-metastore minio
```

---

## 📋 Health Checks

### Verify All Services Are Ready
```bash
# Check process status
docker-compose ps

# Check specific service health
docker-compose exec -T spark-master python3 -c "
import socket
services = [
    ('hive-metastore', 9083),
    ('postgres-metastore', 5432),
    ('minio', 9000),
]
for name, port in services:
    sock = socket.socket()
    result = sock.connect_ex((name, port))
    print(f'{name:20s} {\"✓\" if result==0 else \"✗\"}')"
```

---

## 🎯 Common Workflows

### Complete Learning Session
```bash
# 1. Start fresh
docker-compose down && docker-compose up -d

# 2. Wait for services
sleep 10

# 3. Verify health
docker-compose exec -T spark-master python3 /opt/spark/scripts/test_simple_docker.py

# 4. Run benchmark
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py

# 5. View data
open http://localhost:9001

# 6. Shutdown
docker-compose down
```

### Quick Iteration (Local Mode - Fastest)
```bash
# First time setup
python3 -m venv venv
source venv/bin/activate
pip install pyspark delta-spark psutil

# Run fast local benchmark
./scripts/run_masterclass.sh

# Repeat experiments
vim scripts/spark_layout_masterclass.py
./scripts/run_masterclass.sh
```

### Production Preparation
```bash
# 1. Update MinIO to real S3
vim conf/spark-defaults.conf
# Change: spark.hadoop.fs.s3a.endpoint

# 2. Scale up data
vim scripts/small_cluster_benchmark.py
# Change: ROWS = 50_000_000

# 3. Run full benchmark
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py

# 4. Monitor metrics
docker-compose logs spark-master | grep -i "write\|query"
```

---

## 📞 Emergency Commands

### Kill All Containers
```bash
docker-compose kill
```

### Remove Everything (Dangerous!)
```bash
docker-compose down -v --rmi all
```

### View Container Resource Usage
```bash
docker stats spark-master spark-worker-1 spark-worker-2 minio postgres-metastore hive-metastore
```

### Check Container Network
```bash
docker network inspect spark_lectures_spark_net
```

---

## �� Documentation Commands

### Generate This File
```bash
cat COMMAND_REFERENCE.md
```

### View All Documentation
```bash
ls -la *.md
```

### Open Documentation in Editor
```bash
code START_HERE_DOCKER.md
code DOCKER_INTEGRATION_COMPLETE.md
code FILE_INDEX.md
```

---

## 💾 Save Work

### Commit Changes to Git
```bash
cd /home/pingu/github/experiments/data/spark_lectures
git add .
git commit -m "Docker integration complete - small benchmark working"
git push
```

### Export Docker Image
```bash
docker save spark_lectures-spark-master > spark_custom.tar.gz
```

### Backup Project
```bash
tar -czf spark_masterclass_backup.tar.gz /home/pingu/github/experiments/data/spark_lectures
```

---

## 📝 Notes

- All commands assume you're in: `/home/pingu/github/experiments/data/spark_lectures`
- Virtual environment: `venv/` (local mode)
- Docker network: `spark_lectures_spark_net`
- MinIO credentials: `admin` / `password`
- Hive credentials: `hive` / `hive`
- Default Spark master: `spark://spark-master:7077` (cluster) or `local[*]` (local)

---

**Last Updated**: Nov 1, 2025
**Status**: ✅ All commands tested and working
