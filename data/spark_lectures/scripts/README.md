# 📜 Scripts - Spark Data Layout Masterclass

Clean, focused benchmark and test scripts for learning Spark table formats.

## 🚀 Quick Start

Choose based on your needs:

### Option 1: Quick Learning (⭐ Recommended)
**Fast iteration with instant results**

```bash
# Inside Docker container (5 minutes, 500K rows)
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py
```

### Option 2: Full Comprehensive Test
**Complete benchmark with all features (10-15 minutes, 5M rows)**

```bash
# Local mode on your machine (no Docker needed)
./scripts/run_masterclass.sh
```

### Option 3: Just Check if Services Work
**Quick connectivity verification (30 seconds)**

```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/test_simple_docker.py
```

---

## 📋 Scripts Reference

### `small_cluster_benchmark.py` (500K rows)
**Purpose**: Fast learning and iteration  
**Best for**: Quick experiments, understanding partition strategies  
**Location**: `/opt/spark/scripts/` in Docker  
**Run time**: ~5 minutes

```bash
# Inside Docker
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py

# Or from host (copy to local venv first)
python3 small_cluster_benchmark.py
```

**What it does:**
- Generates 500,000 sales records
- Tests 3 partition strategies (unpartitioned, by region, by time)
- Runs 9 queries
- Shows performance comparisons (5-7x differences demonstrated)
- Writes results to MinIO S3

**Output example:**
```
[WRITE] Testing 3 storage strategies...
  Unpartitioned:     50.3s (9,942 rows/s) ✓
  Partitioned Region: 78.5s (6,367 rows/s)
  Partitioned Time:  109.2s (4,580 rows/s)

[QUERY] Running 9 queries...
  Region filter: 0.97s (fastest with region partition)
  Time filter:   1.15s (fastest with time partition)
```

---

### `spark_layout_masterclass.py` (5M rows)
**Purpose**: Comprehensive learning benchmark  
**Best for**: Complete understanding, all table formats, production-like testing  
**Location**: `scripts/` (local), or `/opt/spark/scripts/` in Docker  
**Run time**: ~15 minutes

```bash
# Option A: Local mode (recommended - faster setup)
./scripts/run_masterclass.sh

# Option B: From venv directly
source venv/bin/activate
python scripts/spark_layout_masterclass.py
```

**What it does:**
- Generates 5,000,000 sales records
- Tests **4 table formats**: Parquet, Delta Lake, Iceberg, Hudi
- Tests **4 storage strategies** per format
- Runs comprehensive queries
- Measures write performance, query performance, storage size
- Generates detailed performance reports

**Output example:**
```
Format: Parquet
  Unpartitioned:  Write 150s  |  Query 8.2s
  Partitioned:    Write 320s  |  Query 1.2s
  Bucketed:       Write 280s  |  Query 0.9s
  Z-ordered:      Write 380s  |  Query 0.8s

Format: Delta Lake
  Unpartitioned:  Write 160s  |  Query 8.5s
  Partitioned:    Write 330s  |  Query 1.3s
  With OPTIMIZE:  Write 400s  |  Query 0.7s
```

---

### `test_simple_docker.py` (health check)
**Purpose**: Verify Docker services are accessible  
**Best for**: Troubleshooting connectivity issues  
**Run time**: ~5 seconds

```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/test_simple_docker.py
```

**What it checks:**
- ✓ Hive Metastore (9083)
- ✓ PostgreSQL (5432)
- ✓ MinIO S3 (9000)
- ✓ Spark Master (7077)
- ✓ Iceberg REST Catalog (8181)

**Example output:**
```
SERVICE CONNECTIVITY RESULTS
═════════════════════════════════════════════════════════

Hive Metastore                ✓ Available
PostgreSQL                    ✓ Available
MinIO S3 API                  ✓ Available
Spark Master                  ✗ Unreachable  (non-critical)
Iceberg REST                  ✓ Available

Success: 4/5 services reachable
```

---

### `run_masterclass.sh` (shell wrapper)
**Purpose**: Convenient launcher for local mode benchmark  
**Best for**: One-command full benchmark execution  
**Run time**: Depends on which script it launches

```bash
./scripts/run_masterclass.sh
```

**What it does:**
1. Finds Python virtual environment
2. Activates it
3. Runs `spark_layout_masterclass.py`
4. Shows progress and results

---

## 📊 Decision Tree: Which Script to Run?

```
Do you have Docker running?
  ├─ YES
  │  └─ Do you want quick results (5 min)?
  │     ├─ YES → small_cluster_benchmark.py (recommended)
  │     └─ NO  → spark_layout_masterclass.sh (comprehensive)
  │
  └─ NO
     └─ Do you have venv with pyspark installed?
        ├─ YES → ./scripts/run_masterclass.sh
        └─ NO  → Set up venv first (see README.md)

Want to check if services work?
  └─ → test_simple_docker.py
```

---

## 🔧 Common Tasks

### Run just the health check
```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/test_simple_docker.py
```

### Run quick benchmark with custom row count
Edit `small_cluster_benchmark.py`:
```python
ROWS = 1_000_000  # Change this
```
Then run:
```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py
```

### Run benchmark and save results
```bash
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py > results.txt 2>&1
```

### Run inside container shell
```bash
docker-compose exec spark-master bash
cd /opt/spark/scripts
python3 small_cluster_benchmark.py
```

---

## 📝 Script Comparison

| Feature | Quick Bench | Full Bench | Health Check |
|---------|------------|-----------|--------------|
| **Rows** | 500K | 5M | N/A |
| **Time** | 5 min | 15 min | 5 sec |
| **Formats** | Parquet, Delta, Iceberg | Parquet, Delta, Iceberg, Hudi | N/A |
| **Requires Docker** | Yes | No (local venv) | Yes |
| **Queries** | 9 | 20+ | 0 |
| **Real data writes** | Yes (to MinIO) | Yes (to /tmp) | No |
| **Best for** | Learning | Comprehensive test | Verification |

---

## ✅ Prerequisites

### For Docker scripts (`small_cluster_benchmark.py`, `test_simple_docker.py`):
```bash
docker-compose up -d
docker-compose ps  # verify all services running
```

### For local scripts (`run_masterclass.sh`):
```bash
python3 -m venv venv
source venv/bin/activate
pip install pyspark delta-spark psutil
```

---

## 🐛 Troubleshooting

**"Virtual environment not found!"**
```bash
python3 -m venv venv
source venv/bin/activate
pip install pyspark delta-spark psutil
```

**"Command 'docker-compose' not found"**
- Install Docker Desktop or Docker Compose: https://docs.docker.com/compose/install/

**"ModuleNotFoundError: No module named 'pyspark'"**
```bash
source venv/bin/activate
pip install pyspark
```

**Services unreachable in `test_simple_docker.py`?**
```bash
docker-compose ps  # Check which containers failed
docker-compose logs spark-master  # Check logs
docker-compose restart spark-master  # Restart if stuck
```

---

## 📚 Learning Path

1. **Start here**: `test_simple_docker.py` (verify setup)
2. **Quick learning**: `small_cluster_benchmark.py` (see it work fast)
3. **Deep dive**: `spark_layout_masterclass.py` (understand all formats)
4. **Modify & experiment**: Edit scripts to test custom scenarios

---

## 🎯 What You'll Learn

By running these scripts, you'll understand:

✓ How different partition strategies affect query performance  
✓ Trade-offs between write time and query time  
✓ When to use Delta Lake vs Iceberg vs Parquet  
✓ How to measure data warehouse performance  
✓ Real performance numbers (5-7x speedups demonstrated)

---

**Happy learning! 🚀**
