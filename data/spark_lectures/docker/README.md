# 🐳 Docker Configuration

Docker-specific files for the Spark Data Layout Masterclass infrastructure.

## 📁 Contents

- **docker-compose.yaml** - Complete service orchestration (7 services)
- **Dockerfile.spark** - Custom Apache Spark 3.5.3 image with pre-installed packages
- **Dockerfile.hive** - Custom Apache Hive 4.0.0 with PostgreSQL JDBC driver
- **conf_trino/** - Trino SQL engine configuration

## 🚀 Quick Start

From the **project root** (not this folder):

```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Run benchmark
docker-compose exec -T spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py

# Stop all services
docker-compose down
```

## 🏗️ Architecture

```
docker-compose.yaml orchestrates:

┌─────────────────────────────────────────┐
│ Storage Layer                           │
├─────────────────────────────────────────┤
│ • MinIO (S3-compatible object storage) │
│ • PostgreSQL (Hive metastore backend)  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Metadata & Query Layer                  │
├─────────────────────────────────────────┤
│ • Hive Metastore (thrift://hive-...:9083) │
│ • Iceberg REST Catalog (8181)           │
│ • Trino SQL Engine (8085) [optional]    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Compute Layer                           │
├─────────────────────────────────────────┤
│ • Spark Master (7077/8080)              │
│ • Spark Workers (8081, 8082)            │
└─────────────────────────────────────────┘
```

## 📊 Services Reference

| Service | Port | Purpose |
|---------|------|---------|
| **MinIO** | 9000/9001 | S3-compatible storage + console |
| **PostgreSQL** | 5432 | Hive metastore database |
| **Hive Metastore** | 9083 | Metadata management (thrift) |
| **Spark Master** | 7077/8080 | Cluster coordinator + UI |
| **Spark Workers** | 8081-8082 | Computation nodes |
| **Iceberg REST** | 8181 | Iceberg catalog service |
| **Trino** | 8085 | SQL query engine (optional) |

## 🔧 Dockerfile Details

### Dockerfile.spark
- Base: `apache/spark:3.5.3`
- Adds:
  - delta-spark 3.2.0
  - pyiceberg 0.7.1
  - psutil 7.0.0
  - Pre-downloaded JARs (Delta, Iceberg, Hadoop-AWS)
- Purpose: Eliminates Ivy cache issues, faster startup

### Dockerfile.hive
- Base: `apache/hive:4.0.0`
- Adds:
  - PostgreSQL JDBC driver (`postgresql-42.7.1.jar`)
  - Creates metastore schema automatically
- Purpose: Enables Hive Metastore to use PostgreSQL backend

## 📝 Important Configuration

All paths in docker-compose.yaml are relative to the project root (where the symlink is):

```yaml
build:
  context: ..              # Go up to project root
  dockerfile: docker/Dockerfile.spark  # Then into docker folder
```

This allows running `docker-compose` from project root or from this folder.

## 🔐 Default Credentials

| Service | User | Password |
|---------|------|----------|
| **MinIO** | admin | password |
| **Hive/Postgres** | hive | hive |
| **Trino** | default | (no password) |

⚠️ **For production**: Update `.env` file or environment variables

## 🌐 Network

All services connect via `spark_net` Docker network:
- Service-to-service communication via hostname (e.g., `hive-metastore:9083`)
- Access from host via `localhost`

## 📤 Volume Mounts

| Container | Volume | Purpose |
|-----------|--------|---------|
| minio | `minio-data` | Persistent object storage |
| postgres | `postgres-data` | Persistent metadata database |
| iceberg-rest | `iceberg-rest-data` | REST catalog data |

## 🛠️ Common Tasks

### Rebuild images
```bash
docker-compose build
```

### View logs
```bash
docker-compose logs -f spark-master
docker-compose logs -f hive-metastore
docker-compose logs -f minio
```

### Execute commands in container
```bash
docker-compose exec spark-master python3 /opt/spark/scripts/small_cluster_benchmark.py
docker-compose exec spark-master bash
docker-compose exec -T spark-master spark-shell
```

### Stop and clean
```bash
docker-compose down          # Stop services
docker-compose down -v       # Stop and remove volumes
docker-compose restart       # Restart all services
```

## 🐛 Troubleshooting

### Services won't start
```bash
# Check what went wrong
docker-compose logs <service-name>

# Rebuild and restart
docker-compose build
docker-compose up -d
```

### Permission errors
Usually due to volume mount permissions. Rebuild containers:
```bash
docker-compose build --no-cache
docker-compose restart
```

### Out of disk space
Clean up Docker resources:
```bash
docker system prune -a --volumes
docker-compose build
```

### Services intermittently failing
Increase startup time in healthchecks:
```yaml
healthcheck:
  retries: 10        # Increase from 5
  start_period: 30s  # Add this
```

## 📚 Additional Resources

- **Project Documentation**: See `../README.md`
- **Command Reference**: See `../COMMAND_REFERENCE.md`
- **Technical Details**: See `../DOCKER_INTEGRATION_COMPLETE.md`
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **Apache Spark Docs**: https://spark.apache.org/docs/latest/
- **Hive Docs**: https://hive.apache.org/

## ✅ Health Check

Verify all services are healthy:

```bash
docker-compose ps

# Or run the test
docker-compose exec -T spark-master python3 /opt/spark/scripts/test_simple_docker.py
```

Expected output: 4-5 services responding ✓

---

**Note**: All `docker-compose` commands should be run from the project root, not from this folder (thanks to the symlink).
