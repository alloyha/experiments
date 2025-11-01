#!/usr/bin/env python3
"""
test_simple_docker.py

Simple test to verify Docker services (no spark-submit/Ivy issues)
Just tests basic connectivity to all services.
"""

import sys
import socket
from datetime import datetime

print("\n" + "="*80)
print("DOCKER SERVICE CONNECTIVITY TEST")
print("="*80)
print(f"Started: {datetime.now().isoformat()}\n")

services = [
    ("Hive Metastore", "hive-metastore", 9083),
    ("PostgreSQL", "postgres-metastore", 5432),
    ("MinIO S3 API", "minio", 9000),
    ("Spark Master", "spark-master", 7077),
    ("Iceberg REST", "iceberg-rest", 8181),
]

results = {}
for name, host, port in services:
    print(f"Testing {name} ({host}:{port})...", end=" ")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    
    if result == 0:
        print("✓")
        results[name] = True
    else:
        print("✗")
        results[name] = False

print("\n" + "="*80)
print("SERVICE CONNECTIVITY RESULTS")
print("="*80)

success_count = sum(1 for v in results.values() if v)
for name, reachable in results.items():
    status = "✓ Available" if reachable else "✗ Unreachable"
    print(f"{name:30s} {status}")

print("\n" + "="*80)

if success_count >= 5:
    print("✓ All services are reachable!")
    print("\n📝 Next steps:")
    print("  1. Create a smaller benchmark script for cluster")
    print("  2. Test S3 connectivity from PySpark")
    print("  3. Run full benchmark with all table formats")
    sys.exit(0)
else:
    print(f"⚠ {6 - success_count} services unreachable")
    sys.exit(1)
