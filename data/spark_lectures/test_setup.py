#!/usr/bin/env python3
"""
test_spark_setup.py

Quick test to verify PySpark is working correctly in your environment.
Run this first before the full masterclass.

Usage:
    python3 test_spark_setup.py
"""

import sys
import os

print("="*80)
print("PYSPARK ENVIRONMENT TEST")
print("="*80)

# Test 1: Check Python version
print("\n[1/6] Checking Python version...")
py_version = sys.version_info
print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 7):
    print("⚠️  Warning: Python 3.7+ recommended for PySpark")

# Test 2: Check Java
print("\n[2/6] Checking Java installation...")
try:
    import subprocess
    result = subprocess.run(['java', '-version'], 
                          capture_output=True, 
                          text=True,
                          timeout=5)
    if result.returncode == 0 or result.stderr:
        java_info = result.stderr.split('\n')[0] if result.stderr else "Java detected"
        print(f"✓ {java_info}")
        
        # Check JAVA_HOME
        java_home = os.environ.get('JAVA_HOME')
        if java_home:
            print(f"✓ JAVA_HOME is set: {java_home}")
        else:
            print("⚠️  JAVA_HOME not set (usually okay)")
    else:
        print("❌ Java found but returned error")
        sys.exit(1)
except FileNotFoundError:
    print("❌ Java not found!")
    print("\nPlease install Java 8 or 11:")
    print("  Ubuntu/Debian: sudo apt-get install openjdk-11-jdk")
    print("  macOS: brew install openjdk@11")
    print("\nThen optionally set JAVA_HOME:")
    print("  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64")
    sys.exit(1)
except subprocess.TimeoutExpired:
    print("❌ Java check timed out")
    sys.exit(1)

# Test 3: Import PySpark
print("\n[3/6] Importing PySpark...")
try:
    from pyspark import __version__ as spark_version
    from pyspark.sql import SparkSession
    print(f"✓ PySpark {spark_version} imported successfully")
except ImportError as e:
    print(f"❌ Failed to import PySpark: {e}")
    print("\nPlease install: pip install pyspark")
    sys.exit(1)

# Test 4: Create Spark Session
print("\n[4/6] Creating Spark session...")
os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')

try:
    spark = (SparkSession.builder
             .appName("SparkTest")
             .master("local[2]")
             .config("spark.driver.memory", "2g")
             .config("spark.driver.host", "127.0.0.1")
             .config("spark.driver.bindAddress", "127.0.0.1")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    
    spark.sparkContext.setLogLevel("ERROR")
    print(f"✓ Spark session created")
    print(f"  Version: {spark.version}")
    print(f"  Master: {spark.sparkContext.master}")
    print(f"  App Name: {spark.sparkContext.appName}")
    
except Exception as e:
    print(f"❌ Failed to create Spark session: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure no other Spark session is running")
    print("2. Try: export SPARK_LOCAL_IP=127.0.0.1")
    print("3. Check available memory (need at least 2GB free)")
    print("4. Close other heavy applications")
    sys.exit(1)

# Test 5: Run simple DataFrame operation
print("\n[5/6] Testing DataFrame operations...")
try:
    df = spark.range(1000).selectExpr("id", "id * 2 as doubled")
    count = df.count()
    sample = df.first()
    print(f"✓ Created DataFrame with {count} rows")
    print(f"✓ Sample: id={sample.id}, doubled={sample.doubled}")
except Exception as e:
    print(f"❌ DataFrame operation failed: {e}")
    sys.exit(1)

# Test 6: Test Parquet write/read
print("\n[6/6] Testing Parquet I/O...")
test_dir = "/tmp/spark_test_parquet"
try:
    # Clean up if exists
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Write
    df.write.mode("overwrite").parquet(test_dir)
    print(f"✓ Written Parquet to {test_dir}")
    
    # Read
    df_read = spark.read.parquet(test_dir)
    read_count = df_read.count()
    print(f"✓ Read Parquet: {read_count} rows")
    
    # Cleanup
    shutil.rmtree(test_dir)
    print("✓ Cleaned up test files")
    
except Exception as e:
    print(f"❌ Parquet I/O failed: {e}")
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
        except:
            pass
    sys.exit(1)

# Success!
print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nYour environment is ready for the PySpark masterclass.")
print("\nOptional enhancements:")
print("  • pip install psutil      (for automatic memory detection)")
print("  • pip install delta-spark (for Delta Lake examples)")

# Display system info
print("\n📊 System Information:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"  Total RAM: {mem.total / (1024**3):.1f} GB")
    print(f"  Available RAM: {mem.available / (1024**3):.1f} GB")
    print(f"  CPU cores: {os.cpu_count()}")
except ImportError:
    print("  (Install psutil for system info)")

print("\n🚀 Next step: python3 spark_layout_masterclass.py")
print("="*80)

# Clean up
spark.stop()
print("\n✓ Spark session stopped cleanly")