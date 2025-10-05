"""
Installation and Setup Guide
============================

Quick start guide for using the PostgreSQL Schema Generator Python client.
"""

import subprocess
import sys
from pathlib import Path

# Required packages
REQUIREMENTS = [
    "psycopg2-binary>=2.9.0",
    "pandas>=1.3.0", 
    "sqlalchemy>=1.4.0"
]

# Optional packages for enhanced ETL integrations
OPTIONAL_REQUIREMENTS = [
    "duckdb>=0.8.0",  # For OLAP analytics and embedded database
    "polars>=0.19.0",  # For high-performance data processing
]

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    
    for package in REQUIREMENTS:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def install_optional_requirements():
    """Install optional packages for enhanced ETL features"""
    print("\n📦 Installing optional ETL packages...")
    print("   (These enable DuckDB and Polars integrations)")
    
    for package in OPTIONAL_REQUIREMENTS:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to install {package}: {e}")
            print(f"   Install manually with: pip install {package}")

def install_all():
    """Install both required and optional packages"""
    install_requirements()
    install_optional_requirements()

def run_config():
    """Create configuration template file"""
    config_template = '''

'''
    
    with open("config_template.py", "w") as f:
        f.write(config_template.strip())
    
    print("✅ Created config_template.py")

def main():
    """Setup the Python environment"""
    print("🚀 PostgreSQL Schema Generator - Python Setup")
    print("=" * 50)
    
    install_requirements()
    run_config()
    
    print("\n🔄 Enhanced ETL Features Available!")
    print("To enable DuckDB and Polars integrations, run:")
    print("   python setup.py install_optional")
    print("Or install manually:")
    print("   pip install duckdb>=0.8.0 polars>=0.19.0")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Copy config_template.py to config.py")
    print("2. Update config.py with your database settings")
    print("3. Import and use the schema generator:")
    print("   from schema_generator_client import SchemaGenerator")
    print("   from config import DATABASE_CONFIG")
    print("   generator = SchemaGenerator(DATABASE_CONFIG)")

def install_optional():
    """Entry point for installing optional packages"""
    install_optional_requirements()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "install_optional":
        install_optional()
    else:
        main()