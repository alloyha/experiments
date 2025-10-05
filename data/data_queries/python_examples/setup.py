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

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    
    for package in REQUIREMENTS:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def create_config_template():
    """Create configuration template file"""
    config_template = '''
# PostgreSQL Schema Generator Configuration
# Copy this file to config.py and update with your settings

from schema_generator_client import ConnectionConfig

# Database connection settings
DATABASE_CONFIG = ConnectionConfig(
    host="localhost",
    port=5432,
    database="your_database",
    username="your_username", 
    password="your_password",
    schema="public"
)

# Schema generation settings
DEFAULT_SCHEMA = "analytics"
ENABLE_LOGGING = True
'''
    
    with open("config_template.py", "w") as f:
        f.write(config_template.strip())
    
    print("✅ Created config_template.py")

def main():
    """Setup the Python environment"""
    print("🚀 PostgreSQL Schema Generator - Python Setup")
    print("=" * 50)
    
    install_requirements()
    create_config_template()
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Copy config_template.py to config.py")
    print("2. Update config.py with your database settings")
    print("3. Import and use the schema generator:")
    print("   from schema_generator_client import SchemaGenerator")
    print("   from config import DATABASE_CONFIG")
    print("   generator = SchemaGenerator(DATABASE_CONFIG)")

if __name__ == "__main__":
    main()