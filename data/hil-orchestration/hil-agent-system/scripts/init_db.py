#!/usr/bin/env python3
"""
Database initialization script.
Creates the initial database schema.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.core.database import Base, create_database_engine


async def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    
    # Import all models to register them with Base
    from app.models import agent, workflow, tool, memory, execution
    
    settings = get_settings()
    engine = create_database_engine()
    
    try:
        async with engine.begin() as conn:
            # Create pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        
        print("✅ Database tables created successfully!")
        
        # Print created tables
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]
            
        print(f"\n📊 Created {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
            
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        sys.exit(1)
    finally:
        await engine.dispose()


async def main():
    """Main function."""
    print("🚀 HIL Agent System - Database Setup")
    print("=" * 50)
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    settings = get_settings()
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Database URL: {settings.DATABASE_URL}")
    print()
    
    # Create tables
    await create_tables()
    
    print("\n🎉 Database setup completed!")
    print("\nNext steps:")
    print("1. Start PostgreSQL and Redis services")
    print("2. Run the application: python -m app.main")
    print("3. Visit http://localhost:8000/docs for API documentation")


if __name__ == "__main__":
    # Add imports for the script
    from sqlalchemy import text
    asyncio.run(main())