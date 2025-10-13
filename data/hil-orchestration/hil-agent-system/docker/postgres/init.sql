-- PostgreSQL initialization script
-- Creates the database and enables required extensions

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE hil_agent_system'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'hil_agent_system')\gexec

-- Connect to the database and enable extensions
\c hil_agent_system;

-- Enable pgvector extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable uuid-ossp for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pg_trgm for similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Enable btree_gin for better indexing
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create indexes for better performance
-- (Tables will be created by the application)