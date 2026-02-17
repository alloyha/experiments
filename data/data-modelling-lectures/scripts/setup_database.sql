-- ==============================================
-- SCRIPT DE SETUP: CURSO MODELAGEM DE DADOS
-- ==============================================

-- 1. Resetar o Ambiente
DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO public;

-- Criar Schemas Organizacionais
DROP SCHEMA IF EXISTS biblioteca CASCADE;
CREATE SCHEMA biblioteca;

DROP SCHEMA IF EXISTS varejo CASCADE;
CREATE SCHEMA varejo;

DROP SCHEMA IF EXISTS rede_social CASCADE;
CREATE SCHEMA rede_social;

-- Configurar Search Path do usuário aluno
ALTER ROLE aluno SET search_path TO public, biblioteca, varejo, rede_social;
SET search_path TO public, biblioteca, varejo, rede_social;

-- ==============================================
-- SEEDS: Carregar dados iniciais de cada schema
-- ==============================================

\i seed_biblioteca.sql
\i seed_varejo.sql
\i seed_rede_social.sql
