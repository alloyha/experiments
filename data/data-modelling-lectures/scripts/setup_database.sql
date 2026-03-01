-- ==============================================
-- SCRIPT DE SETUP: CURSO MODELAGEM DE DADOS
-- ==============================================

-- 1. Resetar o Ambiente
DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;

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
-- INICIALIZAÇÃO DE MÓDULOS (DDL + SEED)
-- ==============================================

\i /scripts/setup_biblioteca.sql
\i /scripts/setup_varejo.sql
\i /scripts/setup_rede_social.sql
