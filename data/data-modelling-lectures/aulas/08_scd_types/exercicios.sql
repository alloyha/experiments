-- ==============================================
-- EXERCÍCIOS AULA 08: SLOWLY CHANGING DIMENSIONS
-- ==============================================

-- EXERCÍCIO 1: Estratégias SCD (Conceitual)
-- Cenário: Você tem uma tabela de Clientes.
-- a) O campo 'telefone' muda frequentemente. Qual tipo de SCD é **geralmente** recomendado para evitar crescimento excessivo da tabela?
-- b) O campo 'categoria_cliente' (VIP, Regular) é crítico para análise histórica de vendas. Qual tipo SCD usar?

-- EXERCÍCIO 2: Type 2 vs. Log Append-Only
-- a) Compare a estratégia clássica de SCD Type 2 (UPDATE + INSERT) com uma estratégia de Append-Only (apenas INSERT com timestamp).
-- b) Qual delas é mais performática para carga massiva (Big Data)?

-- EXERCÍCIO 3: Implementação Type 2 Básica
-- a) Crie uma tabela 'dim_cliente_scd2' com colunas: id, nome, cidade, data_validade_inicio, data_validade_fim, ativo.
-- b) Simule a mudança de cidade do cliente 101.
