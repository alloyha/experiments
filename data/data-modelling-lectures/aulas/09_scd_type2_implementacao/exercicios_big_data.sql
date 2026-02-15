-- ==============================================
-- EXERCÍCIOS AULA 09: SCD TYPE 2 PARA BIG DATA (HISTÓRICO ANINHADO)
-- ==============================================

-- EXERCÍCIO 1: Estrutura Aninhada (Comparação)
-- Cenário: Cliente mudou de segmento 5 vezes.
-- a) Quantas linhas você terá na abordagem clássica Type 2 (SCD em linhas)?
-- b) Quantas linhas você terá na abordagem moderna (SCD aninhado - STRUCT[])?
-- c) Qual o benefício principal desta abordagem para joins e shuffle?

-- EXERCÍCIO 2: Implementação de Histórico
-- a) Crie a tabela 'dim_cliente_historico_array' com um campo 'historico' TYPE array de structs.
-- b) Adicione manualmente um cliente com um "histórico de 2 anos" (2 eventos diferentes:
--    Bronze -> Prata).

-- EXERCÍCIO 3: Query "Point-in-Time"
-- a) Extraia as versões de segmento do cliente 1.
-- b) Use a cláusula UNNEST para encontrar em qual segmento o cliente estava na data '2022-06-15'.
