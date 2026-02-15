-- ==============================================================================
-- Aula 4: Modelagem Dimensional - Conceitos
-- ==============================================================================

-- Exemplo de Query Analítica em Modelo Dimensional (Star Schema)
-- Objetivo: Total de vendas e quantidade vendida por Ano, Mês e Categoria

/*
Estrutura assumida:
- fato_vendas (tabela central)
- dim_tempo (dimensão)
- dim_produto (dimensão)
*/

SELECT
    dt.ano,
    dt.mes,
    dp.categoria,
    SUM(fv.valor_total) AS total_vendas,
    SUM(fv.quantidade) AS qtd_vendida
FROM fato_vendas AS fv
INNER JOIN dim_tempo AS dt ON fv.tempo_id = dt.tempo_id
INNER JOIN dim_produto AS dp ON fv.produto_sk = dp.produto_sk
WHERE dt.ano = 2024
GROUP BY dt.ano, dt.mes, dp.categoria;
