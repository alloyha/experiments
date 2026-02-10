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
    SUM(fv.valor_total) as total_vendas,
    SUM(fv.quantidade) as qtd_vendida
FROM fato_vendas fv
JOIN dim_tempo dt ON fv.tempo_id = dt.tempo_id
JOIN dim_produto dp ON fv.produto_id = dp.produto_id
WHERE dt.ano = 2024
GROUP BY dt.ano, dt.mes, dp.categoria;
