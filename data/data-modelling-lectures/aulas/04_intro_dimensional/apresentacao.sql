-- ==============================================================================
-- Aula 4: Modelagem Dimensional - Conceitos
-- ==============================================================================

-- Exemplo de Query Analítica em Modelo Dimensional (Star Schema)
-- Objetivo: Total de vendas e quantidade vendida por Ano, Mês e Categoria

/*
Estrutura assumida:
- fato_vendas (tabela central)
- dim_produto (dimensão)
- dim_cliente (dimensão)
*/

SELECT
    dp.categoria,
    EXTRACT(YEAR FROM fv.data_venda) AS ano,
    EXTRACT(MONTH FROM fv.data_venda) AS mes,
    SUM(fv.valor_total) AS total_vendas,
    SUM(fv.quantidade) AS qtd_vendida
FROM fato_vendas AS fv
INNER JOIN dim_produto AS dp ON fv.produto_sk = dp.produto_sk
WHERE EXTRACT(YEAR FROM fv.data_venda) = 2024
GROUP BY 1, 2, 3;
