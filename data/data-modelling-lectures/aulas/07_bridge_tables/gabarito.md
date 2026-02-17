# GABARITO AULA 07: MODELAGEM DIMENSIONAL PRÁTICA

## EXERCÍCIO 3: Modelagem Dimensional

**a) Inserir 10 novos produtos em categorias diferentes**
Usamos produto_id para a Natural Key (Idempotente).

```sql
INSERT INTO dim_produto (produto_id, nome_produto, categoria, preco_sugerido) VALUES
('P001', 'Smartphone X', 'Eletrônicos', 2500.00),
('P002', 'Smart TV 50', 'Eletrônicos', 3200.00),
('P003', 'Fone Bluetooth', 'Acessórios', 150.00),
('P004', 'Cadeira Gamer', 'Móveis', 1200.00),
('P005', 'Mesa Escritório', 'Móveis', 800.00),
('P006', 'Cafeteira', 'Eletroportáteis', 300.00),
('P007', 'Liquidificador', 'Eletroportáteis', 200.00),
('P008', 'Tênis Corrida', 'Esportes', 450.00),
('P009', 'Bola Futebol', 'Esportes', 100.00),
('P010', 'Mochila', 'Acessórios', 180.00);
```

**b) Configurar bridge table com pesos para produtos multi-categoria**

```sql
INSERT INTO dim_categoria (nome_categoria, departamento) VALUES
('Acessórios', 'Varejo'), ('Móveis', 'Casa'), ('Eletroportáteis', 'Casa'), ('Esportes', 'Lazer');
```

**c) Criar 50 vendas distribuídas em junho/2024**
Note: Agora usamos produto_sk e cliente_sk para integridade referencial correta, e datas nativas.

```sql
INSERT INTO fato_vendas (
    data_venda, produto_sk, cliente_sk, quantidade, valor_total
)
SELECT
    '2024-06-01'::DATE + (n % 30) * INTERVAL '1 day',
    p.produto_sk,
    c.cliente_sk,
    1 AS quantidade,
    p.preco_sugerido AS valor_total
FROM GENERATE_SERIES(1, 50) n
CROSS JOIN (SELECT produto_sk, preco_sugerido FROM dim_produto LIMIT 1) p
CROSS JOIN (SELECT cliente_sk FROM dim_cliente LIMIT 1) c
LIMIT 50;
```

## EXERCÍCIO 4: Análises Dimensionais

**a) Total de vendas por mês e categoria**

```sql
SELECT
    TO_CHAR(fv.data_venda, 'Month') AS mes_nome,
    dc.nome_categoria,
    SUM(fv.valor_total) AS total_vendas
FROM fato_vendas AS fv
INNER JOIN dim_produto AS dp ON fv.produto_sk = dp.produto_sk
LEFT JOIN bridge_produto_categoria AS bpc ON dp.produto_sk = bpc.produto_sk
LEFT JOIN dim_categoria AS dc ON bpc.categoria_id = dc.categoria_id
GROUP BY 1, 2;
```

**b) Top 5 produtos mais vendidos**

```sql
SELECT
    dp.nome_produto,
    SUM(fv.quantidade) AS total_qtd
FROM fato_vendas AS fv
INNER JOIN dim_produto AS dp ON fv.produto_sk = dp.produto_sk
GROUP BY dp.nome_produto
ORDER BY total_qtd DESC
LIMIT 5;
```

**c) Análise de vendas por segmento de cliente**

```sql
SELECT
    dc.segmento,
    SUM(fv.valor_total) AS total_vendas
FROM fato_vendas AS fv
INNER JOIN dim_cliente AS dc ON fv.cliente_sk = dc.cliente_sk
GROUP BY dc.segmento;
```

**d) Comparar vendas por região**

```sql
SELECT
    dl.regiao,
    SUM(fv.valor_total) AS total_vendas
FROM fato_vendas AS fv
INNER JOIN dim_loja AS dl ON fv.loja_id = dl.loja_id
GROUP BY dl.regiao;
```

### ASSERTIONS (VALIDAÇÃO DE RESULTADOS)

```sql
DO $$
BEGIN
   -- Validação 1: Volume de Vendas
   IF (SELECT COUNT(*) FROM fato_vendas) < 50 THEN
      RAISE EXCEPTION 'Erro: Esperado ao menos 50 vendas, encontrado %', (SELECT COUNT(*) FROM fato_vendas);
   END IF;

   -- Validação 3: Consistência do Join Fato-Cliente (Usando SK)
   IF (SELECT SUM(fv.valor_total) FROM fato_vendas fv JOIN dim_cliente dc ON fv.cliente_sk = dc.cliente_sk) IS NULL THEN
      RAISE EXCEPTION 'Erro: Falha na integridade referencial entre fato e dim_cliente';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 07: SUCESSO! ✅';
END $$;
```
