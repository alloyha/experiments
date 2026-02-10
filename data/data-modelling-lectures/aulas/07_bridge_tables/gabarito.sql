-- ==============================================
-- GABARITO AULA 07: MODELAGEM DIMENSIONAL PRÁTICA
-- ==============================================

-- ==============================================
-- EXERCÍCIO 3: Modelagem Dimensional
-- ==============================================

-- a) Inserir 10 novos produtos em categorias diferentes
INSERT INTO dim_produto (produto_sk, nome_produto, categoria, preco_sugerido) VALUES
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

-- b) Configurar bridge table com pesos para produtos multi-categoria
-- Inserindo categorias novas primeiro se necessário
INSERT INTO dim_categoria (nome_categoria, departamento) VALUES 
('Acessórios', 'Varejo'), ('Móveis', 'Casa'), ('Eletroportáteis', 'Casa'), ('Esportes', 'Lazer');

-- Exemplo fictício de bridge
-- Produto ID X (digamos que seja o Smartphone, id 4)
-- INSERT INTO bridge_produto_categoria... (seria necessário ver os IDs gerados)

-- c) Criar 50 vendas distribuídas em junho/2024
INSERT INTO fato_vendas (tempo_id, produto_id, cliente_sk, loja_id, indicador_id, quantidade, valor_unitario, valor_total, numero_nota_fiscal)
SELECT 
    t.tempo_id,
    p.produto_id,
    (SELECT cliente_sk FROM dim_cliente WHERE registro_ativo = TRUE LIMIT 1), -- Pega um cliente qualquer
    (SELECT loja_id FROM dim_loja LIMIT 1),
    1,
    1,
    p.preco_sugerido,
    p.preco_sugerido,
    'NF-AUTO-' || t.tempo_id
FROM dim_tempo t
CROSS JOIN dim_produto p
WHERE t.mes = 6 AND t.ano = 2024
LIMIT 50;

-- d) Implementar mudança SCD Type 2 em 2 clientes
-- Cliente 102 muda de endereço
UPDATE dim_cliente SET data_fim = CURRENT_DATE - 1, registro_ativo = FALSE 
WHERE cliente_id = 102 AND registro_ativo = TRUE;

INSERT INTO dim_cliente (cliente_id, nome, email, cidade, estado, segmento, data_inicio, versao)
SELECT cliente_id, nome, email, 'Nova Cidade', 'NC', segmento, CURRENT_DATE, versao + 1
FROM dim_cliente WHERE cliente_id = 102 AND versao = (SELECT MAX(versao) FROM dim_cliente WHERE cliente_id = 102);


-- ==============================================
-- EXERCÍCIO 4: Análises Dimensionais
-- ==============================================

-- a) Total de vendas por mês e categoria (usando bridge se necessário, ou direto do produto se simples)
SELECT dt.mes_nome, dc.nome_categoria, SUM(fv.valor_total) 
FROM fato_vendas fv
JOIN dim_tempo dt ON fv.tempo_id = dt.tempo_id
JOIN dim_produto dp ON fv.produto_id = dp.produto_id
-- Se usar bridge, o join muda
JOIN bridge_produto_categoria bpc ON dp.produto_id = bpc.produto_id
JOIN dim_categoria dc ON bpc.categoria_id = dc.categoria_id
GROUP BY dt.mes_nome, dc.nome_categoria;

-- b) Top 5 produtos mais vendidos
SELECT dp.nome_produto, SUM(fv.quantidade) as total_qtd
FROM fato_vendas fv
JOIN dim_produto dp ON fv.produto_id = dp.produto_id
GROUP BY dp.nome_produto
ORDER BY total_qtd DESC
LIMIT 5;

-- c) Análise de vendas por segmento de cliente (considerando SCD)
-- O join deve ser feito pela SK para pegar o segmento NO MOMENTO DA VENDA
SELECT dc.segmento, SUM(fv.valor_total)
FROM fato_vendas fv
JOIN dim_cliente dc ON fv.cliente_sk = dc.cliente_sk
GROUP BY dc.segmento;

-- d) Comparar vendas por região
SELECT dl.regiao, SUM(fv.valor_total)
FROM fato_vendas fv
JOIN dim_loja dl ON fv.loja_id = dl.loja_id
GROUP BY dl.regiao;

-- ==============================================
-- ASSERTIONS (VALIDAÇÃO DE RESULTADOS)
-- ==============================================
DO $$
BEGIN
   -- Validação 1: Volume de Vendas
   IF (SELECT COUNT(*) FROM fato_vendas) < 50 THEN
      RAISE EXCEPTION 'Erro: Esperado ao menos 50 vendas, encontrado %', (SELECT COUNT(*) FROM fato_vendas);
   END IF;

   -- Validação 2: SCD Type 2 do cliente 102
   -- Deve haver 2 registros (um inativo e um ativo)
   IF (SELECT COUNT(*) FROM dim_cliente WHERE cliente_id = 102) != 2 THEN
      RAISE EXCEPTION 'Erro: Falha no versionamento SCD do cliente 102. Esperado 2 versões, encontrado %', (SELECT COUNT(*) FROM dim_cliente WHERE cliente_id = 102);
   END IF;

   -- Validação 3: Consistência do Join Fato-SK
   -- Se o join falhar, o sum seria null ou daria erro
   IF (SELECT SUM(fv.valor_total) FROM fato_vendas fv JOIN dim_cliente dc ON fv.cliente_sk = dc.cliente_sk) IS NULL THEN
      RAISE EXCEPTION 'Erro: Falha na integridade referencial entre fato e dim_cliente (SK)';
   END IF;

   RAISE NOTICE 'VALIDAÇÃO AULA 07: SUCESSO! ✅';
END $$;
