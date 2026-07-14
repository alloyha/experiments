-- Pocket Duel: queries exploratórias
-- Uso: duckdb pocketduel.duckdb < explore.sql
-- ou copie/cole trechos individualmente na CLI interativa (duckdb pocketduel.duckdb)


-- 1) Visão geral: quantos registros temos e cobertura do crawl_log
SELECT 'cards' AS table_name, COUNT(*) AS rows FROM cards
UNION ALL
SELECT 'duelists', COUNT(*) FROM duelists;

SELECT entity_type, status, COUNT(*) AS n
FROM crawl_log
GROUP BY entity_type, status
ORDER BY entity_type, status;


-- 2) Top 15 cartas por ataque
SELECT
    id,
    name,
    (fields->>'attack')::INT AS attack,
    (fields->>'defense')::INT AS defense,
    fields->>'type' AS type,
    fields->>'attribute' AS attribute
FROM cards
WHERE fields->>'attack' IS NOT NULL
ORDER BY attack DESC
LIMIT 15;


-- 3) Distribuição por tipo de carta (Dragon, Warrior, Spell, Trap, etc.)
SELECT fields->>'type' AS card_type, COUNT(*) AS n
FROM cards
GROUP BY card_type
ORDER BY n DESC;


-- 4) Distribuição por atributo (Light, Dark, Water, Fire, Earth, Wind)
SELECT fields->>'attribute' AS attribute, COUNT(*) AS n
FROM cards
WHERE fields->>'attribute' IS NOT NULL
GROUP BY attribute
ORDER BY n DESC;


-- 5) Cartas por raridade (contagem de estrelas no nome, campo `stars`)
SELECT (fields->>'stars')::INT AS stars, COUNT(*) AS n
FROM cards
GROUP BY stars
ORDER BY stars DESC;


-- 6) Cartas sem NENHUMA fonte de drop conhecida (nem em deck de duelista, nem no
--    próprio campo `drops` da carta) -- útil para achar cartas raras/inobtíveis
--    ou possíveis lacunas no crawl.
WITH card_drop_counts AS (
    SELECT c.id, c.name, json_array_length(c.fields->'drops') AS n_drops
    FROM cards c
)
SELECT id, name
FROM card_drop_counts
WHERE n_drops = 0
ORDER BY id;


-- 7) Para uma carta específica: todas as fontes de drop (duelista + rank + probabilidade)
SELECT
    c.name AS card_name,
    d.value->>'duelist' AS duelist,
    d.value->>'rank' AS rank,
    d.value->>'probability' AS probability
FROM cards c, json_each(c.fields->'drops') AS d
WHERE c.name = 'Blue-eyes White Dragon';


-- 8) Deck completo de um duelista específico, com probabilidade
SELECT
    du.name AS duelist,
    e.value->>'card_name' AS card_name,
    (e.value->>'card_id')::INT AS card_id,
    e.value->>'probability' AS probability
FROM duelists du, json_each(du.fields->'drops'->'deck') AS e
WHERE du.name = 'Kaiba'
ORDER BY probability DESC;


-- 9) Junta o deck de um duelista com os dados reais da carta (ATK/DEF/tipo)
--    -- mostra o que ele realmente pode largar, com stats
SELECT
    du.name AS duelist,
    (e.value->>'card_id')::INT AS card_id,
    e.value->>'probability' AS probability,
    c.name AS card_name,
    (c.fields->>'attack')::INT AS attack,
    (c.fields->>'defense')::INT AS defense,
    c.fields->>'type' AS type
FROM duelists du, json_each(du.fields->'drops'->'deck') AS e
JOIN cards c ON c.id = (e.value->>'card_id')::INT
WHERE du.name = 'Kaiba'
ORDER BY attack DESC;


-- 10) Quais cartas mais aparecem como material de equipamento (equip_materials)
--     em outras cartas -- um proxy de "popularidade"/utilidade de equip
SELECT
    e.value->>'card_name' AS equip_name,
    (e.value->>'card_id')::INT AS equip_card_id,
    COUNT(*) AS used_by_n_cards
FROM cards c, json_each(c.fields->'equip_materials') AS e
GROUP BY equip_name, equip_card_id
ORDER BY used_by_n_cards DESC
LIMIT 15;


-- 11) Duelistas ranqueados por tamanho do deck (quantas cartas distintas cada um pode largar)
SELECT
    du.name AS duelist,
    json_array_length(du.fields->'drops'->'deck') AS deck_size
FROM duelists du
ORDER BY deck_size DESC;


-- 12) Cartas cujo password não segue o padrão esperado (8 dígitos) -- checagem de qualidade
SELECT id, name, fields->>'password' AS password
FROM cards
WHERE fields->>'password' IS NOT NULL
  AND LENGTH(fields->>'password') <> 8;


-- 13) Full-text-ish: buscar cartas pela descrição (ex: menções a "burn", "direct damage", etc.)
SELECT id, name, fields->>'description' AS description
FROM cards
WHERE fields->>'description' ILIKE '%burn%';


-- 14) MANUTENÇÃO: corrige entity_type antigo ('duelists' -> 'duelist') e remove
--     resquícios de 'starters' no crawl_log (dados de quando ainda existia esse crawl)
UPDATE crawl_log SET entity_type = 'duelist' WHERE entity_type = 'duelists';
DELETE FROM crawl_log WHERE entity_type = 'starters';


-- 15) Dos 59 erros de card, quantos são ids <=722 (dentro do range válido, valem
--     retry) vs >722 (passaram do teto e são só ruído dos 500 esperados)
SELECT
    CASE WHEN entity_id <= 722 THEN 'dentro do range (retry útil)' ELSE 'além do teto (ruído esperado)' END AS bucket,
    COUNT(DISTINCT entity_id) AS n_ids
FROM crawl_log
WHERE entity_type = 'card' AND status = 'error'
GROUP BY bucket;


-- 16) "Realmente inobtenível": sem drop na aba Drops da própria carta E sem
--     aparecer em nenhum deck de duelista (mais completo que a query 6, que só
--     olhava a aba Drops da carta)
WITH via_card_drops AS (
    SELECT DISTINCT c.id
    FROM cards c
    WHERE json_array_length(c.fields->'drops') > 0
),
via_duelist_deck AS (
    SELECT DISTINCT (e.value->>'card_id')::INT AS card_id
    FROM duelists du, json_each(du.fields->'drops'->'deck') AS e
)
SELECT c.id, c.name
FROM cards c
WHERE c.id NOT IN (SELECT id FROM via_card_drops)
  AND c.id NOT IN (SELECT card_id FROM via_duelist_deck)
ORDER BY c.id;


-- 17) Amostra de cartas com type/attribute NULL -- provavelmente Spell/Trap,
--     cuja tabela Overview tem um layout diferente do de monstros.
--     Rode e me mande o overview_raw de 2-3 exemplos para eu ajustar o parser.
SELECT id, name, fields->'overview_raw' AS overview_raw
FROM cards
WHERE fields->>'type' IS NULL
LIMIT 3;


SELECT
    CASE
        WHEN (fields->>'is_monster')::BOOLEAN THEN fields->>'type'  -- Dragon, Warrior, etc.
        ELSE fields->>'type'  -- Equip, Magic, Ritual, Trap
    END AS category,
    (fields->>'is_monster')::BOOLEAN AS is_monster,
    COUNT(*) AS n
FROM cards
GROUP BY category, is_monster
ORDER BY is_monster DESC, n DESC;
