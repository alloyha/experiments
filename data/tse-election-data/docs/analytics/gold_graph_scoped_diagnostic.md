# Gold Graph — diagnóstico físico por ciclo

O filtro `WHERE cycle_id = '2024:municipal'` nas views globais de
`graph_vertices` e `graph_edges` **não prova** pruning do `READ_PARQUET`.
No EXPLAIN do canário, o scan candidato foi estimado em ~26,5 milhões de
linhas; isso não demonstra quantos Parquets serão efetivamente lidos.

`certify_gold_graph_cycle.py` usa o SQL original dos modelos e das oito
asserções do grafo, mas substitui suas fontes por views temporárias, em memória,
filtradas para um único ciclo. O fato de votos nominais é lido de um **único**
`data.parquet` publicado (caminho físico explícito, sem wildcard). As dimensões
Gold e os fatos partidários são obtidos do DuckDB com `ATTACH ... READ_ONLY`.
Nenhuma tabela/view é criada no warehouse anexado.

```bash
MAIN="$HOME/github/experiments/data/tse-election-data"
PY="$MAIN/.venv/bin/python"
CANARY="$MAIN/data/warehouse/tse_event_registry_preview.duckdb"
FACTS="$MAIN/data/warehouse/fact_candidate_votes"

# Somente planejamento; nenhuma asserção histórica executada.
"$PY" scripts/certify_gold_graph_cycle.py \
  --db "$CANARY" --source-root "$FACTS" --cycle 2024:municipal

# Executar apenas depois do diagnóstico inicial e com timeout externo.
timeout --signal=INT --kill-after=30s 180s \
  "$PY" scripts/certify_gold_graph_cycle.py \
    --db "$CANARY" --source-root "$FACTS" --cycle 2024:municipal \
    --memory-limit 4GB --execute \
    --tests candidate_vote_conservation party_vote_conservation
```

Isto é um **diagnóstico**, não atestação criptográfica de freshness, snapshot
bloqueado, publicação certificada ou interface segura de consumo. Fontes e
dimensões podem mudar concorrentemente. As views criadas por dbt continuam
globais e essa ferramenta não prova pruning no caminho de consulta global.
O registro de invariantes `--strict` permanece bloqueado até a implementação
das lacunas críticas de promoção.
