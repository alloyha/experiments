# PR43 — Certificação downstream dbt após recuperação offline

## Escopo científico e operacional

PR42 certificou **bytes de CSV e leitura DuckDB** para o recurso `candidate_history/2026`. Esse domínio não alimenta o grafo dbt selecionado. O PR43 introduz a primeira certificação **com execução real de modelos dbt**, para a cadeia existente no repositório:

`candidate_assets → bronze_candidate_assets → silver_candidate_assets`

Uma única fonte `candidate_assets` ativa (por padrão, a menor que atende aos limites) é copiada para um lake temporário independente, com índice e estado mínimos válidos. **Não se afirma equivalência de `dim_candidate`, do warehouse inteiro ou das demais organizações.**

## Procedimento implementado

1. Selecionar um recurso completo com ZIP, SHA-256 e registro no manifesto, verificando seus arquivos no índice físico.
2. Preservar hashes dos metadados e arquivos do lake canônico antes de iniciar.
3. Copiar a fonte e sua materialização para um `TemporaryDirectory`; copiar o projeto dbt e dependências instaladas para outra árvore temporária; escrever um `profiles.yml` novo, sem `httpfs` e com caminhos DuckDB temporários.
4. Executar **`dbt run --select +silver_candidate_assets`** na primeira base DuckDB isolada; capturar esquema, número de linhas e SHA-256 de multiconjunto de linhas para `bronze_candidate_assets` e `silver_candidate_assets`. Ordem não altera o hash, mas duplicações alteram.
5. Executar **o podador oficial** com `--apply` **somente** no lake descartável; comprovar que o estado está `evicted`, que o índice foi esvaziado e que o cache deixou de existir.
6. Executar **o recuperador oficial** `ensure_materialized.py` com `--extractor python`, com bloqueio Python de conexões de rede herdado do PR42.
7. Verificar SHA-256 da fonte e materializações, estado `present`, índice reconstruído e manifesto preservado.
8. Executar a mesma cadeia de modelos numa **segunda base DuckDB independente**, com novas configurações temporárias; comparar integralmente os esquemas, cardinalidades e hashes das linhas.
9. Verificar os contratos físicos por `report_local_storage.py --verify-hashes --strict`; revalidar hashes dos arquivos canônicos; destruir o scratch mesmo em caso de falha.

**Importante:** o bloqueio de rede atua nos sockets Python usados pelos processos filhos e o perfil evita carregar `httpfs`. Não é uma sandbox de rede em nível de kernel para bibliotecas nativas. O processo não chama download de fontes ou `dbt deps`; exige dependências dbt já instaladas. Os dois bancos são criados somente em scratch, não há leitura/escrita de bancos persistidos de produção.

## Execução no WSL

No novo worktree após aplicar o pacote:

```bash
cd ~/github/experiments-dbt-roundtrip/data/tse-election-data
make unit
make medallion-contracts
make dbt-deps

RAW="$HOME/github/experiments/data/tse-election-data/data/tse"
make dbt-cache-roundtrip DATA_ROOT="$RAW" \
  DBT_ROUNDTRIP_YEAR=2026 \
  DBT_ROUNDTRIP_JSON=/tmp/tse-pr43-dbt-roundtrip.json
python3 -m json.tool /tmp/tse-pr43-dbt-roundtrip.json
```

Por padrão o runner escolhe a menor fonte `candidate_assets` completa dentro dos limites (256 MiB comprimidos, 512 MiB extraídos; 500 mil linhas por relação). Para selecionar manualmente um ID:

```bash
make dbt-cache-roundtrip DATA_ROOT="$RAW" \
  DBT_ROUNDTRIP_YEAR=2026 \
  DBT_ROUNDTRIP_RESOURCE_ID='<resource-id>' \
  DBT_ROUNDTRIP_JSON=/tmp/tse-pr43-dbt-roundtrip.json
```

Se os limites impedirem qualquer seleção, examine os recursos no `resource_state.json` e ajuste conscientemente `DBT_ROUNDTRIP_MAX_SOURCE_MIB`, `DBT_ROUNDTRIP_MAX_EXTRACTED_MIB` ou `DBT_ROUNDTRIP_MAX_ROWS`. NUNCA execute `prune-cache APPLY=1` no lake canônico para essa certificação.

## Aprovação e limitações

`result: PASS` requer uma execução dbt real bem-sucedida nos dois momentos, **resultado não vazio em ambos os modelos**, mesmo esquema e mesmo multiconjunto de linhas. A integridade do ZIP e CSV, o estado durável, o índice e o manifesto também precisam permanecer corretos; comparação de somatórios agregados isolados não é suficiente.

Há testes unitários dos invariantes e do empacotamento do comando dbt. A **execução do dbt real no WSL e seus resultados ainda precisam ser coletados**; não inferir PASS a partir dos testes sintéticos ou da certificação do PR42.

Depois deste slice: ampliar aos modelos Gold/Semantic e dependências múltiplas, cobrir falhas/retentativas e volumes grandes. A liberação dos ~39,2 GiB de cache da instalação principal não faz parte do PR43.

## Read-only preflight when selection fails

The candidate picker intentionally refuses to fabricate an eligible resource.
The reason may be an absent domain/year, a source that is not ZIP-backed, an
unmaterialized active version, or the copy/row budgets. Diagnose without
reading or mutating CSV/ZIP payloads:

```bash
.venv/bin/python scripts/certify_dbt_cache_roundtrip.py \
  --raw-root "$RAW" --year 2026 --inspect-candidates \
  --output-json /tmp/tse-pr43-candidates.json
python3 -m json.tool /tmp/tse-pr43-candidates.json
```

The JSON reports source/output sizes, source type, availability and exclusion
reasons for each `candidate_assets` resource in **all** years. Only run an
alternate-year physical roundtrip when the report identifies a compatible
`present` ZIP with extracted objects inside budgets. Do not increase limits
blindly and do not relabel a non-ZIP input as evictable cache.

## Legacy metadata compatibility (PR43 only)

The real lake may contain active versions written before `materialization_state`
was introduced. `selection_complete=true` plus an absent key does **not** certify
physical presence by itself. PR43 accepts an *absent key only* as a candidate,
then requires all of PR42's manifest publication, ZIP size + SHA-256, selected
members, safe paths, existing outputs and **exact current-index membership**.
PR42 itself stays strict unless its validator receives the explicit legacy
opt-in from PR43. Explicit `null`, `evicted`, or `not_required` remain rejected.
Canonical metadata are never backfilled by the certifier. The scratch copy
receives `evicted` and later `present` through the official maintenance tools.
The JSON evidence includes `legacy_materialization_state_missing` so the
successful certification cannot be mistaken for a schema migration.
The read-only inventory's "eligible" means metadata preflight only: mandatory
SHA-256 and manifest verification still run before any scratch mutation.


## Physical downstream certification — 2026-10-13

**Result: PASS** for the real `candidate_assets/2026` resource.

- Resource ID: `33fbda56-eb41-46f5-a8a0-8b499c285a1d`
- Source SHA-256: `f912568aa1275c9ce8a0d029780569aeef93745c76c318081ff7206a951b9277`
- Bronze rows: **77,297**
- Silver rows: **13,917**
- Before/after dbt execution: **2/2 PASS** each
- Schema, cardinality and row-multiset equivalence: **PASS**
- Offline rehydration, index reconstruction and strict audit: **PASS**
- Canonical source and control-plane preservation: **PASS**
- Legacy missing materialization-state field: explicitly validated
- Disposable scratch removed: **YES**

Evidence: `contracts/certification/pr43_candidate_assets_2026.json`.

This gate covers the selected Bronze/Silver graph only.
Gold, Semantic, complete warehouse equivalence, and global cache
eviction remain outside this certification.
