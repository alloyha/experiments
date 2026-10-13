# Gate único de liberação do cache extraído

## Escopo

Este PR substitui a autorização implícita para poda por um fluxo **planejar → certificar canary → aplicar em janela sem consumidores → recuperar após falha**. Ele atua **somente nos objetos `raw/**/extracted/`** de versões ativas que comprovadamente derivam dos ZIPs locais. Nunca apaga `source/`, Parquet, bancos DuckDB, manifesto ou versões históricas.

O `make prune-cache APPLY=1` passa a ser **bloqueado**, evitando a via usual de poda não certificada. A interface legada `scripts/prune_extracted_raw.py --apply` continua existindo para compatibilidade com as certificações PR42–45 que a invocam em scratch; **não a execute contra o lake principal**. O bloqueio no Makefile não constitui uma barreira de segurança para chamadas diretas ao script legado.

## Garantias e limitações

- PR42 verifica fonte ZIP, SHA-256, tamanho, manifesto, membros selecionados, caminho e índice ativo. O novo gate confere, adicionalmente, por streaming, **SHA-256 do conteúdo de cada membro ZIP contra o CSV extraído**, verificando também a integridade CRC do ZIP. Recursos sem prova são recusados, não eliminados.
- O planejamento e o canary **não escrevem metadados nem removem arquivos no lake principal**. O bloqueio cooperativo do novo gate é armazenado em `/tmp`.
- Cada aplicação usa plano amarrado ao SHA-256 dos metadados, limites explícitos, canary aprovado para aquele domínio/ano e hash do código do gate. **Uma prova canary pode ser reutilizada por lotes do mesmo domínio/ano**; cada novo plano verifica independentemente seus próprios ZIPs e CSVs.
- `--apply` exige `--ack-quiescent`: o operador garante que TODOS os leitores dbt/DuckDB, ingesters, rehydrators e escritores externos estão PARADOS durante a janela. O `flock` do gate protege apenas invocações cooperativas deste programa. Não existe transação ACID entre banco, arquivos e consumidores que ignorem essa janela.
- Após validação, a operação registra um **diário durável**. Reinicialização com `--recover --ack-quiescent` continua para frente após falha. Divergência inesperada do estado/índice, fonte ou alvo impede a recuperação automática. Não se promete rollback instantâneo, e um diário pendente deve ser resolvido antes de qualquer novo plano.
- Não prova equivalência completa do warehouse nem libera fontes imutáveis para exclusão. Conclusões do PR43–45 permanecem limitadas aos subgrafos ali testados.

## Primeiro canary — apenas scratch, sem exclusão canônica

```bash
cd ~/github/experiments-cache-release/data/tse-election-data
RAW="$HOME/github/experiments/data/tse-election-data/data/tse"
SCRIPT="scripts/safe_cache_evict.py"

python3 "$SCRIPT" --root "$RAW" --plan \
  --domain candidate_assets --year 2026 \
  --max-resources 1 --max-bytes 33554432 \
  --output-json /tmp/tse-cache-plan.json

python3 "$SCRIPT" --root "$RAW" --canary \
  --plan-file /tmp/tse-cache-plan.json \
  --output-json /tmp/tse-cache-canary.json

python3 -m json.tool /tmp/tse-cache-canary.json
```

Somente depois de obter `result: PASS`, com fontes verificadas e o relatório de auditoria estrita sem questões, considere uma **janela de manutenção real**. Nunca aplique o `--ack-quiescent` enquanto houver consumidores ativos.

## Primeiro lote opcional — exclusão real, deliberada

```bash
python3 scripts/safe_cache_evict.py --root "$RAW" --apply \
  --plan-file /tmp/tse-cache-plan.json \
  --canary-report /tmp/tse-cache-canary.json \
  --ack-quiescent
```

A exclusão invalida o índice físico dessas materializações, marca suas versões ativas como `evicted` e mantém os ZIPs. **Antes de executar consultas dos domínios afetados, execute** `make materialize-raw DATA_ROOT="$RAW" YEARS=2026 MATERIALIZE_DOMAINS='candidate_assets' MATERIALIZE_EXTRACTOR=python` (após encerrar a janela de evicção; a operação de reidratação também deve ser coordenada com consumidores), ou aceite que essas consultas falhem enquanto o cache estiver evictado. Essa ação deliberadamente repõe os CSVs e, por isso, reocupa o espaço liberado.

Para um novo lote, gere outro plano com `--max-resources` e `--max-bytes` maiores e **reutilize** `/tmp/tse-cache-canary.json` se todos os domínios/anos forem os mesmos e o código do gate não tiver mudado. Se houver diário pendente após morte/crash:

```bash
python3 scripts/safe_cache_evict.py --root "$RAW" --recover --ack-quiescent
```

## Critérios objetivos de encerramento

`PASS` exige: fonte e manifesto íntegros, ZIP→CSV byte a byte, canary offline reconstruído com índice correto, nenhuma modificação de fonte na janela, publicação durável de `evicted`/índice sem referências e journal limpo. Não há previsão de ganho fixo de 39 GiB: o plano mede e limita **bytes efetivamente elimináveis**. Quando consumidores exigirem novos CSVs, a reidratação ocupará espaço novamente.
