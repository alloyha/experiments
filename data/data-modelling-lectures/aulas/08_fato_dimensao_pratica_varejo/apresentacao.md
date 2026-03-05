# Aula 8: Fato e Dimensão na Prática (Caso Varejo + SCD)

## 🎯 Objetivos
- Consolidar a relação Fato x Dimensão usando o **Dataset de Varejo**.
- Aprofundar o conceito de SCD abordado na Aula 6 focando no caso prático de Clientes e Produtos.
- Analisar as diferenças práticas entre "As-Is" e "As-Was" em relatórios num contexto real de negócio.
- Implementar as abordagens clássicas (Kimball) e modernas (Big Data/Nested).

---

## 🛠️ Revisitando: Impacto nas Queries
Na Aula 6, conhecemos a taxonomia. Mas como eles afetam os relatórios no dia a dia?

### Type 1 (Sobrescrever): Análise "As-Is" (Como está agora)
Ao preencher a tabela usando Type 1, toda e qualquer métrica do passado será agrupada mostrando o estado **atual**.
- **Impacto:** O histórico fica reescrito. Se a meta para a "Região Sul" foi batida em 2022, mas a loja mudou de região em 2023, o relatório de 2022 parecerá que a meta foi batida no Norte!
- **Uso Crítico:** Quando o negócio precisa ver "quem os clientes SÃO hoje".

### Type 2 (Histórico Completo): Análise "As-Was" (Como era na época)
Permite saber o exato estado da dimensão no instante em que o fato ocorreu (o "Padrão Ouro").
- **Impacto:** A chave estrangeira na Tabela Fato precisa necessariamente apontar para a `Surrogate Key` correspondente àquele período, e não para o ID original (`Natural Key`).
- **Uso Crítico:** Transparência em análises temporais e auditorias.

### Type 3 (Histórico Parcial): Comparação Imediata
Como mantém o `estado_atual` e `estado_anterior` na mesma linha.
- **Impacto:** Queries de transição ficam simples (Ex: `WHERE segmento_atual != segmento_anterior`) sem a necessidade de joins complexos da dimensão com ela mesma num formato Type 2.

### Type 4: Tabela de Estado Atual + Histórico
Usa uma tabela (acessada rapidamente) para o estado mais recente e uma tabela separada para a trilha de auditoria completa.
- **Uso:** Mantém as queries na tabela do dia a dia enxutas sem perder os deltas históricos na retaguarda.

### Type 5: Mini-Dimension
Estratégia híbrida onde os dados demográficos que mudam muito rápido e sem previsibilidade (ex: Idade, Score Serasa, Banda de Renda) são abstraídos para uma "Mini-Dimensão" desconectada e as Foreign Keys da mini-dimensão são colocadas direto no próprio Fato.

### Type 6: O Híbrido (1+2+3)
Você gera a linha como no Tipo 2 (`data_inicio`, `data_fim`), mas se cria uma coluna "Atributo Atual" na dimensão onde todos os registros antigos daquele indivíduo são atualizados ("Tipo 1") para o valor atual ("Tipo 3"). Facilita agregações atuais sem ter que ignorar o passado.

### Type 7: Dupla Chave (SK/NK)
Estratégia onde a Tabela Fato ganha duas Foreign Keys para a mesma Dimensão (Ex: `cliente_sk` e `cliente_nk`). A `cliente_sk` traz como cliente era na hora do fato (Tipo 2). A `cliente_nk` é unida para trazer como o cliente é hoje (Tipo 1).

---

## 🏗️ Implementando SCD Type 2 Avançado (JSONB + Diffs)
O SCD Type 2 é o padrão ouro. Para evoluir a resiliência a esquemas, em bancos modernos como PostgreSQL, agrupamos as características de forma flexível em propriedades JSON, além de registramos ativamente as diferenças (deltas) entre as versões, provando valor imenso para auditoria. A dimensão profissional passa a ser:
- **Surrogate Key (ex: `cliente_sk`):** Identificador único daquela versão.
- **Natural Key:** Identificador único da origem.
- **`properties` (JSONB):** Dados consolidados daquele instante (ex: `{"estado": "SP", "segmento": "Bronze"}`).
- **`properties_diff` (JSONB):** Registra exatamente as colunas modificadas em relação à versão passada (ex: `{"estado": {"from": "SP", "to": "PR"}}`). Isso responde *imediatamente* a pergunta analítica: "O que justificou a alteração desta linha de histórico?".
- **Datas de Validade e Controle:** `data_inicio`, `data_fim`, `versao` e `ativo`.

### O Processo de Carga e Automação (ETL)
Toda Procedure é escrita como **Incremental por natureza**:
- Ela identifica apenas os **deltas** (registros novos ou que mudaram) através de um `LEFT JOIN + IS DISTINCT FROM` entre Origem e Destino.
- Linhas iguais = **zero I/O**. Não tocamos no que não mudou.
- Usa `ON CONFLICT DO UPDATE` para garantir **Idempotência** (reprocessar o mesmo dia não gera duplicata).
- Datas indefinidas usam `data_fim = NULL` (agnóstico de engine).

Com essa premissa, **Backfill vira um Fast-Forward de chamadas incrementais**:
```sql
-- Backfill = Loop de chamadas da MESMA procedure incremental
FOR dt IN generate_series('2023-01-01', '2023-12-31', '1 day') LOOP
    CALL varejo.proc_upsert_clientes_scd2(dt);
END LOOP;
```
Cada iteração processa SOMENTEo delta daquele dia. Não existe uma "Procedure de Backfill" separada — o mesmo código serve para ambos os modos.

*(Nota: Em Data Lakes Spark/Delta, esse mesmo princípio se traduz em fluxos Append-Only ou Merge incrementais por partição.)*

### Consultando o Passado (Point-in-Time)
Para saber o estado em 15/03/2023:
```sql
SELECT * 
FROM varejo.dim_cliente_scd2 
WHERE cliente_id = 101 
  AND '2023-03-15' >= data_inicio
  AND ('2023-03-15' <= data_fim OR data_fim IS NULL);
```

---

## 🏁 Fechamento
- SCD Type 2 garante a rastreabilidade dos indicadores.
- Escolha o tipo de SCD baseado na importância do histórico.
- **Preview:** Na próxima aula, vamos explorar Modelagem de Grafos!
