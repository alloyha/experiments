# Aula 9: SCD Type 2 - Implementação Prática

## 🎯 Objetivos
- Implementar o fluxo completo de SCD Type 2 no PostgreSQL.
- Aprender a lógica de detecção de mudanças e versionamento.
- Realizar consultas históricas (*Point-in-Time*).

---

## 🏗️ Estrutura Recomendada
Uma dimensão SCD Type 2 profissional deve conter:
- **cliente_sk (Surrogate Key):** Identificador único daquela versão.
- **cliente_id (Natural Key):** Identificador único do cliente no mundo real.
- **Datas de Validade:** `data_inicio` e `data_fim`.
- **Controle de Versão:** `versao` (1, 2, 3...) e `registro_ativo`.

---

## 🔄 O Processo de Carga (ETL)
Para cada registro novo no sistema de origem:
1.  **Comparar:** Os atributos que trackamos mudaram?
2.  **Fechar:** Se sim, atualizamos o registro antigo (`ativo = FALSE`, `data_fim = ONTEM`).
3.  **Abrir:** Inserimos uma nova linha com os dados atuais (`ativo = TRUE`, `data_fim = 9999-12-31`).

*Dica: Use `INSERT ... ON CONFLICT` ou `WITH` clauses para tornar o processo atômico.*

---

## 🔍 Consultando o Passado
Para saber qual era o estado de um cliente em uma data específica (ex: 15/03/2023):
```sql
SELECT * 
FROM dim_cliente 
WHERE cliente_id = 101 
  AND '2023-03-15' BETWEEN data_inicio AND data_fim;
```

---

## 💡 Boas Práticas
- **Índices:** Crie índices nos campos `natural_key` e `registro_ativo`.
- **NULLs:** Use `COALESCE` ao comparar campos, pois `NULL != NULL` retorna Falso no SQL.
- **Híbrido:** Você pode ter colunas Type 1 e Type 2 na mesma tabela.

---

## 🏁 Fechamento
- SCD Type 2 garante a rastreabilidade dos seus indicadores.
- Sempre conecte as tabelas fato através da **Surrogate Key** (SK) para garantir que o fato aponte para a versão correta da dimensão no momento da venda.
- **Preview:** E para fechar o curso, vamos ver Modelagem de Grafos!
