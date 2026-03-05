# Aula 5: Tabelas Fato

## 🎯 Objetivos
- Entender a anatomia e granularidade das tabelas fato.
- Conhecer os tipos de métricas (Aditivas, Semi-aditivas, Não-aditivas).
- Diferenciar os 3 tipos principais de tabelas fato.

---

## 📐 Granularidade
Define o que exatamente uma linha da tabela fato representa.

- **Regra de Ouro:** Sempre comece com a granularidade mais fina possível (ex: cada item de um cupom fiscal).
- **Por quê?** Você sempre pode somar dados detalhados, mas nunca pode desdobrar dados agregados.

---

## ➕ Tipos de Métricas
1. **Aditivas:** Podem ser somadas em todas as dimensões (ex: Venda Total).
2. **Semi-aditivas:** Podem ser somadas em algumas dimensões, mas não no tempo (ex: Saldo de Estoque).
3. **Não-aditivas:** Não podem ser somadas (ex: Taxa de Conversão %). Devem ser recalculadas após a soma dos componentes.

---

## 📁 Tipos de Tabelas Fato
1. **Fato Transacional:** Registra um evento no exato momento em que ocorre (ex: Venda, Clique). Nunca para de crescer.
2. **Snapshot Periódico:** Registra o estado das coisas em intervalos regulares (ex: Saldo de estoque no fim de cada dia). Tamanho previsível.
3. **Snapshot acumulativo:** Registra o progresso de um processo com início, meio e fim (ex: Pipeline de um pedido desde a compra até a entrega). A linha é atualizada.

---

## 🛠️ Exemplo de Implementação
```sql
CREATE TABLE fato_vendas (
    venda_id SERIAL PRIMARY KEY,
    tempo_id INTEGER REFERENCES dim_tempo(tempo_id),
    cliente_id INTEGER REFERENCES dim_cliente(cliente_id),
    valor_total DECIMAL(10,2), -- Métrica Aditiva
    numero_nota_fiscal VARCHAR(20) -- Dimensão Degenerada
);
```

---

## 🏁 Fechamento
- Defina a granularidade antes de qualquer coisa.
- Cuidado ao somar métricas semi-aditivas (use médias ou o último valor).
- **Preview:** Na próxima aula, vamos explorar as Tabelas Dimensão!
