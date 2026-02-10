# Aula 7: Tabelas Ponte (Bridge Tables)

## 🎯 Objetivos
- Entender quando usar bridge tables.
- Resolver relacionamentos *Many-to-Many* (N:N) em Modelagem Dimensional.
- Implementar pesos e alocações para evitar dupla contagem.

---

## ❌ O Problema: Muitos-para-Muitos
No Star Schema clássico, o fato tem uma FK para a dimensão (1:N). Mas e se:
- Um **Produto** pertence a múltiplas **Categorias** simultaneamente?
- Uma **Conta Bancária** tem múltiplos **Titulares**?
- Um **Paciente** tem múltiplos **Diagnósticos** em uma consulta?

*Não podemos colocar múltiplas Foreign Keys na mesma coluna do fato!*

---

## 🌉 A Solução: Bridge Table
Uma tabela intermediária que fica entre a Dimensão e a Fato (ou entre duas Dimensões).

### Estrutura Sugerida:
- FK para a Dimensão A.
- FK para a Dimensão B.
- **Peso de Alocação (%):** Define quanto de cada métrica pertence a cada registro (essencial para que a soma total feche em 100%).

---

## ⚖️ Evitando a Dupla Contagem
Se um Notebook de R$ 3.000 pertence às categorias "Informática" e "Eletro", ao somar por categoria sem pesos, o total seria R$ 6.000 (errado!).

- **Solução:** Atribuir 50% de peso para cada.
- **Query:** `SUM(valor_venda * peso_alocacao)`.

---

## 🛠️ Exemplo: Conta Conjunta
```sql
CREATE TABLE bridge_conta_titular (
    conta_id INTEGER REFERENCES dim_conta(conta_id),
    cliente_id INTEGER REFERENCES dim_cliente(cliente_id),
    peso_alocacao DECIMAL(5,4) -- Ex: 0.5 para cada titular
);
```

---

## 🏁 Fechamento
- Bridge tables resolvem flexibilidade, mas aumentam a complexidade.
- Sempre verifique se a soma dos pesos por grupo é igual a 1.0.
- **Preview:** Na próxima aula, vamos aprender a lidar com mudanças históricas com SCD!
