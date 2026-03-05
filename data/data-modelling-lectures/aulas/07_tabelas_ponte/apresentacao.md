# Aula 7: Tabelas Ponte (Bridge Tables)

## 🎯 Objetivos
- Entender quando usar bridge tables.
- Resolver relacionamentos *Many-to-Many* (N:N) em Modelagem Dimensional.
- Implementar pesos e alocações para evitar dupla contagem.

---

## ❌ O Problema: Muitos-para-Muitos
No Star Schema clássico, um registro na Fato aponta para um único registro na Dimensão (relacionamento 1:1 por linha da fato). Mas e se:
- Uma **Consulta Médica (Fato)** envolver múltiplos **Diagnósticos (Dimensão)**?
- Uma **Conta Bancária (Dimensão)** tiver múltiplos **Titulares (Dimensão)**?

*A limitação não é apenas "não poder ter múltiplas Foreign Keys na mesma coluna". Se simplesmente **duplicarmos a linha da Fato** para cada diagnóstico ou titular, estaremos multiplicando o valor das métricas (valores financeiros, contagens) gerando a grave **dupla contagem**!*

---

## 🌉 A Solução: Bridge Table
Uma tabela intermediária que resolve a cardinalidade N:N e protege a integridade das métricas do fato. Ocorre de duas formas em modelagem dimensional:

### 1. Dimensão Multi-valorada (Ex: Fato Consulta ↔ Bridge ↔ Dim Diagnóstico)
- A Tabela Fato ganha uma **Chave de Grupo** (`grupo_diagnostico_key`).
- A Bridge mapeia a `grupo_diagnostico_key` para múltiplas FKs da Dimensão de Diagnóstico.

### 2. Bridge entre Dimensões (Ex: Fato Transação ↔ Dim Conta ↔ Bridge ↔ Dim Cliente)
- A Fato aponta para a Dimensão Principal (`Dim_Conta`).
- A Bridge conecta a `FK_Conta` a múltiplas `FK_Cliente`.

**Elemento Adicional (Opcional):**
- **Peso de Alocação (Weighting Factor %):** O uso de pesos **não é obrigatório**, é apenas uma possibilidade. Depende do caso de uso de negócio.

---

## ⚖️ Rateio: Evitando a Dupla Contagem
*Quando seu caso de uso envolve somar métricas aditivas (faturamento, contagem, transações), o uso de pesos torna-se essencial.* Se um faturamento de R$ 3.000 for atribuído simultaneamente para as categorias "Informática" e "Eletro", um analista fazendo SUM() por categoria achará a soma final de R$ 6.000.

- **Solução:** Atribuir 50% de peso para cada "linha" gerada no relacionamento (uma decisão de modelagem).
- **Query:** Em vez de `SUM(valor_venda)`, usa-se `SUM(valor_venda * peso_alocacao)`.

---

## 🛠️ Exemplos Estruturais

### 1. Bridge Fato ↔ Dimensão (Multi-valorada)
```sql
-- Fato Consulta contém a coluna: grupo_diagnostico_key
CREATE TABLE bridge_grupo_diagnostico (
    grupo_diagnostico_key INTEGER,
    diagnostico_id INTEGER REFERENCES dim_diagnostico(diagnostico_id),
    peso_alocacao DECIMAL(5,4) -- Opcional, se houver regra de rateio
);
```

### 2. Bridge Dimensão ↔ Dimensão
```sql
-- Fato Transação contém a FK normal para a Conta Bancária
CREATE TABLE bridge_conta_titular (
    conta_id INTEGER REFERENCES dim_conta(conta_id),
    cliente_id INTEGER REFERENCES dim_cliente(cliente_id),
    peso_alocacao DECIMAL(5,4) -- Opcional (ex: dividir métricas da transação por titular)
);
```
---

## 🏁 Fechamento
- Bridge tables resolvem flexibilidade, mas aumentam a complexidade.
- Sempre verifique se a soma dos pesos por grupo é igual a 1.0.
- **Preview:** Na próxima aula, vamos aprender a lidar com mudanças históricas com SCD!
