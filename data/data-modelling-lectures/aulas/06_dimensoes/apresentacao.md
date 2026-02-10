# Aula 6: Tabelas Dimensão

## 🎯 Objetivos
- Entender a anatomia e o propósito das dimensões.
- Aprender sobre Hierarquias e Dimensões Especiais.
- Implementar dimensões robustas no PostgreSQL.

---

## 📦 Anatomia de uma Dimensão
Diferente das tabelas operacionais, dimensões são largas e desnormalizadas.

- **Surrogate Key (PK):** ID gerado internamente (SERIAL). Protege o DW de mudanças no sistema origem.
- **Natural Key:** O ID original do sistema operacional (ex: Código do Produto).
- **Atributos:** Textos descritivos usados para filtrar e agrupar dados.

---

## ⏳ Dimensão Tempo (Calendário)
A dimensão mais importante de qualquer DW. Nunca use `EXTRACT` em tempo de execução se puder ter uma tabela pré-calculada.

- **Vantagem:** Permite filtros complexos como "Finais de Semana", "Feriados Móveis" ou "Aniversário da Loja" de forma instantânea.

---

## 🎭 Dimensões Especiais
1. **Degenerada:** Atributo que vive no fato (ex: Número do Pedido) porque não tem outros atributos próprios.
2. **Role-Playing:** Uma única dimensão usada para múltiplos papéis (ex: Dim Tempo servindo como Data do Pedido E Data da Entrega).
3. **Junk Dimension:** Agrupa flags (SIM/NÃO) e pequenos indicadores para limpar a tabela fato.
4. **Conformada:** Dimensão idêntica compartilhada por múltiplos fatos (ex: Mesma Dim Cliente para Vendas e Suporte).

---

## 🛠️ Exemplo de Registro Especial
Sempre inclua registros para tratar dados faltantes ou nulos:
```sql
INSERT INTO dim_produto (produto_id, nome, categoria) 
VALUES (-1, 'NÃO INFORMADO', 'N/A');
```
*Evite NULLs nas Foreign Keys do Fato!*

---

## 🏁 Fechamento
- Dimensões dão contexto aos números.
- Surrogate keys são obrigatórias para um DW profissional.
- **Preview:** Na próxima aula, vamos resolver casos complexos com Bridge Tables!
