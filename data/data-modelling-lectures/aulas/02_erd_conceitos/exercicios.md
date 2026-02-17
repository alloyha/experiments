# EXERCÍCIOS AULA 02: ERD FUNDAMENTOS & TIPOS DE DADOS

## EXERCÍCIO 1: Entidades (Conceitual)

Em um sistema de vendas, uma entidade 'Pedido' contém as seguintes informações:
- `Cliente_ID`, `Data_Pedido`, `Valor_Total`, `Itens_Do_Pedido`, `Cidade_Entrega`.

**a)** Quais atributos deveriam ser segregados para novas entidades em um modelo de banco bem projetado (1NF/2NF)?

**b)** Por que a Cidade de Entrega não deve ficar repetindo em cada pedido?

## EXERCÍCIO 2: Relacionamentos (N:N vs 1:N)

**a)** Um Autor pode escrever vários Livros, e um Livro pode ter vários Autores. Como modelar isso?

**b)** Um Departamento tem vários Funcionários, mas um Funcionário pertence a apenas um Departamento. Como modelar isso?

## EXERCÍCIO 3: Cardinalidade e Chaves

**a)** Identifique a Chave Primária (PK) e a Chave Estrangeira (FK) na tabela abaixo:
> Tabela 'Emprestimo': (`id`, `data`, `usuario_id`, `livro_id`)

**b)** O campo `usuario_id` pode se repetir nesta tabela? Por quê?
