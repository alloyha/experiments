#!/bin/bash
set -e

# Nome do Container
CONTAINER_NAME="curso_modelagem_postgres"
DB_USER="aluno"
DB_NAME="curso_modelagem"
# A senha deve ser passada para o ambiente do container
export PGPASSWORD="modelagem_password"

echo "==========================================="
echo "   VALIDADOR DE SCRIPTS SQL - INICIANDO"
echo "   (Executando via Docker)"
echo "==========================================="
echo ""

# Função para executar um arquivo SQL via Docker
run_sql_file() {
    local file=$1
    echo -n "Testando $file... "
    
    # Executa o SQL via docker exec. 
    # -i: permite input via stdin (pipe)
    # Passamos PGPASSWORD como variavel de ambiente para o comando dentro do container
    # -q (quiet) esconde o ruído de INSERT/CREATE
    # client_min_messages=notice permite ver nossas mensagens de sucesso (RAISE NOTICE)
    if cat "$file" | docker exec -e PGPASSWORD=$PGPASSWORD -e PGOPTIONS="-c client_min_messages=notice" -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 -q; then
        echo "✅ OK"
    else
        echo "❌ FALHOU"
        echo "-------------------------------------------"
        echo "Erro ao executar $file"
        echo "-------------------------------------------"
    fi
}

# 1. Resetar o Banco de Dados (Setup Inicial)
echo ">> [1/3] Resetando Banco de Dados (Carga Limpa)..."
# Executa com output visível para debug
cat "setup_database.sql" | docker exec -e PGPASSWORD=$PGPASSWORD -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1

# 2. Encontrar e Executar Scripts de Aula e Gabaritos
echo ""
echo ">> [2/3] Executando Scripts de Aula e Gabaritos..."

# Busca arquivos apresentacao.sql e gabarito.sql ordenados
# Exclui pastas ocultas ou arquivos temporários
find . -type f \( -name "apresentacao.sql" -o -name "gabarito*.sql" \) | sort | while read script; do
    # Pula arquivos dentro de .git ou outros diretorios ocultos se houver
    if [[ "$script" == *"/."* ]]; then continue; fi
    
    run_sql_file "$script"
done

echo ""
echo "==========================================="
echo "   TODOS OS SCRIPTS FORAM VALIDADOS! 🎉"
echo "==========================================="
