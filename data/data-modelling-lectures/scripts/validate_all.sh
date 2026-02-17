#!/bin/bash
set -e

# Detecta o diretório raiz do projeto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configurações
CONTAINER_NAME="curso_modelagem_postgres"
DB_USER="aluno"
DB_NAME="curso_modelagem"
LOG_FILE="$PROJECT_ROOT/validation_log.txt"
export PGPASSWORD="modelagem_password"

echo "==============================================="
echo "   VALIDADOR PARALELIZADO POR DOMÍNIOS"
echo "   Logs detalhados em: $LOG_FILE"
echo "==============================================="

# Inicializa o arquivo de log
echo "=== INÍCIO DA VALIDAÇÃO: $(date) ===" > "$LOG_FILE"

# Função para executar arquivos de uma aula de forma sequencial (dentro do domínio)
process_aula() {
    local aula_path=$1
    local target_schema=$2
    
    echo "  📂 Aula: $(basename "$aula_path")"
    
    # Encontra scripts na aula (respeitando ordem alfabética)
    scripts=$(find "$aula_path" -maxdepth 1 -type f \( -name "*.sql" -o -name "*.md" \) | sort)
    
    for script in $scripts; do
        if [[ "$script" == *"exercicios.md"* ]]; then continue; fi
        
        local cmd
        if [[ "$script" == *.md ]]; then
            cmd="sed -n '/^\`\`\`sql$/,/^\`\`\`$/ p' \"$script\" | sed '/^\`\`\`/d'"
        else
            cmd="cat \"$script\""
        fi

        # Execução com isolamento de schema
        if { echo "SET search_path TO $target_schema, public;"; eval "$cmd"; } | docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 -q >> "$LOG_FILE" 2>&1; then
            echo "    ✅ $(basename "$script")"
        else
            echo "    ❌ $(basename "$script") (Falhou)"
            return 1
        fi
    done
}

# Função para processar um domínio inteiro (Grupo de Aulas)
process_dominio() {
    local nome=$1
    local schema=$2
    shift 2
    local aulas=("$@")

    echo "🚀 Iniciando Domínio: $nome (Schema: $schema)"
    
    for aula in "${aulas[@]}"; do
        # Encontra o caminho real da aula
        local path=$(find "$PROJECT_ROOT/aulas" -maxdepth 1 -name "*$aula*" -type d)
        if [ -d "$path" ]; then
            process_aula "$path" "$schema" || return 1
        fi
    done
    echo "🏁 Domínio $nome Concluído!"
}

# 1. Reset Inicial
echo ">> [1/2] Resetando Banco de Dados..."
{
    grep -v '^\\i' "$SCRIPT_DIR/setup_database.sql"
    cat "$SCRIPT_DIR/seed_biblioteca.sql"
    cat "$SCRIPT_DIR/seed_varejo.sql"
    cat "$SCRIPT_DIR/seed_rede_social.sql"
} | docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 >> "$LOG_FILE" 2>&1

# 2. Execução Paralela
echo ""
echo ">> [2/2] Disparando Background Workers por Domínio..."
echo ""

# Dispara os 3 domínios em background
process_dominio "Biblioteca" "biblioteca" "01_introducao" "02_erd_conceitos" "03_erd_pratica_biblioteca" &
PID_BIB=$!
sleep 1

process_dominio "Varejo" "varejo" "04_intro_dimensional" "05_tabelas_fato" "06_dimensoes" "07_bridge_tables" "08_scd_types" "09_scd_type2_implementacao" &
PID_VAR=$!
sleep 1

process_dominio "Grafos" "rede_social" "10_modelagem_grafos" &
PID_GRA=$!

# Aguarda todos terminarem
wait $PID_BIB || ERROR=1
wait $PID_VAR || ERROR=1
wait $PID_GRA || ERROR=1

echo ""
if [ "$ERROR" ]; then
    echo "❌ Validação concluída com erros. Veja $LOG_FILE"
    exit 1
else
    echo "✅ SUCESSO! Todas as aulas de todos os domínios foram validadas."
fi
