# CDC Stack - Índice de Documentação

Bem-vindo ao projeto **PostgreSQL → Debezium → Kafka → dbt-Flink → Iceberg**! 

Este é um guia completo para navegar pela documentação e recursos disponíveis.

## 📚 Documentação Completa

### 🚀 Para Começar (Escolha Um)

- **[GETTING_STARTED.md](GETTING_STARTED.md)** ← **Comece por aqui!**
  - 5 minutos para rodar tudo
  - Passo a passo simples
  - Verifica saúde dos serviços

- **[README.md](README.md)**
  - Documentação completa
  - Configuração detalhada
  - Todos os comandos disponíveis

### 📖 Entender a Arquitetura

- **[ARCHITECTURE.md](ARCHITECTURE.md)**
  - Diagrama visual completo
  - Fluxo de dados detalhado
  - Padrões de design
  - Características por camada
  - Monitoramento

### 💡 Exemplos Práticos

- **[EXAMPLES.md](EXAMPLES.md)** ← **Recomendado para aprender**
  - Exemplo 1: Fluxo completo (10 min)
  - Exemplo 2: Capturar Updates
  - Exemplo 3: Capturar Deletes
  - Exemplo 4: Modificar models dbt
  - Exemplo 5: Adicionar nova tabela
  - Exemplo 6: Criar Mart customizado
  - Exemplo 7: Testes dbt
  - Exemplo 8: Monitorar performance
  - Exemplo 9: Pausar/Resumir CDC
  - Exemplo 10: Reset completo
  - + Dicas importantes

### 🐛 Resolução de Problemas

- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**
  - Diagnóstico rápido
  - Problemas PostgreSQL
  - Problemas Kafka
  - Problemas Debezium
  - Problemas MinIO
  - Problemas Flink
  - Operações de manutenção
  - Performance tuning

## 🛠️ Usando a Stack

### Comandos Rápidos (Makefile)

```bash
make help              # Ver todos os comandos

# Ciclo de vida
make start             # Subir stack
make stop              # Parar stack
make health            # Verificar saúde
make clean             # Reset completo

# Gerenciar dados
make psql              # Conectar ao PostgreSQL
make insert-test-data  # Inserir dados
make kafka-topics      # Listar tópicos Kafka
make kafka-consume TOPIC=customers  # Ver eventos

# Debezium
make register-connector  # Registrar connector
make status-connector    # Ver status

# dbt-Flink
make dbt-run           # Executar models
make dbt-test          # Testar models
make dbt-docs          # Gerar documentação

# Logs
make logs              # Todos os serviços
make logs-postgres     # PostgreSQL
make logs-kafka        # Kafka
make logs-debezium     # Debezium
make logs-flink        # Flink
```

### Scripts Bash

```bash
./scripts/start-all.sh      # Subir tudo
./scripts/stop-all.sh       # Parar tudo
./scripts/health-check.sh   # Verificar saúde
./scripts/debezium_manager.py register  # Registrar connector
```

## 📁 Estrutura de Arquivos

```
postgres-debezium-kafka-flink-iceberg/
│
├── 📄 Documentação
│   ├── README.md                    ← Leia primeiro
│   ├── GETTING_STARTED.md           ← Para começar rapidinho
│   ├── ARCHITECTURE.md              ← Entender design
│   ├── EXAMPLES.md                  ← Aprender com exemplos
│   ├── TROUBLESHOOTING.md           ← Resolver problemas
│   ├── INDEX.md (este arquivo)      ← Navegação
│   └── Makefile                     ← Comandos
│
├── 🐳 Docker & Configuração
│   ├── docker-compose.yml           ← Orquestrador principal
│   ├── .env                         ← Variáveis de ambiente
│   └── docker-compose.override.yml.example
│
├── 📝 Scripts
│   ├── scripts/start-all.sh         ← Subir stack
│   ├── scripts/stop-all.sh          ← Parar stack
│   ├── scripts/health-check.sh      ← Verificar saúde
│   ├── scripts/init-postgres.sql    ← Inicializar DB
│   └── scripts/debezium_manager.py  ← Gerenciar connectors
│
├── ⚙️ Configurações
│   ├── config/debezium/postgres-source.json
│   ├── config/minio/init-buckets.sh
│   └── config/dbt-flink/flink-conf.yaml
│
├── 📊 dbt-Flink Project
│   ├── dbt/dbt_project.yml
│   ├── dbt/profiles.yml
│   ├── dbt/models/staging/stg_customers.sql
│   ├── dbt/models/marts/mart_customer_orders.sql
│   ├── dbt/models/schema.yml
│   ├── dbt/models/tests.yml
│   └── dbt/macros/macros.sql
│
└── 🔧 Extras
    ├── requirements.txt
    └── .gitignore
```

## 🎯 Fluxos de Trabalho Comuns

### Fluxo 1: "Quero começar agora" (5 min)

1. Leia: [GETTING_STARTED.md](GETTING_STARTED.md)
2. Execute: `make start`
3. Registre: `make register-connector`
4. Teste: `make kafka-topics`

### Fluxo 2: "Entender como funciona" (20 min)

1. Leia: [ARCHITECTURE.md](ARCHITECTURE.md)
2. Execute: `make start && make register-connector`
3. Siga: [EXAMPLES.md](EXAMPLES.md) - Exemplo 1

### Fluxo 3: "Algo deu errado" (variável)

1. Consulte: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Execute: `make health` e `make logs-<serviço>`
3. Se não resolver, execute: `make clean && make start`

### Fluxo 4: "Quero customizar" (30+ min)

1. Leia: [ARCHITECTURE.md](ARCHITECTURE.md) - Padrões de Design
2. Siga: [EXAMPLES.md](EXAMPLES.md) - Exemplos 4-7
3. Edite: arquivos em `dbt/models/`
4. Teste: `make dbt-run && make dbt-test`

## 🌐 Acessar Serviços

| Serviço | URL | Credenciais | Docs |
|---------|-----|-----------|------|
| **PostgreSQL** | localhost:5432 | postgres / postgres | [PostgreSQL](https://www.postgresql.org/docs/) |
| **Kafka** | localhost:9092 | - | [Kafka](https://kafka.apache.org/documentation/) |
| **Debezium** | http://localhost:8083 | - | [Debezium](https://debezium.io/documentation/) |
| **MinIO** | http://localhost:9001 | minioadmin / minioadmin | [MinIO](https://min.io/docs/minio/linux/index.html) |
| **Flink** | http://localhost:8081 | - | [Flink](https://flink.apache.org/docs/) |

## 📚 Recursos Externos

### Tecnologias Usadas

- **[PostgreSQL Logical Replication](https://www.postgresql.org/docs/15/logical-replication.html)**
- **[Debezium Documentation](https://debezium.io/)**
- **[Apache Kafka](https://kafka.apache.org/documentation/)**
- **[Apache Flink](https://flink.apache.org/what-is-flink/)**
- **[Apache Iceberg](https://iceberg.apache.org/docs/latest/)**
- **[dbt Documentation](https://docs.getdbt.com/)**
- **[MinIO Documentation](https://min.io/docs/)**

## 🎓 Tópicos Avançados

### CDC Concepts
- Change Data Capture (CDC)
- Replicação lógica vs física
- Binlog vs WAL
- Exactly-once semantics

### Data Engineering
- Medallion Architecture (Bronze → Silver → Gold)
- Data Quality Tests
- dbt best practices
- Iceberg time travel

### DevOps
- Docker Compose scaling
- Performance tuning
- Monitoring & alerting
- Disaster recovery

Veja cada documento para detalhes específicos.

## ✅ Checklist de Configuração

- [ ] Docker & Docker Compose instalados
- [ ] Repository clonado/baixado
- [ ] `make start` executado com sucesso
- [ ] Health check passou (todos os serviços UP)
- [ ] `make register-connector` bem-sucedido
- [ ] Dados de teste inseridos (`make insert-test-data`)
- [ ] Events vistos em Kafka (`make kafka-consume TOPIC=customers`)
- [ ] dbt models executados (`make dbt-run`)
- [ ] Iceberg tables criadas (verificar em MinIO)

## 🤝 Contribuindo

Sinta-se livre para:
- Reportar bugs
- Sugerir melhorias
- Submeter pull requests
- Compartilhar exemplos

## 📞 Suporte

Se encontrar problemas:

1. **Verifique a saúde:** `make health`
2. **Leia os logs:** `make logs-<serviço>`
3. **Consulte:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
4. **Faça reset:** `make clean && make start`

---

## 🎉 Pronto para começar?

### Próxima ação recomendada:

**[→ Acesse GETTING_STARTED.md](GETTING_STARTED.md)**

Boa sorte! 🚀

---

**Última atualização:** 2024
**Versão:** 1.0.0
