# Sistema HIL (Human-in-the-Loop) - Arquitetura Completa

## 📋 Conceito Central

Sistema de conversação híbrido bot-humano com **transição suave** entre atendimento automatizado e manual, usando **uma única flag** (`is_hil`) para controle de estado.

---

## 🎯 Estados e Transições

### Estados
- **BOT_ATIVO**: `is_hil = false` → Bot responde automaticamente
- **HUMANO_ATIVO**: `is_hil = true` → Bot aguarda, humano controla

### Transições Automáticas
```
BOT → HUMANO:
  - Humano envia mensagem qualquer
  - Bot escala por falha/complexidade
  - Cliente solicita atendimento humano

HUMANO → BOT:
  - Humano digita @bot [instrução]
  - Timeout de inatividade (30min)
```

---

## 🗄️ Schema de Dados Mínimo

### Tabela: conversations
```sql
CREATE TABLE conversations (
  id UUID PRIMARY KEY,
  customer_id TEXT NOT NULL,
  is_hil BOOLEAN DEFAULT false,           -- Flag de controle
  last_activity_at TIMESTAMP NOT NULL,
  last_bot_message_id UUID,               -- Para resumo de contexto
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Tabela: messages
```sql
CREATE TABLE messages (
  id UUID PRIMARY KEY,
  conversation_id UUID NOT NULL,
  sender_type TEXT NOT NULL,              -- 'customer' | 'ai_agent' | 'human_agent'
  content TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT NOW()
);
```

---

## 🏗️ Arquitetura em Camadas

```
┌─────────────────────────────────────┐
│   Orquestrador (HIL Manager)        │ ← Gerencia is_hil, rotas
│   - Webhook Handler (async)          │
│   - Command Parser                   │
└──────────┬──────────────────────────┘
           │
           ├─→ is_hil = false
           │   ┌──────────────────────────┐
           │   │   LangGraph Agent        │ ← Raciocínio complexo
           │   │   - Entender intenção    │
           │   │   - Planejar ações       │
           │   │   - Executar (Composio)  │
           │   │   - Validar resultado    │
           │   │   - Responder/Escalar    │
           │   └──────────────────────────┘
           │
           ├─→ Comandos @...
           │   ┌──────────────────────────┐
           │   │   PydanticAI Agents      │ ← Tarefas especializadas
           │   │   - SummaryAgent         │
           │   │   - AnalysisAgent        │
           │   │   - SuggestionAgent      │
           │   │   - SentimentAgent       │
           │   │   - DraftAgent           │
           │   └──────────────────────────┘
           │
           └─→ is_hil = true
               ┌──────────────────────────┐
               │   Timeout Monitor        │
               │   - Aguarda humano       │
               │   - Detecta @bot         │
               │   - Verifica timeout     │
               └──────────────────────────┘
```

---

## 🤖 LangGraph Agent - Fluxo de Decisão

### Grafo de Estados
```python
entender → planejar → executar → validar → responder
                          ↓
                      escalar (se necessário)
```

### Nós Principais
1. **entender**: Classifica intenção do usuário via LLM
2. **planejar**: Escolhe actions do Composio necessárias
3. **executar**: Executa integrações (Shopify, Salesforce, Gmail, etc)
4. **validar**: Verifica se resultado satisfaz intenção
5. **responder**: Gera resposta natural para o cliente
6. **escalar**: Transfere para humano se necessário

### Condições de Escalação
- ❌ Action falhou após retries
- ❌ Fora da política da empresa
- ❌ Requer aprovação humana
- ❌ Múltiplos erros consecutivos
- ❌ Cliente solicita explicitamente

---

## 🔧 Composio - Superpoderes do Bot

### O que é?
Plataforma de integração que fornece **150+ APIs prontas** com:
- Autenticação gerenciada (OAuth, API keys)
- Rate limiting automático
- Retry inteligente
- Execução segura por usuário

### Integrações Comuns
```python
# CRM & Vendas
Action.SALESFORCE_GET_RECORD
Action.HUBSPOT_GET_CONTACT

# E-commerce
Action.SHOPIFY_GET_ORDER
Action.WOOCOMMERCE_GET_PRODUCT

# Comunicação
Action.GMAIL_SEND_EMAIL
Action.SLACK_SEND_MESSAGE
Action.WHATSAPP_SEND_MESSAGE

# Suporte
Action.ZENDESK_CREATE_TICKET
Action.JIRA_CREATE_ISSUE

# Pagamentos
Action.STRIPE_CREATE_REFUND
Action.PAYPAL_GET_TRANSACTION
```

### Gerenciamento de Identidades
Cada `customer_id` mapeia para um `entity_id` no Composio, permitindo:
- Autenticações separadas por cliente
- Permissões granulares
- Auditoria completa

---

## 💬 Sistema de Comandos (@commands)

### Comandos de Controle
- **@bot [instrução]**: Bot assume conversa com instrução específica
- **@pause**: Pausa bot, humano assume
- **@resume**: Resume bot sem instrução específica

### Comandos de Informação
- **@summary [last:N]**: Resume conversa (PydanticAI)
- **@context**: Mostra contexto completo
- **@status**: Status atual (bot/humano ativo)

### Comandos de Ação
- **@analyze**: Analisa sentimento, urgência, complexidade (PydanticAI)
- **@suggest**: Sugere resposta (PydanticAI)
- **@draft [type:X]**: Cria rascunho (apology, followup, resolved) (PydanticAI)
- **@sentiment**: Análise de sentimento detalhada (PydanticAI)

### Comandos de Escalação
- **@escalate [reason:X]**: Escala para supervisor
- **@transfer department:X**: Transfere departamento

### Comandos Utilitários
- **@help**: Lista todos comandos
- **@history [last:N]**: Mostra histórico

---

## 🧠 PydanticAI Agents - Outputs Estruturados

### SummaryAgent
```python
class ConversationSummary(BaseModel):
    main_topic: str
    customer_issue: str
    current_status: str
    next_steps: list[str]
    sentiment_score: float  # -1 a 1
    urgency: Literal["low", "medium", "high"]
```

### AnalysisAgent
```python
class ConversationAnalysis(BaseModel):
    sentiment: Literal["satisfied", "neutral", "dissatisfied"]
    urgency: Literal["low", "medium", "high"]
    complexity: Literal["simple", "moderate", "complex"]
    recommended_actions: list[str]
    bot_can_handle: bool
    escalation_reason: str | None
```

### SuggestionAgent
```python
class ResponseSuggestion(BaseModel):
    suggested_response: str
    tone: Literal["professional", "empathetic", "apologetic", "casual"]
    confidence: float  # 0 a 1
    alternative_approaches: list[str]
```

### SentimentAgent
```python
class SentimentAnalysis(BaseModel):
    overall_sentiment: float  # -1 a 1
    sentiment_trend: Literal["improving", "worsening", "stable"]
    emotional_keywords: list[str]
    customer_satisfaction: Literal["very_satisfied", "satisfied", "neutral", "dissatisfied", "very_dissatisfied"]
    approach_recommendations: list[str]
```

---

## 🔄 Fluxo Completo de Mensagem

### Cliente envia mensagem
```python
if not conv.is_hil:  # Bot ativo
    # LangGraph processa
    result = langgraph_agent.invoke({
        "messages": history + [new_message],
        "composio_entity_id": entity_id
    })
    
    if result.get("should_escalate"):
        conv.is_hil = True  # Escala para humano
        notify_human_agent(reason, summary)
    else:
        send_response(result["final_response"])
else:  # Humano ativo
    # Apenas encaminha para atendente
    notify_human_agent(new_message)
```

### Humano envia mensagem
```python
command, args = CommandParser.parse(message.content)

if command:  # É um comando @...
    result = await command_handler.handle(conv_id, command, args)
    send_to_agent(result["formatted"])
else:  # Mensagem normal
    conv.is_hil = True  # Humano assume
    start_timeout_monitor(conv_id)
    send_to_customer(message.content)
```

---

## ⏱️ Sistema de Timeout

### Monitor de Inatividade
```python
# Roda a cada 5 minutos
async def check_all_timeouts():
    stalled_convs = db.query("""
        SELECT * FROM conversations 
        WHERE is_hil = true 
        AND last_activity_at < NOW() - INTERVAL '30 minutes'
    """)
    
    for conv in stalled_convs:
        # Sumariza antes de encerrar
        summary = await summary_agent.summarize(messages)
        
        # Encerra conversa
        send_to_customer("Detectamos inatividade...")
        db.archive_conversation(conv.id, summary=summary)
        
        # Notifica atendente
        notify_human_agent("timeout_closed", summary)
```

---

## 📊 Observability & Métricas

### Métricas Prometheus
```python
# Contadores
messages_received_total{sender_type}
commands_executed_total{command}
escalations_total{reason}

# Histogramas (latência)
message_processing_seconds{sender_type}
command_execution_seconds{command}

# Gauges (estado)
active_conversations
bot_active_conversations
human_active_conversations
```

### Logging Estruturado
```python
logger.info(
    "message_processed",
    conversation_id=conv_id,
    sender_type=sender_type,
    duration_ms=duration,
    is_hil=is_hil
)
```

### Tracing (OpenTelemetry)
- Rastreamento completo de cada mensagem
- Spans por comando executado
- Integração com Jaeger

---

## 🚀 Stack Tecnológico

### Core
- **FastAPI**: API assíncrona e webhooks
- **PostgreSQL**: Armazenamento de conversas
- **Redis**: Cache e timeout monitoring

### AI/ML
- **LangGraph**: Orquestração do agente conversacional
- **PydanticAI**: Agents especializados com outputs estruturados
- **OpenAI GPT-4**: LLM base

### Integrações
- **Composio**: 150+ integrações prontas (APIs externas)

### Observability
- **Prometheus**: Métricas
- **Grafana**: Dashboards
- **Jaeger**: Tracing distribuído
- **Structlog**: Logging estruturado

---

## ✅ Vantagens da Arquitetura

### Separação de Responsabilidades
- **HIL Manager**: Apenas roteamento (simples)
- **LangGraph**: Complexidade do agente (isolada)
- **PydanticAI**: Tarefas especializadas (modular)

### Testabilidade
- Cada agente PydanticAI é testável isoladamente
- LangGraph pode ser testado com mock de Composio
- HIL Manager é stateless e fácil de testar

### Escalabilidade
- LangGraph pode rodar em workers separados
- PydanticAI agents são independentes
- Checkpoint permite pausar/resumir grafos

### Flexibilidade
- Adicionar novos comandos não afeta o core
- Mudar lógica de escalação fica no LangGraph
- Múltiplos grafos podem coexistir

---

## 🔒 Segurança

### Gerenciamento de Credenciais
- **Composio** gerencia OAuth e API keys
- Tokens nunca expostos ao código
- Autenticação por usuário (entity_id)

### Validação
- **PydanticAI** valida outputs do LLM
- Schema enforcement em todos os dados
- Rate limiting automático via Composio

---

## 📝 Exemplo de Uso Real

### Cenário: Cliente pergunta sobre pedido

1. **Cliente**: "Onde está meu pedido #12345?"
2. **LangGraph**: 
   - Entende: intenção = "track_order"
   - Planeja: usar `SHOPIFY_GET_ORDER`
   - Executa: busca pedido via Composio
   - Valida: pedido encontrado
   - Responde: "Seu pedido está em trânsito, chega amanhã"
3. **Cliente**: "Quero cancelar"
4. **LangGraph**: 
   - Entende: intenção = "cancel_order"
   - Planeja: verificar política
   - Valida: fora da janela de cancelamento
   - **Escala**: `is_hil = True`
5. **Humano recebe notificação**: "Cliente quer cancelar pedido #12345 (fora da política)"
6. **Humano**: @analyze
7. **SentimentAgent**: Análise detalhada do sentimento
8. **Humano**: @suggest
9. **SuggestionAgent**: Sugere resposta empática
10. **Humano**: [edita e envia resposta personalizada]
11. **Humano**: @bot Continue monitorando
12. **Bot**: Resume conversa

---

## 🎯 Conclusão

Sistema **simples na superfície** (uma flag `is_hil`) mas **poderoso nas capacidades**:

- ✅ Bot inteligente com 150+ integrações
- ✅ Transição suave bot ↔ humano
- ✅ Comandos avançados para atendentes
- ✅ Análise de sentimento e contexto
- ✅ Timeout automático
- ✅ Observabilidade completa
- ✅ Testável e escalável

**Resultado**: Atendimento híbrido eficiente que combina o melhor da automação com toque humano quando necessário.