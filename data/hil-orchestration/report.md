# Plano de Remediação HIL - Versão Corrigida

## Executive Summary

O sistema HIL possui design sólido com PydanticAI agents, LangGraph e Composio, mas requer remediação de riscos sistêmicos antes de produção: (A) governança de LLM (custo/latência), (B) consistência de estado concorrente, (C) segurança/privacidade (PII, compliance LGPD/GDPR), (D) observabilidade (traces, métricas), (E) operação (rate limiting, idempotência, budgets). Este plano corrigido endereça esses riscos com fases implementáveis, critérios objetivos e rollback explícito.

---

## Categorias de Prioridade

- **P0 (Bloqueadores)**: Deve implementar antes de produção
- **P1 (Alto Risco)**: Implementar antes de rollout amplo
- **P2 (Médio)**: Importante, pode lançar com mitigações
- **P3 (Hardening)**: Melhorias incrementais pós-lançamento

---

# P0 — Bloqueadores Críticos

## P0.1 - LLM Governance & Safe-Call Wrapper (FASEADO)

**Problema**: Uso desregulado de GPT-4, sem timeouts, retries ou circuit breaker.

**Impacto**: Custos imprevisíveis, timeouts frequentes, SLA inconsistente.

---

### P0.1a - MVP Wrapper (Semana 1)

**Implementar agora:**

```python
import asyncio
from typing import TypeVar, Callable
from functools import wraps
import time

T = TypeVar('T')

class LLMCallMetrics:
    def __init__(self):
        self.calls_total = Counter('llm_calls_total', ['model', 'outcome'])
        self.duration = Histogram('llm_call_duration_seconds', ['model'])
        self.tokens = Counter('llm_tokens_total', ['model', 'type'])

metrics = LLMCallMetrics()

async def safe_llm_call_mvp(
    func: Callable[..., T],
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    model_name: str = "gpt-4"
) -> T:
    """MVP wrapper: timeout + retries básicos + métricas"""
    
    for attempt in range(max_retries):
        start = time.time()
        try:
            # Timeout simples
            result = await asyncio.wait_for(
                func(),
                timeout=timeout_seconds
            )
            
            # Métricas de sucesso
            duration = time.time() - start
            metrics.calls_total.labels(model=model_name, outcome="success").inc()
            metrics.duration.labels(model=model_name).observe(duration)
            
            return result
            
        except asyncio.TimeoutError:
            metrics.calls_total.labels(model=model_name, outcome="timeout").inc()
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Backoff fixo
            
        except Exception as e:
            metrics.calls_total.labels(model=model_name, outcome="error").inc()
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

# Uso imediato:
async def call_summary_agent(messages):
    return await safe_llm_call_mvp(
        lambda: summary_agent.summarize(messages),
        timeout_seconds=20.0,
        model_name="gpt-4"
    )
```

**Owner**: Backend Engineer  
**Timeline**: 3-5 dias  
**Acceptance Criteria**:
- ✅ Todos agentes PydanticAI usam wrapper MVP
- ✅ Métricas Prometheus expostas: `llm_calls_total`, `llm_call_duration_seconds`
- ✅ Timeouts configuráveis por agent
- ✅ Retries com backoff exponencial (2^n)

**Testes**:
```python
@pytest.mark.asyncio
async def test_safe_llm_call_timeout():
    async def slow_llm():
        await asyncio.sleep(100)
    
    with pytest.raises(asyncio.TimeoutError):
        await safe_llm_call_mvp(slow_llm, timeout_seconds=1.0)

@pytest.mark.asyncio
async def test_safe_llm_call_retry():
    call_count = 0
    async def flaky_llm():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Transient")
        return "Success"
    
    result = await safe_llm_call_mvp(flaky_llm, max_retries=3)
    assert result == "Success"
    assert call_count == 3
```

**Rollback**: Feature flag `use_llm_wrapper_mvp`. Se false, reverte para chamadas diretas.

---

### P1.5a - Iteração 2: Circuit Breaker + Model Tiering (Semana 3-4)

**Adicionar após MVP estável:**

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.last_failure_time = None
        self.state = "closed"  # closed | open | half_open
    
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning("circuit_breaker_opened", threshold=self.failure_threshold)
    
    def can_attempt(self) -> bool:
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check se timeout passou
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_seconds):
                self.state = "half_open"
                return True
            return False
        
        # half_open: permite uma tentativa
        return True

# Circuit breaker por modelo
circuit_breakers = {
    "gpt-4": CircuitBreaker(failure_threshold=5, timeout_seconds=60),
    "gpt-3.5-turbo": CircuitBreaker(failure_threshold=10, timeout_seconds=30)
}

# Model tiering
MODEL_TIERS = {
    "classification": "gpt-3.5-turbo",
    "summary": "gpt-4",
    "analysis": "gpt-4",
    "draft": "gpt-4"
}

async def safe_llm_call_v2(
    func: Callable,
    task_type: str = "summary",
    timeout_seconds: float = 30.0,
    max_retries: int = 3
):
    """V2: Circuit breaker + model tiering"""
    
    # Seleciona modelo pelo tipo de tarefa
    model = MODEL_TIERS.get(task_type, "gpt-4")
    breaker = circuit_breakers[model]
    
    # Verifica circuit breaker
    if not breaker.can_attempt():
        logger.error("circuit_breaker_open", model=model)
        raise CircuitBreakerOpenError(f"Circuit breaker open for {model}")
    
    try:
        result = await safe_llm_call_mvp(func, timeout_seconds, max_retries, model)
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure()
        raise
```

**Owner**: ML Infra Engineer  
**Acceptance Criteria**:
- ✅ Circuit breaker abre após 5 falhas consecutivas
- ✅ Half-open permite retry após 60s
- ✅ Model tiering reduz custos (classification usa gpt-3.5)
- ✅ Métrica `circuit_breaker_state{model}` (0=closed, 1=open, 2=half_open)

---

### P2.15 - Iteração 3: Token Accounting + Pydantic Re-ask (Semana 6-8)

**Adicionar após circuit breaker estável.**

---

## P0.2 - Concurrency & State Consistency (PESSIMISTIC LOCK)

**Problema**: `is_hil`, `last_bot_message_id` têm race conditions. Workers podem sobrescrever estado.

**Decisão**: Usar **pessimistic locking** (`FOR UPDATE NOWAIT`) porque:
- Comandos `@bot`/`@pause` são raros (<1/seg)
- Consistência de `is_hil` é crítica (não pode retry)
- Timeout monitor já é naturalmente serializado

---

### Implementação

**DDL (migrations):**

```sql
-- Migration 001: Add concurrency control
ALTER TABLE conversations
  ADD COLUMN version BIGINT DEFAULT 0 NOT NULL,
  ADD COLUMN escalated BOOLEAN DEFAULT false,
  ADD COLUMN department TEXT;

CREATE INDEX idx_conversations_is_hil ON conversations(is_hil) WHERE is_hil = true;
CREATE INDEX idx_conversations_last_activity ON conversations(last_activity_at) WHERE is_hil = true;

-- Migration 002: Message idempotency
ALTER TABLE messages
  ADD COLUMN message_origin_id TEXT;

CREATE UNIQUE INDEX messages_conv_origin_idx 
ON messages (conversation_id, message_origin_id) 
WHERE message_origin_id IS NOT NULL;

-- Migration 003: Audit trail
CREATE TABLE conversation_audit (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  actor TEXT NOT NULL,  -- 'system' | 'agent_<id>' | 'bot'
  action TEXT NOT NULL,  -- 'hil_activated' | 'hil_deactivated' | 'escalated' | 'timeout'
  old_is_hil BOOLEAN,
  new_is_hil BOOLEAN,
  reason TEXT,
  details JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_audit_conversation ON conversation_audit(conversation_id, created_at DESC);
CREATE INDEX idx_audit_created ON conversation_audit(created_at DESC);
```

**Python implementation:**

```python
from contextlib import asynccontextmanager
import asyncpg

class ConversationRepository:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool
    
    @asynccontextmanager
    async def lock_conversation(self, conversation_id: str):
        """Adquire lock pessimístico na conversa"""
        async with self.db.acquire() as conn:
            async with conn.transaction():
                # Pessimistic lock: falha rápido se locked
                conv = await conn.fetchrow("""
                    SELECT * FROM conversations 
                    WHERE id = $1 
                    FOR UPDATE NOWAIT
                """, conversation_id)
                
                if not conv:
                    raise ConversationNotFoundError(conversation_id)
                
                yield conv
    
    async def update_is_hil(
        self, 
        conversation_id: str, 
        new_is_hil: bool,
        actor: str,
        reason: str
    ):
        """Atualiza is_hil com lock + audit"""
        async with self.lock_conversation(conversation_id) as conv:
            old_is_hil = conv['is_hil']
            
            await self.db.execute("""
                UPDATE conversations 
                SET 
                    is_hil = $1,
                    version = version + 1,
                    last_activity_at = NOW()
                WHERE id = $2
            """, new_is_hil, conversation_id)
            
            # Audit log
            await self.db.execute("""
                INSERT INTO conversation_audit 
                (conversation_id, actor, action, old_is_hil, new_is_hil, reason)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, 
                conversation_id,
                actor,
                'hil_activated' if new_is_hil else 'hil_deactivated',
                old_is_hil,
                new_is_hil,
                reason
            )

# Uso:
async def handle_bot_takeover(conv_id: str, agent_id: str):
    try:
        await repo.update_is_hil(
            conv_id, 
            new_is_hil=False,
            actor=f"agent_{agent_id}",
            reason="@bot command"
        )
    except asyncpg.LockNotAvailableError:
        # Outro worker está modificando
        raise ConflictError("Conversation is being modified by another process")
```

**Owner**: Backend + DB Engineer  
**Timeline**: 5-7 dias  
**Acceptance Criteria**:
- ✅ Todas mutações de `is_hil` usam `FOR UPDATE NOWAIT`
- ✅ Audit log registra todas transições com actor/reason
- ✅ Lock timeout = 5s (config `lock_timeout`)
- ✅ Métrica `conversation_lock_conflicts_total`

**Testes**:
```python
@pytest.mark.asyncio
async def test_concurrent_is_hil_toggle():
    """Simula dois workers tentando toggle simultâneo"""
    conv_id = await create_test_conversation()
    
    async def worker1():
        await repo.update_is_hil(conv_id, True, "agent_1", "pause")
    
    async def worker2():
        await asyncio.sleep(0.01)  # Pequeno delay
        await repo.update_is_hil(conv_id, False, "agent_2", "resume")
    
    # Um deve suceder, outro deve falhar com LockNotAvailable
    results = await asyncio.gather(
        worker1(), 
        worker2(), 
        return_exceptions=True
    )
    
    # Exatamente um deve ser LockNotAvailableError
    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 1
    assert isinstance(errors[0], asyncpg.LockNotAvailableError)
    
    # Audit log deve ter ambas tentativas
    audit = await db.fetch("SELECT * FROM conversation_audit WHERE conversation_id = $1", conv_id)
    assert len(audit) >= 1  # Pelo menos a que sucedeu
```

**Rollback**: Feature flag `use_pessimistic_locks`. Se desabilitado, usa updates simples (aceita race condition temporariamente).

---

## P0.3 - Idempotency no Webhook (PROVIDER RESPONSIBILITY)

**Problema**: Webhooks podem ser re-entregues, causando mensagens duplicadas.

**Decisão**: **Provider deve fornecer `origin_id` estável**. Sistema não assume ownership de gerar hash canônico.

---

### Implementação

```python
from fastapi import HTTPException
from datetime import timedelta

class MessageDeduplication:
    def __init__(self, redis_client, db_pool):
        self.redis = redis_client
        self.db = db_pool
    
    async def ensure_idempotent(
        self, 
        conversation_id: str,
        message: IncomingMessage
    ) -> bool:
        """
        Retorna True se mensagem deve ser processada.
        Retorna False se é duplicata (já processada).
        """
        
        # Caso 1: Provider forneceu origin_id (OBRIGATÓRIO)
        if message.origin_id:
            # Check no DB (persistente)
            exists = await self.db.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM messages 
                    WHERE conversation_id = $1 
                    AND message_origin_id = $2
                )
            """, conversation_id, message.origin_id)
            
            if exists:
                logger.info(
                    "duplicate_message_rejected",
                    conversation_id=conversation_id,
                    origin_id=message.origin_id
                )
                return False
            
            return True
        
        # Caso 2: Provider NÃO forneceu origin_id (EMERGÊNCIA)
        # Dedupe efêmero de 5 minutos apenas
        logger.warning(
            "message_without_origin_id",
            conversation_id=conversation_id,
            provider=message.provider
        )
        
        # Hash efêmero (NÃO persistente)
        ephemeral_key = f"dedupe:emerg:{conversation_id}:{hash(message.content)}"
        
        # Tenta set com NX (only if not exists) + TTL 5min
        was_set = await self.redis.set(
            ephemeral_key, 
            "1", 
            nx=True, 
            ex=300  # 5 minutos
        )
        
        if not was_set:
            logger.warning(
                "duplicate_message_rejected_ephemeral",
                conversation_id=conversation_id
            )
            return False
        
        return True

# Webhook handler
@app.post("/webhook/message")
async def receive_message(message: IncomingMessage):
    # Validação
    if not message.origin_id:
        logger.error(
            "provider_missing_origin_id",
            provider=message.provider,
            conversation_id=message.conversation_id
        )
        # AINDA aceita (com dedupe efêmero), mas loga erro
    
    # Dedupe check
    dedupe = MessageDeduplication(redis, db)
    should_process = await dedupe.ensure_idempotent(
        message.conversation_id,
        message
    )
    
    if not should_process:
        # Duplicata: retorna 200 para provider não retry
        return {"status": "duplicate", "processed": False}
    
    # Processa mensagem
    background_tasks.add_task(process_message, message)
    
    return {"status": "received", "processed": True}

# Inserção de mensagem
async def save_message(conversation_id: str, message: IncomingMessage):
    await db.execute("""
        INSERT INTO messages 
        (conversation_id, sender_type, sender_id, content, message_origin_id)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (conversation_id, message_origin_id) 
        DO NOTHING
    """, 
        conversation_id,
        message.sender_type,
        message.sender_id,
        message.content,
        message.origin_id  # Pode ser None
    )
```

**Owner**: Integration Engineer  
**Timeline**: 3-4 dias  
**Acceptance Criteria**:
- ✅ Mensagens com `origin_id` nunca duplicam (DB constraint)
- ✅ Mensagens sem `origin_id` têm dedupe efêmero (5min window)
- ✅ Provider é notificado sobre `origin_id` ausente (docs)
- ✅ Webhook retorna 200 mesmo para duplicatas
- ✅ Métrica `messages_deduplicated_total{method}`

**Testes**:
```python
@pytest.mark.asyncio
async def test_idempotent_webhook_with_origin_id():
    message = IncomingMessage(
        conversation_id="conv_123",
        origin_id="msg_abc",
        content="Hello"
    )
    
    # Primeira entrega
    response1 = await client.post("/webhook/message", json=message.dict())
    assert response1.status_code == 200
    assert response1.json()["processed"] == True
    
    # Retry (duplicata)
    response2 = await client.post("/webhook/message", json=message.dict())
    assert response2.status_code == 200
    assert response2.json()["processed"] == False
    
    # Verifica: apenas 1 mensagem no DB
    count = await db.fetchval(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = $1",
        "conv_123"
    )
    assert count == 1

@pytest.mark.asyncio
async def test_ephemeral_dedupe_without_origin_id():
    message = IncomingMessage(
        conversation_id="conv_456",
        origin_id=None,  # SEM origin_id
        content="Hello"
    )
    
    # Primeira entrega
    response1 = await client.post("/webhook/message", json=message.dict())
    assert response1.json()["processed"] == True
    
    # Retry imediato (dentro de 5min)
    response2 = await client.post("/webhook/message", json=message.dict())
    assert response2.json()["processed"] == False
    
    # Após 5min, permite reprocessar (limitação do ephemeral)
    await asyncio.sleep(301)
    response3 = await client.post("/webhook/message", json=message.dict())
    assert response3.json()["processed"] == True
```

**Rollback**: Feature flag `enforce_message_origin_id`. Se false, aceita qualquer mensagem (útil para desenvolvimento).

---

## P0.4 - PII Sanitization (LAYERED DEFENSE)

**Problema**: Conversas contêm PII que vão para logs e prompts LLM. Regex sozinho é insuficiente.

**Solução**: Defesa em camadas (Regex + NER + Opt-in) + Audit trail.

---

### Implementação

```python
import re
from typing import Tuple
import spacy

# Layer 1: Regex patterns (rápido, básico)
PII_PATTERNS = {
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'phone_br': re.compile(r'\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b'),
    'cpf': re.compile(r'\b\d{3}\.?\d{3}\.?\d{3}[-\s]?\d{2}\b'),
    'cnpj': re.compile(r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}[-\s]?\d{2}\b'),
    'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
}

# Layer 2: NER model (mais preciso)
nlp = spacy.load("pt_core_news_lg")

class PIISanitizer:
    def __init__(self):
        self.nlp = nlp
    
    def redact_layer1_regex(self, text: str) -> Tuple[str, list]:
        """Layer 1: Regex-based redaction"""
        warnings = []
        
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)
                warnings.append(f"regex_detected_{pii_type}")
        
        return text, warnings
    
    def redact_layer2_ner(self, text: str) -> Tuple[str, list]:
        """Layer 2: NER-based redaction"""
        warnings = []
        doc = self.nlp(text)
        
        entities_to_redact = ["PER", "LOC", "ORG"]  # Person, Location, Organization
        
        for ent in doc.ents:
            if ent.label_ in entities_to_redact:
                text = text.replace(ent.text, f"[{ent.label_}_REDACTED]")
                warnings.append(f"ner_detected_{ent.label_}")
        
        return text, warnings
    
    def redact_full(self, text: str, use_ner: bool = True) -> Tuple[str, dict]:
        """Full redaction pipeline"""
        warnings = []
        
        # Layer 1: Regex (sempre)
        text, regex_warnings = self.redact_layer1_regex(text)
        warnings.extend(regex_warnings)
        
        # Layer 2: NER (opcional, mais lento)
        if use_ner:
            text, ner_warnings = self.redact_layer2_ner(text)
            warnings.extend(ner_warnings)
        
        return text, {
            "redacted": len(warnings) > 0,
            "warnings": warnings
        }

sanitizer = PIISanitizer()

# Modelos Pydantic com PII awareness
class Message(BaseModel):
    content: str
    contains_pii: bool = False
    pii_consent_given: bool = False
    pii_consent_at: datetime | None = None

# Uso em logs
import structlog

logger = structlog.get_logger()

async def log_message_safely(message: Message, conversation_id: str):
    """Loga mensagem com PII redactado"""
    
    if message.contains_pii and not message.pii_consent_given:
        # Redacta antes de logar
        safe_content, redaction_info = sanitizer.redact_full(message.content)
        
        logger.info(
            "message_logged",
            conversation_id=conversation_id,
            content=safe_content,
            redaction_applied=True,
            redaction_warnings=redaction_info["warnings"]
        )
    else:
        # Loga completo (com consent)
        logger.info(
            "message_logged",
            conversation_id=conversation_id,
            content=message.content[:100],  # Trunca
            has_pii_consent=message.pii_consent_given
        )

# Uso em prompts LLM
async def build_llm_prompt(conversation: Conversation, messages: list[Message]) -> str:
    """Constrói prompt com PII handling"""
    
    prompt_parts = []
    
    for msg in messages:
        if msg.contains_pii:
            if msg.pii_consent_given:
                # Cliente autorizou uso de PII
                content = msg.content
                
                # Audit: registra uso de PII
                await audit_pii_usage(
                    conversation.id,
                    reason="llm_prompt",
                    consent_timestamp=msg.pii_consent_at
                )
            else:
                # SEM consent: redacta
                content, _ = sanitizer.redact_full(msg.content)
        else:
            content = msg.content
        
        prompt_parts.append(f"[{msg.sender_type}] {content}")
    
    return "\n".join(prompt_parts)

# Audit de uso de PII
async def audit_pii_usage(
    conversation_id: str,
    reason: str,
    consent_timestamp: datetime
):
    await db.execute("""
        INSERT INTO pii_usage_audit 
        (conversation_id, used_at, reason, consent_timestamp)
        VALUES ($1, NOW(), $2, $3)
    """, conversation_id, reason, consent_timestamp)
```

**DDL adicional:**

```sql
-- Audit de uso de PII
CREATE TABLE pii_usage_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    used_at TIMESTAMPTZ DEFAULT now(),
    reason TEXT NOT NULL,  -- 'llm_prompt' | 'composio_action' | 'log_debug'
    consent_timestamp TIMESTAMPTZ,
    details JSONB
);

CREATE INDEX idx_pii_audit_conversation ON pii_usage_audit(conversation_id, used_at DESC);
```

**Owner**: Security Engineer + Backend  
**Timeline**: 7-10 dias  
**Acceptance Criteria**:
- ✅ Logs NUNCA contêm PII não-redactado
- ✅ Regex detecta: email, phone, CPF, CNPJ, cartão de crédito
- ✅ NER detecta: nomes, localizações
- ✅ PII usado em prompts requer consent + audit trail
- ✅ Métrica `pii_redacted_total{layer}` (regex/ner)

**Testes**:
```python
def test_pii_redaction_regex():
    text = "Meu email é john@example.com e CPF 123.456.789-00"
    
    redacted, info = sanitizer.redact_full(text, use_ner=False)
    
    assert "john@example.com" not in redacted
    assert "123.456.789-00" not in redacted
    assert "[EMAIL_REDACTED]" in redacted
    assert "[CPF_REDACTED]" in redacted
    assert len(info["warnings"]) == 2

def test_pii_not_in_logs(caplog):
    message = Message(
        content="Meu CPF é 123.456.789-00",
        contains_pii=True,
        pii_consent_given=False
    )
    
    log_message_safely(message, "conv_123")
    
    # Garante que CPF NÃO aparece em logs
    log_output = caplog.text
    assert "123.456.789-00" not in log_output
    assert "[CPF_REDACTED]" in log_output
```

**Rollback**: Feature flag `enforce_pii_redaction`. Se false, apenas loga warnings sem redactar.

---

# P1 — Alto Risco (antes de rollout amplo)

## P1.1 - LGPD/GDPR Compliance (PROMOVIDO de P3)

**Problema**: Ausência de mecanismos de retenção e deleção de dados. Violação de LGPD/GDPR desde o primeiro usuário.

**Impacto**: Multas de até 2% do faturamento (LGPD) ou €20M (GDPR).

---

### Implementação

**DDL:**

```sql
-- Data retention policy
CREATE TABLE data_retention_policy (
    data_type TEXT PRIMARY KEY,
    retention_days INT NOT NULL,
    deletion_method TEXT NOT NULL CHECK (deletion_method IN ('soft_delete', 'hard_delete', 'anonymize'))
);

-- Políticas iniciais
INSERT INTO data_retention_policy VALUES
    ('conversation_messages', 90, 'soft_delete'),
    ('conversation_summary', 730, 'anonymize'),  -- 2 anos
    ('audit_logs', 2555, 'hard_delete'),  -- 7 anos (compliance)
    ('pii_usage_audit', 2555, 'hard_delete');

-- Soft delete em messages
ALTER TABLE messages ADD COLUMN deleted_at TIMESTAMPTZ;
ALTER TABLE messages ADD COLUMN anonymized BOOLEAN DEFAULT false;

CREATE INDEX idx_messages_deleted ON messages(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX idx_messages_retention ON messages(timestamp) WHERE deleted_at IS NULL;
```

**API de Deleção:**

```python
from fastapi import HTTPException, BackgroundTasks
from datetime import datetime, timedelta

class GDPRService:
    def __init__(self, db_pool, redis_client):
        self.db = db_pool
        self.redis = redis_client
    
    async def delete_customer_data(
        self, 
        customer_id: str,
        requester_email: str,
        reason: str = "customer_request"
    ) -> dict:
        """
        Endpoint LGPD/GDPR: Deleta todos dados de um customer.
        Processo:
        1. Soft delete de mensagens
        2. Anonymiza conversas
        3. Mantém audit logs (compliance)
        4. Registra request de deleção
        """
        
        async with self.db.acquire() as conn:
            async with conn.transaction():
                # 1. Find all conversations
                conversations = await conn.fetch("""
                    SELECT id FROM conversations 
                    WHERE customer_id = $1
                """, customer_id)
                
                conv_ids = [c['id'] for c in conversations]
                
                if not conv_ids:
                    raise HTTPException(404, f"No data found for customer {customer_id}")
                
                # 2. Soft delete messages
                deleted_messages = await conn.fetchval("""
                    UPDATE messages 
                    SET 
                        deleted_at = NOW(),
                        content = '[DELETED PER GDPR REQUEST]'
                    WHERE conversation_id = ANY($1)
                    AND deleted_at IS NULL
                    RETURNING COUNT(*)
                """, conv_ids)
                
                # 3. Anonymize conversations
                await conn.execute("""
                    UPDATE conversations
                    SET 
                        customer_id = 'ANONYMIZED_' || id::text,
                        last_activity_at = NOW()
                    WHERE id = ANY($1)
                """, conv_ids)
                
                # 4. Log deletion request (mantém para compliance)
                await conn.execute("""
                    INSERT INTO gdpr_deletion_requests
                    (customer_id, requester_email, reason, conversations_affected, messages_deleted, completed_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                """, customer_id, requester_email, reason, len(conv_ids), deleted_messages)
                
                # 5. Invalida caches
                for conv_id in conv_ids:
                    await self.redis.delete(f"conversation:{conv_id}")
                
                logger.info(
                    "gdpr_deletion_completed",
                    customer_id=customer_id,
                    conversations_affected=len(conv_ids),
                    messages_deleted=deleted_messages
                )
                
                return {
                    "status": "completed",
                    "customer_id": customer_id,
                    "conversations_deleted": len(conv_ids),
                    "messages_deleted": deleted_messages,
                    "completed_at": datetime.now().isoformat()
                }

# Endpoint FastAPI
@app.delete("/api/v1/gdpr/customer/{customer_id}")
async def delete_customer_data(
    customer_id: str,
    requester: GDPRDeleteRequest,
    background_tasks: BackgroundTasks
):
    """
    GDPR/LGPD: Delete customer data.
    
    Requires:
    - requester_email: Email of person requesting deletion
    - confirmation_token: Token sent via email to customer
    """
    
    # Verifica token de confirmação
    token_valid = await verify_gdpr_token(customer_id, requester.confirmation_token)
    if not token_valid:
        raise HTTPException(403, "Invalid confirmation token")
    
    gdpr_service = GDPRService(db, redis)
    
    # Executa deleção
    result = await gdpr_service.delete_customer_data(
        customer_id,
        requester.requester_email,
        requester.reason
    )
    
    # Notifica customer
    background_tasks.add_task(
        send_deletion_confirmation_email,
        requester.requester_email,
        result
    )
    
    return result

# Automated retention cleanup (cron job)
async def cleanup_expired_data():
    """
    Job que roda diariamente para limpar dados expirados.
    """
    policies = await db.fetch("SELECT * FROM data_retention_policy")
    
    for policy in policies:
        cutoff_date = datetime.now() - timedelta(days=policy['retention_days'])
        
        if policy['data_type'] == 'conversation_messages':
            if policy['deletion_method'] == 'soft_delete':
                deleted = await db.fetchval("""
                    UPDATE messages 
                    SET deleted_at = NOW(),
                        content = '[EXPIRED]'
                    WHERE timestamp < $1
                    AND deleted_at IS NULL
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                logger.info(
                    "retention_cleanup",
                    data_type=policy['data_type'],
                    method='soft_delete',
                    records_deleted=deleted
                )
            
            elif policy['deletion_method'] == 'hard_delete':
                # Hard delete: remove permanentemente
                deleted = await db.fetchval("""
                    DELETE FROM messages 
                    WHERE timestamp < $1
                    AND deleted_at IS NOT NULL
                    AND deleted_at < NOW() - INTERVAL '30 days'
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                logger.info(
                    "retention_cleanup_hard",
                    data_type=policy['data_type'],
                    records_deleted=deleted
                )
        
        elif policy['data_type'] == 'conversation_summary':
            if policy['deletion_method'] == 'anonymize':
                # Mantém summaries mas remove PII
                anonymized = await db.fetchval("""
                    UPDATE conversations
                    SET customer_id = 'ANONYMIZED_' || id::text
                    WHERE created_at < $1
                    AND customer_id NOT LIKE 'ANONYMIZED_%'
                    RETURNING COUNT(*)
                """, cutoff_date)
                
                logger.info(
                    "retention_anonymization",
                    records_anonymized=anonymized
                )

# Schedule cleanup job
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()
scheduler.add_job(cleanup_expired_data, 'cron', hour=3, minute=0)  # 3 AM diário
scheduler.start()
```

**DDL adicional:**

```sql
CREATE TABLE gdpr_deletion_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id TEXT NOT NULL,
    requester_email TEXT NOT NULL,
    reason TEXT,
    conversations_affected INT,
    messages_deleted INT,
    requested_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed'))
);

CREATE INDEX idx_gdpr_requests_customer ON gdpr_deletion_requests(customer_id);
CREATE INDEX idx_gdpr_requests_completed ON gdpr_deletion_requests(completed_at DESC);
```

**Owner**: Legal + Backend Engineer  
**Timeline**: 10-14 dias  
**Acceptance Criteria**:
- ✅ Endpoint `/gdpr/customer/{id}` deleta dados <24h
- ✅ Cleanup automático roda diariamente às 3 AM
- ✅ Audit logs preservados por 7 anos (compliance)
- ✅ Email de confirmação enviado ao customer
- ✅ Métricas: `gdpr_deletions_total`, `retention_cleanup_records_total{type}`

**Testes**:
```python
@pytest.mark.asyncio
async def test_gdpr_deletion():
    # Setup: cria customer com dados
    customer_id = "test_customer_123"
    conv_id = await create_test_conversation(customer_id)
    await create_test_messages(conv_id, count=10)
    
    # Executa deleção
    token = await generate_gdpr_token(customer_id)
    response = await client.delete(
        f"/api/v1/gdpr/customer/{customer_id}",
        json={"requester_email": "test@example.com", "confirmation_token": token}
    )
    
    assert response.status_code == 200
    assert response.json()["messages_deleted"] == 10
    
    # Verifica: mensagens soft-deleted
    messages = await db.fetch(
        "SELECT * FROM messages WHERE conversation_id = $1",
        conv_id
    )
    assert all(m['deleted_at'] is not None for m in messages)
    assert all(m['content'] == '[DELETED PER GDPR REQUEST]' for m in messages)
    
    # Verifica: conversation anonymizada
    conv = await db.fetchrow(
        "SELECT * FROM conversations WHERE id = $1",
        conv_id
    )
    assert conv['customer_id'].startswith('ANONYMIZED_')
    
    # Verifica: audit trail existe
    audit = await db.fetchrow(
        "SELECT * FROM gdpr_deletion_requests WHERE customer_id = $1",
        customer_id
    )
    assert audit is not None
    assert audit['status'] == 'completed'

@pytest.mark.asyncio
async def test_retention_cleanup():
    # Setup: cria mensagens antigas
    old_date = datetime.now() - timedelta(days=100)
    conv_id = await create_test_conversation()
    await db.execute("""
        INSERT INTO messages (conversation_id, sender_type, content, timestamp)
        VALUES ($1, 'customer', 'Old message', $2)
    """, conv_id, old_date)
    
    # Roda cleanup
    await cleanup_expired_data()
    
    # Verifica: mensagem antiga foi soft-deleted
    msg = await db.fetchrow(
        "SELECT * FROM messages WHERE conversation_id = $1",
        conv_id
    )
    assert msg['deleted_at'] is not None
    assert msg['content'] == '[EXPIRED]'
```

**Rollback**: Feature flag `enforce_gdpr_compliance`. Se false, endpoints retornam 501 Not Implemented.

---

## P1.2 - Rate Limiting & Abuse Prevention (NOVO)

**Problema**: Nenhum rate limiting nos webhooks. Abuso pode causar DoS e custos explosivos.

**Impacto**: 
- Customer malicioso envia 1000 msgs/seg → 1000 LLM calls → $100+ gasto em segundos
- Retry loops infinitos do provider → sistema paralisa

---

### Implementação

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis.asyncio as aioredis

# Setup SlowAPI
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class RateLimiter:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def check_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> bool:
        """
        Verifica rate limit usando sliding window no Redis.
        Retorna True se dentro do limite, False se excedeu.
        """
        now = time.time()
        window_start = now - window_seconds
        
        pipe = self.redis.pipeline()
        
        # Remove entradas antigas
        pipe.zremrangebyscore(key, '-inf', window_start)
        
        # Adiciona entrada atual
        pipe.zadd(key, {str(now): now})
        
        # Conta entradas na janela
        pipe.zcard(key)
        
        # Set TTL
        pipe.expire(key, window_seconds)
        
        results = await pipe.execute()
        count = results[2]  # zcard result
        
        return count <= limit
    
    async def check_customer_limit(self, customer_id: str) -> bool:
        """10 mensagens/minuto por customer"""
        return await self.check_limit(
            f"rate:customer:{customer_id}",
            limit=10,
            window_seconds=60
        )
    
    async def check_customer_hourly_limit(self, customer_id: str) -> bool:
        """100 mensagens/hora por customer"""
        return await self.check_limit(
            f"rate:customer:hourly:{customer_id}",
            limit=100,
            window_seconds=3600
        )
    
    async def check_agent_limit(self, agent_id: str) -> bool:
        """50 comandos/minuto por agent"""
        return await self.check_limit(
            f"rate:agent:{agent_id}",
            limit=50,
            window_seconds=60
        )
    
    async def get_retry_after(self, key: str, window_seconds: int) -> int:
        """Calcula Retry-After header (segundos até window reset)"""
        scores = await self.redis.zrange(key, 0, 0, withscores=True)
        if not scores:
            return window_seconds
        
        oldest_timestamp = scores[0][1]
        window_start = time.time() - window_seconds
        
        if oldest_timestamp < window_start:
            return 0
        
        return int(oldest_timestamp + window_seconds - time.time())

rate_limiter = RateLimiter(redis)

# Middleware de rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Global rate limit
    client_ip = request.client.host
    within_global_limit = await rate_limiter.check_limit(
        f"rate:global:{client_ip}",
        limit=100,
        window_seconds=60
    )
    
    if not within_global_limit:
        retry_after = await rate_limiter.get_retry_after(
            f"rate:global:{client_ip}",
            60
        )
        
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "retry_after": retry_after},
            headers={"Retry-After": str(retry_after)}
        )
    
    return await call_next(request)

# Webhook com rate limiting específico
@app.post("/webhook/message")
async def receive_message(message: IncomingMessage):
    # Per-customer rate limiting
    within_customer_limit = await rate_limiter.check_customer_limit(message.customer_id)
    within_hourly_limit = await rate_limiter.check_customer_hourly_limit(message.customer_id)
    
    if not within_customer_limit:
        retry_after = await rate_limiter.get_retry_after(
            f"rate:customer:{message.customer_id}",
            60
        )
        
        logger.warning(
            "customer_rate_limit_exceeded",
            customer_id=message.customer_id,
            limit="10/minute"
        )
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Customer rate limit exceeded (10/minute)",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )
    
    if not within_hourly_limit:
        retry_after = await rate_limiter.get_retry_after(
            f"rate:customer:hourly:{message.customer_id}",
            3600
        )
        
        logger.warning(
            "customer_hourly_rate_limit_exceeded",
            customer_id=message.customer_id,
            limit="100/hour"
        )
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Customer rate limit exceeded (100/hour)",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )
    
    # Agent command rate limiting
    if message.sender_type == 'human_agent':
        within_agent_limit = await rate_limiter.check_agent_limit(message.sender_id)
        
        if not within_agent_limit:
            retry_after = await rate_limiter.get_retry_after(
                f"rate:agent:{message.sender_id}",
                60
            )
            
            logger.warning(
                "agent_rate_limit_exceeded",
                agent_id=message.sender_id,
                limit="50/minute"
            )
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Agent rate limit exceeded (50/minute)",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
    
    # Processa mensagem
    background_tasks.add_task(process_message, message)
    
    return {"status": "received"}

# Métricas
rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit violations',
    ['entity_type', 'limit_type']
)

# Budget protection adicional
class BudgetProtection:
    def __init__(self, db_pool, redis_client):
        self.db = db_pool
        self.redis = redis_client
    
    async def check_customer_budget(self, customer_id: str) -> bool:
        """Verifica se customer ainda tem budget disponível"""
        # Check cache primeiro
        cache_key = f"budget:{customer_id}"
        cached = await self.redis.get(cache_key)
        
        if cached == "exceeded":
            return False
        
        # Busca budget do DB
        budget = await self.db.fetchrow("""
            SELECT 
                monthly_budget_usd,
                current_spend_usd,
                budget_reset_at
            FROM customer_budgets
            WHERE customer_id = $1
        """, customer_id)
        
        if not budget:
            # Sem budget configurado = unlimited
            return True
        
        # Reset mensal
        if datetime.now() > budget['budget_reset_at']:
            await self.db.execute("""
                UPDATE customer_budgets
                SET 
                    current_spend_usd = 0,
                    budget_reset_at = budget_reset_at + INTERVAL '1 month'
                WHERE customer_id = $1
            """, customer_id)
            budget['current_spend_usd'] = 0
        
        # Check budget
        if budget['current_spend_usd'] >= budget['monthly_budget_usd']:
            # Cache resultado por 1 hora
            await self.redis.setex(cache_key, 3600, "exceeded")
            
            logger.warning(
                "customer_budget_exceeded",
                customer_id=customer_id,
                spend=budget['current_spend_usd'],
                limit=budget['monthly_budget_usd']
            )
            
            return False
        
        return True
    
    async def record_llm_cost(self, customer_id: str, cost_usd: float):
        """Registra custo de chamada LLM"""
        await self.db.execute("""
            UPDATE customer_budgets
            SET current_spend_usd = current_spend_usd + $1
            WHERE customer_id = $2
        """, cost_usd, customer_id)
        
        # Invalida cache
        await self.redis.delete(f"budget:{customer_id}")

budget_protection = BudgetProtection(db, redis)

# Integração com safe_llm_call
async def safe_llm_call_with_budget(
    func: Callable,
    customer_id: str,
    **kwargs
):
    """Wrapper que inclui budget check"""
    
    # Pre-flight budget check
    has_budget = await budget_protection.check_customer_budget(customer_id)
    if not has_budget:
        logger.error(
            "llm_call_blocked_budget",
            customer_id=customer_id
        )
        raise BudgetExceededError(
            f"Customer {customer_id} has exceeded monthly budget"
        )
    
    # Executa call
    result = await safe_llm_call_v2(func, **kwargs)
    
    # Estima custo
    cost = estimate_llm_cost(result)
    
    # Registra custo
    await budget_protection.record_llm_cost(customer_id, cost)
    
    return result
```

**DDL adicional:**

```sql
CREATE TABLE customer_budgets (
    customer_id TEXT PRIMARY KEY,
    monthly_budget_usd DECIMAL(10, 2) NOT NULL DEFAULT 100.00,
    current_spend_usd DECIMAL(10, 2) NOT NULL DEFAULT 0,
    budget_reset_at TIMESTAMPTZ NOT NULL DEFAULT date_trunc('month', NOW()) + INTERVAL '1 month',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_customer_budgets_reset ON customer_budgets(budget_reset_at);
```

**Owner**: Backend + SRE  
**Timeline**: 5-7 dias  
**Acceptance Criteria**:
- ✅ Rate limits: 10 msg/min/customer, 100 msg/hora/customer
- ✅ Agent rate limit: 50 comandos/min
- ✅ Global rate limit: 100 req/min/IP
- ✅ HTTP 429 com `Retry-After` header
- ✅ Budget check antes de cada LLM call
- ✅ Métricas: `rate_limit_exceeded_total{entity_type,limit_type}`

**Testes**:
```python
@pytest.mark.asyncio
async def test_customer_rate_limit():
    customer_id = "test_customer"
    
    # Envia 10 mensagens (limite)
    for i in range(10):
        response = await client.post("/webhook/message", json={
            "customer_id": customer_id,
            "content": f"Message {i}"
        })
        assert response.status_code == 200
    
    # 11ª mensagem deve ser rejeitada
    response = await client.post("/webhook/message", json={
        "customer_id": customer_id,
        "content": "Message 11"
    })
    assert response.status_code == 429
    assert "retry_after" in response.json()
    assert "Retry-After" in response.headers

@pytest.mark.asyncio
async def test_budget_protection():
    customer_id = "test_customer_budget"
    
    # Setup: define budget baixo
    await db.execute("""
        INSERT INTO customer_budgets (customer_id, monthly_budget_usd, current_spend_usd)
        VALUES ($1, 10.00, 9.50)
    """, customer_id)
    
    # Tenta LLM call que excederia budget
    with pytest.raises(BudgetExceededError):
        await safe_llm_call_with_budget(
            lambda: some_expensive_call(),
            customer_id=customer_id
        )
```

**Rollback**: Feature flag `enforce_rate_limits`. Se false, apenas loga violations sem bloquear.

---

## P1.3 - Make CommandHandler Consistently Async

**Problema**: Mix de sync/async causa deadlocks no FastAPI.

**Ação**: Converter todos handlers para `async def` e garantir que bibliotecas síncronas rodem em executor.

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Executor para código síncrono
executor = ThreadPoolExecutor(max_workers=10)

class CommandHandler:
    def __init__(self, db, langgraph_agent, composio_toolset):
        self.db = db
        self.agent = langgraph_agent
        self.composio = composio_toolset
        self.executor = executor
        
        # Agents PydanticAI (async)
        self.summary_agent = SummaryAgent()
        self.analysis_agent = AnalysisAgent()
        self.suggestion_agent = SuggestionAgent()
    
    async def handle(self, conversation_id: str, command: str, args: dict) -> dict:
        """Handler principal - ASYNC"""
        handlers = {
            "@bot": self._handle_bot_takeover,
            "@pause": self._handle_bot_pause,
            "@resume": self._handle_bot_resume,
            "@summary": self._handle_summary,
            "@analyze": self._handle_analyze,
            # ... todos async
        }
        
        handler = handlers.get(command)
        if not handler:
            return {"error": f"Unknown command: {command}"}
        
        # Todos handlers são async agora
        return await handler(conversation_id, args)
    
    async def _handle_summary(self, conv_id: str, args: dict) -> dict:
        """ASYNC handler"""
        messages = await self.db.fetch_messages(conv_id)  # await
        
        # PydanticAI agent é async
        summary = await self.summary_agent.summarize(messages)  # await
        
        return {"success": True, "summary": summary.model_dump()}
    
    async def _handle_bot_takeover(self, conv_id: str, args: dict) -> dict:
        """ASYNC handler com Composio (síncrono)"""
        conv = await self.db.get_conversation(conv_id)
        
        # Se Composio SDK é síncrono, roda em executor
        result = await self._run_sync_composio_action(
            conv.customer_id,
            args.get("instruction")
        )
        
        return {"success": True, "bot_response": result}
    
    async def _run_sync_composio_action(self, customer_id: str, instruction: str):
        """Wrapper para rodar Composio síncrono em thread separada"""
        loop = asyncio.get_event_loop()
        
        def sync_call():
            # Código síncrono do Composio
            return self.composio.execute_action(
                action=Action.SOME_ACTION,
                params={"instruction": instruction},
                entity_id=customer_id
            )
        
        # Roda em ThreadPoolExecutor
        result = await loop.run_in_executor(self.executor, sync_call)
        return result

# Webhook handler - ASYNC
@app.post("/webhook/message")
async def receive_message(message: IncomingMessage, background_tasks: BackgroundTasks):
    """Webhook completamente async"""
    
    # Todas operações são await
    conv = await repo.get_or_create_conversation(message.conversation_id)
    
    if message.sender_type == 'human_agent':
        command, args = CommandParser.parse(message.content)
        
        if command:
            # CommandHandler.handle é async
            result = await command_handler.handle(
                message.conversation_id,
                command,
                args
            )
            
            # Envia resultado
            await send_to_agent(message.conversation_id, result)
        else:
            await handle_human_message(conv, message)
    
    return {"status": "received"}
```

**Owner**: Backend Engineer  
**Timeline**: 3-4 dias  
**Acceptance Criteria**:
- ✅ Todos métodos públicos são `async def`
- ✅ Nenhum `asyncio.run()` dentro de request handlers
- ✅ Código síncrono (Composio) roda em `run_in_executor`
- ✅ Profiling não mostra blocking no event loop

**Testes**:
```python
@pytest.mark.asyncio
async def test_all_handlers_are_async():
    handler = CommandHandler(mock_db, mock_agent, mock_composio)
    
    # Todos handlers devem ser coroutines
    for command in ["@bot", "@pause", "@summary", "@analyze"]:
        method = getattr(handler, f"_handle_{command.replace('@', '')}")
        assert asyncio.iscoroutinefunction(method)

@pytest.mark.asyncio
async def test_no_blocking_in_event_loop():
    """Verifica que não há blocking operations"""
    import time
    
    start = time.time()
    
    # Múltiplas operações concorrentes
    results = await asyncio.gather(
        command_handler.handle("conv1", "@summary", {}),
        command_handler.handle("conv2", "@analyze", {}),
        command_handler.handle("conv3", "@status", {})
    )
    
    elapsed = time.time() - start
    
    # Se fosse síncrono, levaria 3x o tempo
    # Async deve ser ~1x (paralelo)
    assert elapsed < 2.0  # Ajustar threshold baseado em operações reais
```

**Rollback**: N/A (mudança de código, não config).

---

## P1.4 - Timeout Monitor com Redis Sorted Set (CORRIGIDO)

**Problema**: Keyspace notifications não são garantidas e perdem eventos em restart.

**Solução**: Usar Sorted Set + worker polling (mais confiável).

```python
import asyncio
from datetime import datetime, timedelta
import redis.asyncio as aioredis

class TimeoutMonitor:
    def __init__(self, redis_client: aioredis.Redis, db_pool, summary_agent):
        self.redis = redis_client
        self.db = db_pool
        self.summary_agent = summary_agent
    
    async def register_human_takeover(
        self, 
        conversation_id: str,
        timeout_minutes: int = 30
    ):
        """Registra timeout quando humano assume conversa"""
        expire_at = time.time() + (timeout_minutes * 60)
        
        # Adiciona à sorted set (score = expire timestamp)
        await self.redis.zadd(
            "timeout_queue",
            {conversation_id: expire_at}
        )
        
        logger.info(
            "timeout_registered",
            conversation_id=conversation_id,
            expires_at=datetime.fromtimestamp(expire_at).isoformat()
        )
    
    async def cancel_timeout(self, conversation_id: str):
        """Cancela timeout (quando bot assume ou humano responde)"""
        removed = await self.redis.zrem("timeout_queue", conversation_id)
        
        if removed:
            logger.info(
                "timeout_cancelled",
                conversation_id=conversation_id
            )
    
    async def handle_expired_conversation(self, conversation_id: str):
        """Processa conversa expirada por timeout"""
        
        # Adquire lock distribuído (evita duplicação em múltiplos workers)
        lock_key = f"lock:timeout:{conversation_id}"
        lock = await self.redis.set(lock_key, "1", nx=True, ex=60)
        
        if not lock:
            logger.debug(
                "timeout_already_being_processed",
                conversation_id=conversation_id
            )
            return
        
        try:
            # Verifica se ainda está em HIL
            conv = await self.db.fetchrow("""
                SELECT * FROM conversations 
                WHERE id = $1 AND is_hil = true
            """, conversation_id)
            
            if not conv:
                logger.info(
                    "timeout_conversation_already_closed",
                    conversation_id=conversation_id
                )
                return
            
            # Gera summary antes de fechar
            messages = await self.db.fetch("""
                SELECT * FROM messages 
                WHERE conversation_id = $1
                ORDER BY timestamp ASC
            """, conversation_id)
            
            message_dicts = [
                {
                    "sender_type": m['sender_type'],
                    "content": m['content'],
                    "timestamp": m['timestamp']
                }
                for m in messages
            ]
            
            summary = await self.summary_agent.summarize(message_dicts)
            
            # Envia mensagem de encerramento ao customer
            await send_to_customer(
                conv['customer_id'],
                "Esta conversa foi encerrada por inatividade. "
                "Se precisar de ajuda, inicie uma nova conversa."
            )
            
            # Atualiza conversa
            await self.db.execute("""
                UPDATE conversations
                SET 
                    is_hil = false,
                    closed = true,
                    closed_at = NOW(),
                    close_reason = 'timeout',
                    summary = $1
                WHERE id = $2
            """, summary.model_dump_json(), conversation_id)
            
            # Registra no audit log
            await self.db.execute("""
                INSERT INTO conversation_audit
                (conversation_id, actor, action, old_is_hil, new_is_hil, reason, details)
                VALUES ($1, 'system', 'timeout_closed', true, false, 'inactivity_timeout', $2)
            """, conversation_id, summary.model_dump_json())
            
            # Notifica human agent
            await notify_human_agent(
                conversation_id,
                notification_type="timeout_closed",
                summary=summary
            )
            
            logger.info(
                "timeout_conversation_closed",
                conversation_id=conversation_id,
                summary=summary.model_dump()
            )
            
            # Remove da fila
            await self.redis.zrem("timeout_queue", conversation_id)
            
        finally:
            # Libera lock
            await self.redis.delete(lock_key)
    
    async def worker_loop(self, poll_interval_seconds: int = 10):
        """
        Worker loop principal que processa timeouts.
        Roda continuamente checkando sorted set.
        """
        logger.info("timeout_monitor_started", poll_interval=poll_interval_seconds)
        
        while True:
            try:
                now = time.time()
                
                # Busca conversas expiradas (score <= now)
                expired = await self.redis.zrangebyscore(
                    "timeout_queue",
                    min='-inf',
                    max=now,
                    start=0,
                    num=10  # Processa max 10 por vez (backpressure)
                )
                
                if expired:
                    logger.info(
                        "timeout_processing_batch",
                        count=len(expired)
                    )
                    
                    # Processa cada conversa expirada
                    for conv_id in expired:
                        try:
                            await self.handle_expired_conversation(conv_id.decode())
                        except Exception as e:
                            logger.error(
                                "timeout_processing_error",
                                conversation_id=conv_id.decode(),
                                error=str(e),
                                exc_info=True
                            )
                
                # Aguarda antes do próximo poll
                await asyncio.sleep(poll_interval_seconds)
                
            except Exception as e:
                logger.error(
                    "timeout_worker_loop_error",
                    error=str(e),
                    exc_info=True
                )
                # Aguarda um pouco antes de retry
                await asyncio.sleep(30)

# Inicialização do worker
timeout_monitor = TimeoutMonitor(redis, db, SummaryAgent())

@app.on_event("startup")
async def start_timeout_worker():
    """Inicia timeout worker em background"""
    asyncio.create_task(timeout_monitor.worker_loop(poll_interval_seconds=10))
    logger.info("timeout_worker_started")

# Integração com HIL transitions
async def handle_human_message(conv, message):
    """Quando humano envia mensagem"""
    if not message.content.startswith('@'):
        # Humano assume controle
        await repo.update_is_hil(conv.id, True, f"agent_{message.sender_id}", "manual_takeover")
        
        # Registra timeout
        await timeout_monitor.register_human_takeover(
            conv.id,
            timeout_minutes=HUMAN_TIMEOUT_MINUTES
        )
    else:
        # É um comando
        command, args = CommandParser.parse(message.content)
        
        if command == "@bot":
            # Bot assume - cancela timeout
            await timeout_monitor.cancel_timeout(conv.id)

async def handle_customer_message(conv, message):
    """Quando customer envia mensagem"""
    if conv.is_hil:
        # Humano está no controle: renova timeout
        await timeout_monitor.register_human_takeover(
            conv.id,
            timeout_minutes=HUMAN_TIMEOUT_MINUTES
        )
```

**Owner**: Backend + SRE  
**Timeline**: 4-5 dias  
**Acceptance Criteria**:
- ✅ Timeouts sobrevivem a restarts (sorted set persiste)
- ✅ Lock distribuído evita processamento duplicado
- ✅ Worker processa max 10 timeouts por batch (backpressure)
- ✅ Métricas: `timeout_processed_total`, `timeout_errors_total`
- ✅ Timeout configurável via env var `HUMAN_TIMEOUT_MINUTES`

**Testes**:
```python
@pytest.mark.asyncio
async def test_timeout_registration():
    conv_id = "test_conv"
    
    await timeout_monitor.register_human_takeover(conv_id, timeout_minutes=1)
    
    # Verifica que está na sorted set
    score = await redis.zscore("timeout_queue", conv_id)
    assert score is not None
    
    # Score deve ser ~60 segundos no futuro
    expected_time = time.time() + 60
    assert abs(score - expected_time) < 5  # Margem de 5s

@pytest.mark.asyncio
async def test_timeout_expiration():
    conv_id = await create_test_conversation(is_hil=True)
    
    # Registra timeout de 1 segundo (para teste rápido)
    await timeout_monitor.register_human_takeover(conv_id, timeout_minutes=0.0166)  # ~1s
    
    # Aguarda expiração
    await asyncio.sleep(2)
    
    # Força processamento
    await timeout_monitor.handle_expired_conversation(conv_id)
    
    # Verifica: conversa fechada
    conv = await db.fetchrow("SELECT * FROM conversations WHERE id = $1", conv_id)
    assert conv['closed'] == True
    assert conv['close_reason'] == 'timeout'
    
    # Verifica: removido da fila
    score = await redis.zscore("timeout_queue", conv_id)
    assert score is None

@pytest.mark.asyncio
async def test_timeout_cancellation():
    conv_id = "test_conv"
    
    await timeout_monitor.register_human_takeover(conv_id, timeout_minutes=30)
    
    # Cancela timeout
    await timeout_monitor.cancel_timeout(conv_id)
    
    # Verifica: removido da fila
    score = await redis.zscore("timeout_queue", conv_id)
    assert score is None

@pytest.mark.asyncio
async def test_timeout_distributed_lock():
    """Testa que múltiplos workers não processam mesma conversa"""
    conv_id = await create_test_conversation(is_hil=True)
    await redis.zadd("timeout_queue", {conv_id: time.time() - 1})  # Já expirado
    
    # Simula 2 workers tentando processar simultaneamente
    async def worker():
        await timeout_monitor.handle_expired_conversation(conv_id)
    
    results = await asyncio.gather(
        worker(),
        worker(),
        return_exceptions=True
    )
    
    # Verifica: conversa processada apenas 1 vez
    audit_entries = await db.fetch("""
        SELECT * FROM conversation_audit 
        WHERE conversation_id = $1 AND action = 'timeout_closed'
    """, conv_id)
    
    assert len(audit_entries) == 1  # Apenas uma entrada de timeout
```

**Rollback**: Feature flag `use_redis_timeout_monitor`. Se false, usa cron job original (menos confiável mas funcional).

---

## P1.5 - Observability: Trace/Log Correlation + Metric Hygiene

**Problema**: Logs sem trace IDs. Métricas com high-cardinality labels.

**Solução**: OpenTelemetry + structlog integration + metric cleanup.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
import structlog

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
    agent_port=int(os.getenv("JAEGER_PORT", 6831)),
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrumenta FastAPI automaticamente
FastAPIInstrumentor.instrument_app(app)

# Instrumenta AsyncPG
AsyncPGInstrumentor().instrument()

# Structlog processor que adiciona trace context
def add_trace_context(logger, method_name, event_dict):
    """Adiciona trace_id e span_id aos logs"""
    span = trace.get_current_span()
    if span:
        span_context = span.get_span_context()
        if span_context.is_valid:
            event_dict['trace_id'] = format(span_context.trace_id, '032x')
            event_dict['span_id'] = format(span_context.span_id, '016x')
    return event_dict

# Configura structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_trace_context,  # Adiciona trace context
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Exemplo de uso com traces
async def process_message(message: IncomingMessage):
    """Processa mensagem com tracing automático"""
    
    # FastAPI já criou span automaticamente
    # Adiciona atributos ao span atual
    span = trace.get_current_span()
    span.set_attribute("conversation_id", message.conversation_id)
    span.set_attribute("sender_type", message.sender_type)
    
    # Logs automaticamente incluem trace_id
    logger.info(
        "message_processing_started",
        conversation_id=message.conversation_id,
        sender_type=message.sender_type
        # trace_id e span_id adicionados automaticamente
    )
    
    # Child spans para operações específicas
    with tracer.start_as_current_span("command_parsing") as parse_span:
        command, args = CommandParser.parse(message.content)
        parse_span.set_attribute("command", command or "none")
    
    if command:
        with tracer.start_as_current_span("command_execution") as cmd_span:
            cmd_span.set_attribute("command", command)
            result = await command_handler.handle(
                message.conversation_id,
                command,
                args
            )
    
    logger.info(
        "message_processing_completed",
        conversation_id=message.conversation_id
        # Mesmo trace_id
    )

# Métricas com cardinality controlada
class MetricsRegistry:
    """Registry centralizado com validação de cardinality"""
    
    # Enums permitidos para labels
    ALLOWED_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
    ALLOWED_OUTCOMES = ["success", "error", "timeout", "rate_limit", "circuit_open"]
    ALLOWED_COMMANDS = ["bot", "pause", "resume", "summary", "analyze", "suggest", "draft", "status", "help"]
    ALLOWED_ESCALATION_REASONS = ["timeout", "client_request", "bot_failure", "policy_violation", "manual"]
    
    def __init__(self):
        # LLM metrics
        self.llm_calls_total = Counter(
            'llm_calls_total',
            'Total LLM API calls',
            ['model', 'outcome']  # Bounded cardinality
        )
        
        self.llm_call_duration_seconds = Histogram(
            'llm_call_duration_seconds',
            'LLM call duration',
            ['model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.llm_tokens_total = Counter(
            'llm_tokens_total',
            'Total tokens consumed',
            ['model', 'type']  # type = 'prompt' | 'completion'
        )
        
        self.llm_cost_usd_total = Counter(
            'llm_cost_usd_total',
            'Total estimated LLM cost in USD',
            ['model']
        )
        
        # Command metrics
        self.commands_executed_total = Counter(
            'commands_executed_total',
            'Total commands executed',
            ['command', 'outcome']  # NO user_id, NO free-text
        )
        
        # Escalation metrics
        self.escalations_total = Counter(
            'escalations_total',
            'Total escalations to human',
            ['reason']  # Enumerated reasons only
        )
        
        # Conversation metrics
        self.conversations_active = Gauge(
            'conversations_active',
            'Currently active conversations'
        )
        
        self.conversations_bot_active = Gauge(
            'conversations_bot_active',
            'Conversations with bot active (is_hil=false)'
        )
        
        self.conversations_human_active = Gauge(
            'conversations_human_active',
            'Conversations with human active (is_hil=true)'
        )
        
        # Rate limit metrics
        self.rate_limit_exceeded_total = Counter(
            'rate_limit_exceeded_total',
            'Rate limit violations',
            ['entity_type', 'limit_type']  # entity_type = 'customer' | 'agent'
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['model']
        )
    
    def record_llm_call(
        self,
        model: str,
        outcome: str,
        duration: float,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        cost_usd: float = 0.0
    ):
        """Record LLM call com validação de cardinality"""
        
        # Valida model
        if model not in self.ALLOWED_MODELS:
            logger.warning("unknown_model_in_metrics", model=model)
            model = "unknown"
        
        # Valida outcome
        if outcome not in self.ALLOWED_OUTCOMES:
            logger.warning("unknown_outcome_in_metrics", outcome=outcome)
            outcome = "unknown"
        
        self.llm_calls_total.labels(model=model, outcome=outcome).inc()
        self.llm_call_duration_seconds.labels(model=model).observe(duration)
        
        if tokens_prompt > 0:
            self.llm_tokens_total.labels(model=model, type='prompt').inc(tokens_prompt)
        if tokens_completion > 0:
            self.llm_tokens_total.labels(model=model, type='completion').inc(tokens_completion)
        if cost_usd > 0:
            self.llm_cost_usd_total.labels(model=model).inc(cost_usd)
    
    def record_command(self, command: str, outcome: str):
        """Record command com validação"""
        # Remove '@' prefix
        command = command.lstrip('@')
        
        if command not in self.ALLOWED_COMMANDS:
            logger.warning("unknown_command_in_metrics", command=command)
            command = "unknown"
        
        self.commands_executed_total.labels(command=command, outcome=outcome).inc()
    
    def record_escalation(self, reason: str):
        """Record escalation com enum"""
        if reason not in self.ALLOWED_ESCALATION_REASONS:
            logger.warning("unknown_escalation_reason", reason=reason)
            reason = "other"
        
        self.escalations_total.labels(reason=reason).inc()

metrics = MetricsRegistry()

# Prometheus endpoint (non-blocking ASGI app)
from prometheus_client import make_asgi_app

# Monta sub-app para métricas
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Job que atualiza gauges periodicamente
async def update_gauge_metrics():
    """Background job que atualiza métricas de gauge"""
    while True:
        try:
            # Conta conversas ativas
            counts = await db.fetchrow("""
                SELECT 
                    COUNT(*) FILTER (WHERE NOT closed) as active,
                    COUNT(*) FILTER (WHERE is_hil = false AND NOT closed) as bot_active,
                    COUNT(*) FILTER (WHERE is_hil = true AND NOT closed) as human_active
                FROM conversations
            """)
            
            metrics.conversations_active.set(counts['active'])
            metrics.conversations_bot_active.set(counts['bot_active'])
            metrics.conversations_human_active.set(counts['human_active'])
            
        except Exception as e:
            logger.error("gauge_update_error", error=str(e), exc_info=True)
        
        await asyncio.sleep(30)  # Atualiza a cada 30s

@app.on_event("startup")
async def start_gauge_updater():
    asyncio.create_task(update_gauge_metrics())
```

**Grafana Dashboard JSON** (exemplo):

```json
{
  "dashboard": {
    "title": "HIL System Metrics",
    "panels": [
      {
        "title": "LLM Call Success Rate",
        "targets": [{
          "expr": "rate(llm_calls_total{outcome='success'}[5m]) / rate(llm_calls_total[5m])"
        }],
        "type": "graph"
      },
      {
        "title": "LLM Cost (USD/hour)",
        "targets": [{
          "expr": "rate(llm_cost_usd_total[1h])"
        }],
        "type": "graph"
      },
      {
        "title": "P95 LLM Latency",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(llm_call_duration_seconds_bucket[5m]))"
        }],
        "type": "graph"
      },
      {
        "title": "Active Conversations",
        "targets": [
          {"expr": "conversations_active", "legendFormat": "Total"},
          {"expr": "conversations_bot_active", "legendFormat": "Bot"},
          {"expr": "conversations_human_active", "legendFormat": "Human"}
        ],
        "type": "graph"
      },
      {
        "title": "Escalation Rate",
        "targets": [{
          "expr": "sum by (reason) (rate(escalations_total[5m]))"
        }],
        "type": "graph"
      }
    ]
  }
}
```

**Owner**: Observability + Backend  
**Timeline**: 5-7 dias  
**Acceptance Criteria**:
- ✅ Logs incluem `trace_id` e `span_id`
- ✅ Nenhuma métrica com cardinality >50 valores por label
- ✅ Jaeger mostra traces end-to-end (webhook → LLM → DB)
- ✅ Grafana dashboard funcional com LLM cost, latency, escalations
- ✅ Prometheus `/metrics` endpoint responde <100ms

**Testes**:
```python
def test_logs_include_trace_context(caplog):
    """Verifica que logs incluem trace_id"""
    with tracer.start_as_current_span("test_span"):
        logger.info("test_message", foo="bar")
    
    log_output = caplog.records[0].__dict__
    assert 'trace_id' in log_output
    assert 'span_id' in log_output
    assert len(log_output['trace_id']) == 32  # hex format

def test_metrics_cardinality():
    """Verifica que métricas não têm high-cardinality"""
    from prometheus_client import REGISTRY
    
    for collector in REGISTRY._collector_to_names.keys():
        for metric in collector.collect():
            for sample in metric.samples:
                for label_value in sample.labels.values():
                    # Verifica que não são UUIDs ou free-text
                    assert not re.match(r'[0-9a-f-]{36}', label_value)  # UUID
                    assert len(label_value) < 50  # Não é free-text longo
```

**Rollback**: Feature flag `enable_tracing`. Se false, apenas loga sem traces (graceful degradation).

---

## P1.6 - Error Handling & Graceful Degradation

**Problema**: Agents assumem sucesso. Falhas causam crashes ou escalações sem contexto.

**Solução**: Taxonomy de erros + fallback behavior + user-friendly messaging.

```python
from enum import Enum
from typing import Optional

class ErrorCategory(Enum):
    TRANSIENT = "transient"  # Retry pode resolver
    RATE_LIMIT = "rate_limit"  # Aguardar e retry
    PERMANENT = "permanent"  # Não adianta retry
    BUDGET_EXCEEDED = "budget_exceeded"  # Customer sem budget

# Error taxonomy
class TransientError(Exception):
    """Erro transitório, retry recomendado"""
    pass

class RateLimitError(Exception):
    """Rate limit atingido"""
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after

class PermanentFailure(Exception):
    """Falha permanente, não retry"""
    pass

class BudgetExceededError(Exception):
    """Budget do customer esgotado"""
    pass

class CircuitBreakerOpenError(Exception):
    """Circuit breaker aberto"""
    pass

# Mapeamento de erros HTTP
def categorize_http_error(status_code: int) -> ErrorCategory:
    """Categoriza erro HTTP"""
    TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
    PERMANENT_CODES = {400, 401, 403, 404, 422}
    
    if status_code in TRANSIENT_CODES:
        return ErrorCategory.TRANSIENT
    elif status_code in PERMANENT_CODES:
        return ErrorCategory.PERMANENT
    elif status_code == 429:
        return ErrorCategory.RATE_LIMIT
    else:
        return ErrorCategory.PERMANENT  # Assume permanent

# Graceful degradation handler
class GracefulErrorHandler:
    def __init__(self, db_pool, notification_service):
        self.db = db_pool
        self.notifications = notification_service
    
    async def handle_llm_failure(
        self,
        conversation_id: str,
        error: Exception,
        context: dict
    ) -> dict:
        """Handle LLM failures gracefully"""
        
        # Determina categoria
        if isinstance(error, TransientError):
            category = ErrorCategory.TRANSIENT
        elif isinstance(error, RateLimitError):
            category = ErrorCategory.RATE_LIMIT
        elif isinstance(error, BudgetExceededError):
            category = ErrorCategory.BUDGET_EXCEEDED
        elif isinstance(error, CircuitBreakerOpenError):
            category = ErrorCategory.TRANSIENT
        else:
            category = ErrorCategory.PERMANENT
        
        logger.error(
            "llm_failure",
            conversation_id=conversation_id,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context
        )
        
        await self.db.execute("""
            UPDATE conversations
            SET is_hil = true
            WHERE id = $1
        """, conversation_id)
        
        await self.notifications.notify_agent(
            conversation_id=conversation_id,
            priority="critical",
            message=f"Bot failure (permanent). Ticket #{ticket['id']} created.",
            ticket_id=ticket['id']
        )
        
        await send_to_customer(
            await self._get_customer_id(conversation_id),
            "Encontramos um problema técnico. "
            "Um especialista irá ajudá-lo em breve."
        )
        
        return {
            "escalated": True,
            "reason": "permanent_failure",
            "ticket_id": ticket['id']
        }
    
    async def _create_support_ticket(
        self,
        conversation_id: str,
        error_type: str,
        error_message: str,
        context: dict
    ) -> dict:
        """Cria ticket de suporte interno"""
        ticket_id = await self.db.fetchval("""
            INSERT INTO support_tickets
            (conversation_id, error_type, error_message, context, status, created_at)
            VALUES ($1, $2, $3, $4, 'open', NOW())
            RETURNING id
        """, conversation_id, error_type, error_message, context)
        
        return {"id": ticket_id}
    
    async def _get_customer_id(self, conversation_id: str) -> str:
        return await self.db.fetchval(
            "SELECT customer_id FROM conversations WHERE id = $1",
            conversation_id
        )

error_handler = GracefulErrorHandler(db, notification_service)

# Integração com safe_llm_call
async def safe_llm_call_with_error_handling(
    func: Callable,
    conversation_id: str,
    context: dict,
    **kwargs
):
    """Wrapper completo: timeout + retry + circuit breaker + error handling"""
    
    try:
        result = await safe_llm_call_v2(func, **kwargs)
        return result
    
    except (TransientError, CircuitBreakerOpenError, asyncio.TimeoutError) as e:
        # Transient: tenta error handler
        return await error_handler.handle_llm_failure(
            conversation_id,
            e,
            context
        )
    
    except RateLimitError as e:
        return await error_handler.handle_llm_failure(
            conversation_id,
            e,
            context
        )
    
    except BudgetExceededError as e:
        return await error_handler.handle_llm_failure(
            conversation_id,
            e,
            context
        )
    
    except Exception as e:
        # Permanent ou unknown: escala
        logger.error(
            "unexpected_llm_error",
            conversation_id=conversation_id,
            error_type=type(e).__name__,
            exc_info=True
        )
        
        return await error_handler.handle_llm_failure(
            conversation_id,
            e,
            context
        )
```

**DDL adicional:**

```sql
CREATE TABLE support_tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    error_type TEXT NOT NULL,
    error_message TEXT,
    context JSONB,
    status TEXT DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'resolved', 'closed')),
    assigned_to TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX idx_support_tickets_status ON support_tickets(status, created_at DESC);
CREATE INDEX idx_support_tickets_conversation ON support_tickets(conversation_id);
```

**Owner**: Backend Engineer  
**Timeline**: 5-7 dias  
**Acceptance Criteria**:
- ✅ Todos erros externos são categorizados (transient/permanent/rate_limit)
- ✅ Transient errors escalation com retry automático
- ✅ User-facing messages são amigáveis (sem stack traces)
- ✅ Support tickets criados para permanent failures
- ✅ Métricas: `error_handled_total{category}`, `tickets_created_total`

**Testes**:
```python
@pytest.mark.asyncio
async def test_transient_error_escalation():
    conv_id = await create_test_conversation(is_hil=False)
    
    error = TransientError("Temporary service unavailable")
    result = await error_handler.handle_llm_failure(
        conv_id,
        error,
        {"command": "@summary"}
    )
    
    assert result["escalated"] == True
    assert result["reason"] == "transient_error"
    
    # Verifica escalação no DB
    conv = await db.fetchrow("SELECT * FROM conversations WHERE id = $1", conv_id)
    assert conv['is_hil'] == True

@pytest.mark.asyncio
async def test_permanent_error_creates_ticket():
    conv_id = await create_test_conversation()
    
    error = PermanentFailure("Invalid API key")
    result = await error_handler.handle_llm_failure(
        conv_id,
        error,
        {"agent": "summary"}
    )
    
    assert result["escalated"] == True
    assert "ticket_id" in result
    
    # Verifica ticket criado
    ticket = await db.fetchrow(
        "SELECT * FROM support_tickets WHERE id = $1",
        result["ticket_id"]
    )
    assert ticket is not None
    assert ticket['status'] == 'open'
    assert ticket['error_type'] == 'PermanentFailure'
```

**Rollback**: Feature flag `enable_error_handling`. Se false, propaga exceptions (desenvolvimento).

---

# P2 — Médio (importante mas pode lançar com mitigações)

## P2.1 - Secrets & Infrastructure Hardening

**Problema**: `.env` exposto, Docker compose sem healthchecks.

```python
# secrets_manager.py
import os
from abc import ABC, abstractmethod

class SecretsManager(ABC):
    @abstractmethod
    async def get_secret(self, key: str) -> str:
        pass

class EnvSecretsManager(SecretsManager):
    """Dev/Test: usa .env"""
    async def get_secret(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Secret {key} not found in environment")
        return value

class VaultSecretsManager(SecretsManager):
    """Production: usa HashiCorp Vault"""
    def __init__(self, vault_addr: str, vault_token: str):
        import hvac
        self.client = hvac.Client(url=vault_addr, token=vault_token)
    
    async def get_secret(self, key: str) -> str:
        secret = self.client.secrets.kv.v2.read_secret_version(
            path=key,
            mount_point='secret'
        )
        return secret['data']['data']['value']

# Factory
def get_secrets_manager() -> SecretsManager:
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return VaultSecretsManager(
            vault_addr=os.getenv("VAULT_ADDR"),
            vault_token=os.getenv("VAULT_TOKEN")
        )
    else:
        return EnvSecretsManager()

secrets = get_secrets_manager()

# Uso
OPENAI_API_KEY = await secrets.get_secret("OPENAI_API_KEY")
DATABASE_URL = await secrets.get_secret("DATABASE_URL")
```

**docker-compose.yml hardened:**

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - VAULT_ADDR=${VAULT_ADDR}
      - VAULT_TOKEN=${VAULT_TOKEN}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - internal
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=conversations
      - POSTGRES_USER_FILE=/run/secrets/db_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_user
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - internal
    ports: []  # Não expõe externamente
  
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    networks:
      - internal
    ports: []  # Interno apenas
    command: redis-server --requirepass ${REDIS_PASSWORD}

networks:
  internal:
    driver: bridge

secrets:
  db_user:
    file: ./secrets/db_user.txt
  db_password:
    file: ./secrets/db_password.txt

volumes:
  postgres_data:
```

**Owner**: SRE  
**Timeline**: 3-5 dias  
**Acceptance**: Secrets não em plaintext, healthchecks funcionais

---

## P2.2 - Unit Test & CI Improvements

**Problema**: Testes chamam APIs reais (GPT-4).

```python
# conftest.py
import pytest
from unittest.mock import AsyncMock, Mock

@pytest.fixture
def mock_llm_agent():
    """Mock PydanticAI agent"""
    agent = AsyncMock()
    agent.summarize = AsyncMock(return_value=ConversationSummary(
        main_topic="Test topic",
        customer_issue="Test issue",
        current_status="Test status",
        next_steps=["Step 1", "Step 2"],
        sentiment_score=0.5,
        urgency="medium"
    ))
    agent.analyze = AsyncMock(return_value=ConversationAnalysis(
        sentiment="neutral",
        urgency="medium",
        complexity="simple",
        recommended_actions=["Test action"],
        bot_can_handle=True,
        escalation_reason=None
    ))
    return agent

@pytest.fixture
def mock_composio():
    """Mock Composio toolset"""
    toolset = Mock()
    toolset.execute_action = Mock(return_value={"status": "success"})
    return toolset

@pytest.fixture
def mock_db(mocker):
    """Mock database"""
    db = AsyncMock()
    db.fetchrow = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.execute = AsyncMock()
    db.fetchval = AsyncMock(return_value=1)
    return db

# Testes sem API calls
@pytest.mark.asyncio
async def test_summary_agent_structure(mock_llm_agent):
    """Testa estrutura do output sem chamar OpenAI"""
    result = await mock_llm_agent.summarize([])
    
    assert isinstance(result, ConversationSummary)
    assert result.urgency in ["low", "medium", "high"]
    assert -1 <= result.sentiment_score <= 1

# CI Configuration (.github/workflows/test.yml)
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      
      - name: Run tests (without API keys)
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost/test
          REDIS_URL: redis://localhost:6379
          # NO OPENAI_API_KEY - testes devem usar mocks
        run: |
          pytest --cov=. --cov-report=xml -v
      
      - name: Check PII in logs
        run: |
          # Scan logs para PII patterns
          python scripts/check_pii.py
      
      - name: Check metric cardinality
        run: |
          # Valida que métricas não têm high-cardinality
          python scripts/check_metrics.py
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**scripts/check_pii.py:**

```python
import re
import sys

PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'cpf': r'\b\d{3}\.?\d{3}\.?\d{3}[-]?\d{2}\b',
}

def scan_logs_for_pii(log_dir='./logs'):
    violations = []
    
    for log_file in Path(log_dir).glob('*.log'):
        with open(log_file) as f:
            for line_num, line in enumerate(f, 1):
                for pii_type, pattern in PII_PATTERNS.items():
                    if re.search(pattern, line):
                        violations.append({
                            'file': log_file.name,
                            'line': line_num,
                            'type': pii_type,
                            'content': line.strip()[:100]
                        })
    
    if violations:
        print("❌ PII FOUND IN LOGS:")
        for v in violations:
            print(f"  {v['file']}:{v['line']} - {v['type']}")
        sys.exit(1)
    
    print("✅ No PII detected in logs")

if __name__ == '__main__':
    scan_logs_for_pii()
```

**Owner**: QA Engineer  
**Timeline**: 5-7 dias  
**Acceptance**: CI passa sem API keys, PII scanner integrado

---

## P2.3 - Command Authorization (RBAC)

**Problema**: Qualquer agent pode executar qualquer comando.

```python
from enum import Enum

class AgentRole(Enum):
    OPERATOR = "operator"  # Atendentes básicos
    SUPERVISOR = "supervisor"  # Supervisores
    ADMIN = "admin"  # Administradores

# Permissions matrix
COMMAND_PERMISSIONS = {
    "@bot": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@pause": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@resume": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@summary": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@analyze": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@suggest": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@draft": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@status": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@help": [AgentRole.OPERATOR, AgentRole.SUPERVISOR, AgentRole.ADMIN],
    
    # Supervisor+
    "@escalate": [AgentRole.SUPERVISOR, AgentRole.ADMIN],
    "@transfer": [AgentRole.SUPERVISOR, AgentRole.ADMIN],
    
    # Admin only
    "@close": [AgentRole.ADMIN],
    "@export": [AgentRole.ADMIN],
}

class RBACService:
    def __init__(self, db_pool):
        self.db = db_pool
    
    async def get_agent_role(self, agent_id: str) -> AgentRole:
        """Busca role do agent"""
        role_str = await self.db.fetchval("""
            SELECT role FROM agents WHERE id = $1
        """, agent_id)
        
        if not role_str:
            return AgentRole.OPERATOR  # Default
        
        return AgentRole(role_str)
    
    async def check_permission(
        self,
        agent_id: str,
        command: str
    ) -> bool:
        """Verifica se agent tem permissão para comando"""
        agent_role = await self.get_agent_role(agent_id)
        allowed_roles = COMMAND_PERMISSIONS.get(command, [])
        
        return agent_role in allowed_roles
    
    async def audit_authorization_attempt(
        self,
        agent_id: str,
        command: str,
        allowed: bool,
        conversation_id: str
    ):
        """Registra tentativas de autorização"""
        await self.db.execute("""
            INSERT INTO authorization_audit
            (agent_id, command, allowed, conversation_id, attempted_at)
            VALUES ($1, $2, $3, $4, NOW())
        """, agent_id, command, allowed, conversation_id)

rbac = RBACService(db)

# Integração no CommandHandler
async def handle_command_with_rbac(
    conversation_id: str,
    command: str,
    args: dict,
    agent_id: str
):
    """Handler com RBAC"""
    
    # Check permission
    has_permission = await rbac.check_permission(agent_id, command)
    
    # Audit tentativa
    await rbac.audit_authorization_attempt(
        agent_id,
        command,
        has_permission,
        conversation_id
    )
    
    if not has_permission:
        logger.warning(
            "unauthorized_command_attempt",
            agent_id=agent_id,
            command=command,
            conversation_id=conversation_id
        )
        
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Forbidden",
                "message": f"You don't have permission to execute {command}"
            }
        )
    
    # Executa comando
    return await command_handler.handle(conversation_id, command, args)
```

**DDL:**

```sql
CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('operator', 'supervisor', 'admin')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE authorization_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id TEXT NOT NULL,
    command TEXT NOT NULL,
    allowed BOOLEAN NOT NULL,
    conversation_id UUID,
    attempted_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_auth_audit_agent ON authorization_audit(agent_id, attempted_at DESC);
CREATE INDEX idx_auth_audit_denied ON authorization_audit(allowed) WHERE allowed = false;
```

**Owner**: Backend + Product  
**Timeline**: 4-6 dias  
**Acceptance**: Unauthorized attempts retornam 403, audit log registra

---

# P3 — Hardening (pós-lançamento)

## P3.1 - Cost & Budget Monitoring Dashboard

**Ação**: Dashboard Grafana com alertas de custo LLM, budget tracking por customer.

## P3.2 - LangGraph Node Idempotency

**Ação**: Garantir que Composio actions com side-effects (refunds, tickets) sejam idempotentes ou tenham compensation logic.

## P3.3 - Advanced PII Detection

**Ação**: Integrar NER model (spaCy) + LLM-based PII check para casos edge.

---

# Validation & Monitoring Plan

## Alertas Críticos (P0)

```yaml
# prometheus_alerts.yml
groups:
  - name: hil_critical
    rules:
      - alert: LLMErrorRateHigh
        expr: rate(llm_calls_total{outcome="error"}[5m]) / rate(llm_calls_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM error rate above 10%"
      
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker open for {{ $labels.model }}"
      
      - alert: EscalationRateSpike
        expr: rate(escalations_total[5m]) > 2 * rate(escalations_total[1h] offset 1h)
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Escalation rate 2x baseline"
      
      - alert: BudgetExceeded
        expr: sum by (customer_id) (customer_budget_usage_pct) > 0.9
        labels:
          severity: warning
        annotations:
          summary: "Customer {{ $labels.customer_id }} at 90% budget"
```

## Tests e Acceptance (Consolidado)

**P0 Acceptance:**
- ✅ 100% LLM calls usam wrapper MVP (code audit)
- ✅ Zero lost updates sob concurrent load (stress test)
- ✅ Webhook dedupe: 0 duplicatas em 1000 retries
- ✅ Logs escaneados: 0 PII detectado (CI)
- ✅ GDPR deletion completa em <24h

**P1 Acceptance:**
- ✅ Rate limits enforced: 429 após threshold
- ✅ Traces end-to-end visíveis no Jaeger
- ✅ Métricas cardinality <50 por label
- ✅ Timeout worker processa expirations <30s
- ✅ Errors escalate gracefully com user-friendly messages

---

# Rollout Strategy

## Fase 1: Core Fixes (Semana 1-2)
- Deploy P0.1a (LLM wrapper MVP)
- Deploy P0.2 (Pessimistic locks)
- Deploy P0.3 (Idempotency)
- **Canary**: 10% do tráfego

## Fase 2: Security & Compliance (Semana 3-4)
- Deploy P0.4 (PII redaction)
- Deploy P1.1 (LGPD/GDPR)
- Deploy P1.2 (Rate limiting)
- **Canary**: 30% do tráfego

## Fase 3: Observability & Resilience (Semana 5-6)
- Deploy P1.4 (Timeout monitor)
- Deploy P1.5 (Tracing)
- Deploy P1.6 (Error handling)
- **Canary**: 50% → 100%

## Fase 4: Hardening (Semana 7+)
- Deploy P2.x items
- Monitor metrics for 2 weeks
- Implement P3 items incrementally

---

# Incident Response Playbook

## Scenario 1: LLM Cost Spike

**Detect**: `llm_cost_usd_total > $1000/day`

**Actions**:
1. Enable `circuit_breaker_strict=true` (opens after 3 failures)
2. Set all conversations `is_hil=true` via admin command
3. Check top customers by token usage
4. Enable per-customer budget caps
5. Postmortem: add preventive budget alerts

## Scenario 2: Escalation Loop

**Detect**: `escalations_total > 100/min`

**Actions**:
1. Disable bot auto-responses globally
2. Set `is_hil=true` for all active conversations
3. Review recent LangGraph changes (git log)
4. Check escalation reasons in audit log
5. Add circuit breaker to escalation logic

## Scenario 3: Database Lock Contention

**Detect**: `pg_stat_activity shows high lock waits`

**Actions**:
1. Check `conversation_audit` for high-frequency updates
2. Identify problematic conversation_id
3. Temporarily disable bot for that conversation
4. Analyze if optimistic locking would help
5. Consider sharding if single customer

---

# Summary Checklist

## P0 (Blocker) - MUST complete before production:
- [ ] P0.1a: LLM wrapper MVP (timeout + retry + metrics)
- [ ] P0.2: Pessimistic locking (`FOR UPDATE NOWAIT`)
- [ ] P0.3: Webhook idempotency (provider origin_id)
- [ ] P0.4: PII redaction (regex + audit trail)

## P1 (High) - Complete before wide rollout:
- [ ] P1.1: LGPD/GDPR compliance (deletion endpoint + retention)
- [ ] P1.2: Rate limiting (10/min/customer, budgets)
- [ ] P1.3: Async handlers (no blocking)
- [ ] P1.4: Timeout monitor (Redis sorted set)
- [ ] P1.5: Observability (traces + metric hygiene)
- [ ] P1.6: Error handling (graceful degradation)

## P2 (Medium) - Can launch with mitigations:
- [ ] P2.1: Secrets manager + hardened Docker
- [ ] P2.2: Unit tests without API calls + CI
- [ ] P2.3: RBAC on commands

## P3 (Low) - Post-launch hardening:
- [ ] P3.1: Cost monitoring dashboard
- [ ] P3.2: LangGraph idempotency
- [ ] P3.3: Advanced PII (NER + LLM check)

---

**Total Estimated Timeline**: 6-8 semanas para P0+P1 completo.

**Recommendation**: Priorize P0 items para MVP funcional, então itere P1 em sprints de 2 semanas.
            error_category=category.value,
            context=context,
            exc_info=True
        )
        
        # Estratégia de fallback
        if category == ErrorCategory.TRANSIENT:
            return await self._handle_transient_failure(conversation_id, error, context)
        elif category == ErrorCategory.RATE_LIMIT:
            return await self._handle_rate_limit(conversation_id, error)
        elif category == ErrorCategory.BUDGET_EXCEEDED:
            return await self._handle_budget_exceeded(conversation_id, error)
        else:  # PERMANENT
            return await self._handle_permanent_failure(conversation_id, error, context)
    
    async def _handle_transient_failure(
        self,
        conversation_id: str,
        error: Exception,
        context: dict
    ) -> dict:
        """Transient error: escala para humano temporariamente"""
        
        # Escala conversa
        await self.db.execute("""
            UPDATE conversations
            SET is_hil = true
            WHERE id = $1
        """, conversation_id)
        
        # Notifica humano com contexto
        await self.notifications.notify_agent(
            conversation_id=conversation_id,
            priority="high",
            message=f"Bot temporariamente indisponível (erro transitório). "
                    f"Por favor, assuma a conversa.",
            error_details={"type": type(error).__name__, "message": str(error)}
        )
        
        # Mensagem amigável ao customer
        await send_to_customer(
            self._get_customer_id(conversation_id),
            "Estamos com uma dificuldade técnica momentânea. "
            "Um atendente humano irá ajudá-lo em breve."
        )
        
        return {
            "escalated": True,
            "reason": "transient_error",
            "user_message": "Technical issue - human taking over"
        }
    
    async def _handle_rate_limit(
        self,
        conversation_id: str,
        error: RateLimitError
    ) -> dict:
        """Rate limit: escala e agenda retry"""
        
        await self.db.execute("""
            UPDATE conversations
            SET is_hil = true
            WHERE id = $1
        """, conversation_id)
        
        await self.notifications.notify_agent(
            conversation_id=conversation_id,
            priority="medium",
            message=f"Rate limit atingido. Retry em {error.retry_after}s."
        )
        
        await send_to_customer(
            self._get_customer_id(conversation_id),
            "Nosso sistema está temporariamente ocupado. "
            "Por favor, aguarde um momento ou um atendente irá ajudá-lo."
        )
        
        return {
            "escalated": True,
            "reason": "rate_limit",
            "retry_after": error.retry_after
        }
    
    async def _handle_budget_exceeded(
        self,
        conversation_id: str,
        error: BudgetExceededError
    ) -> dict:
        """Budget excedido: notifica admin e escala"""
        
        customer_id = await self._get_customer_id(conversation_id)
        
        # Notifica admin (não o customer diretamente)
        await self.notifications.notify_admin(
            subject=f"Customer {customer_id} budget exceeded",
            message=f"Customer {customer_id} has exceeded their monthly budget. "
                    f"Conversation {conversation_id} escalated to human."
        )
        
        # Escala silenciosamente
        await self.db.execute("""
            UPDATE conversations
            SET is_hil = true
            WHERE id = $1
        """, conversation_id)
        
        await send_to_customer(
            customer_id,
            "Um atendente humano irá ajudá-lo agora."
        )
        
        return {
            "escalated": True,
            "reason": "budget_exceeded"
        }
    
    async def _handle_permanent_failure(
        self,
        conversation_id: str,
        error: Exception,
        context: dict
    ) -> dict:
        """Permanent failure: escala com ticket"""
        
        # Cria ticket de suporte
        ticket = await self._create_support_ticket(
            conversation_id=conversation_id,
            error_type=type(error).__name__,
            error