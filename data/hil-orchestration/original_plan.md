```bash
# .env.example

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/conversations

# Redis (para timeout monitoring e cache)
REDIS_URL=redis://localhost:6379

# OpenAI (para PydanticAI e LangGraph)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Composio (para integrações)
COMPOSIO_API_KEY=...
COMPOSIO_BASE_URL=https://api.composio.dev

# Timeouts
HUMAN_TIMEOUT_MINUTES=30
BOT_RESPONSE_TIMEOUT_SECONDS=30

# Features
ENABLE_AUTO_SUMMARY=true
SUMMARY_THRESHOLD_MESSAGES=10
MAX_CONTEXT_MESSAGES=50

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Observability
ENABLE_METRICS=true
ENABLE_TRACING=true
JAEGER_HOST=localhost
JAEGER_PORT=6# Gerenciamento de Conversação - Versão Simplificada

## 1. Dois Estados, Uma Flag

### Estados
- **BOT_ATIVO**: `is_hil = false` → Bot responde
- **HUMANO_ATIVO**: `is_hil = true` → Bot ignora tudo

### Transições
```
BOT → HUMANO: Quando humano envia mensagem OU bot escala
HUMANO → BOT: Quando humano digita @bot OU timeout
```

## 2. Dados Mínimos

### Tabela: conversations
```sql
CREATE TABLE conversations (
  id UUID PRIMARY KEY,
  customer_id TEXT NOT NULL,
  is_hil BOOLEAN DEFAULT false,
  last_activity_at TIMESTAMP NOT NULL,
  last_bot_message_id UUID,
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Tabela: messages
```sql
CREATE TABLE messages (
  id UUID PRIMARY KEY,
  conversation_id UUID NOT NULL,
  sender_type TEXT NOT NULL, -- 'customer' | 'ai_agent' | 'human_agent'
  content TEXT NOT NULL,
  timestamp TIMESTAMP DEFAULT NOW()
);
```

## 3. Sistema de Comandos com PydanticAI

### 3.1 Agentes Especializados

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import Literal

# === Modelos de Output ===

class ConversationSummary(BaseModel):
    """Output estruturado do agente de sumarização"""
    main_topic: str = Field(description="Tópico principal da conversa")
    customer_issue: str = Field(description="Problema ou pedido do cliente")
    current_status: str = Field(description="Status atual da conversa")
    next_steps: list[str] = Field(description="Próximos passos recomendados")
    sentiment_score: float = Field(ge=-1, le=1, description="Sentimento geral (-1 a 1)")
    urgency: Literal["low", "medium", "high"] = Field(description="Nível de urgência")

class ConversationAnalysis(BaseModel):
    """Output do agente de análise"""
    sentiment: Literal["satisfied", "neutral", "dissatisfied"]
    urgency: Literal["low", "medium", "high"]
    complexity: Literal["simple", "moderate", "complex"]
    recommended_actions: list[str]
    bot_can_handle: bool = Field(description="Se o bot pode resolver sozinho")
    escalation_reason: str | None = Field(default=None, description="Razão para escalar se necessário")

class ResponseSuggestion(BaseModel):
    """Output do agente de sugestão de resposta"""
    suggested_response: str = Field(description="Resposta sugerida para o cliente")
    tone: Literal["professional", "empathetic", "apologetic", "casual"]
    confidence: float = Field(ge=0, le=1, description="Confiança na sugestão")
    alternative_approaches: list[str] = Field(description="Abordagens alternativas")

class SentimentAnalysis(BaseModel):
    """Output do agente de análise de sentimento"""
    overall_sentiment: float = Field(ge=-1, le=1, description="Sentimento geral")
    sentiment_trend: Literal["improving", "worsening", "stable"]
    emotional_keywords: list[str] = Field(description="Palavras-chave emocionais detectadas")
    customer_satisfaction: Literal["very_satisfied", "satisfied", "neutral", "dissatisfied", "very_dissatisfied"]
    approach_recommendations: list[str]

class DraftResponse(BaseModel):
    """Output do agente de rascunho"""
    draft: str
    template_used: str
    personalization_notes: list[str] = Field(description="O que foi personalizado")
    suggested_edits: list[str] = Field(description="Sugestões de edição")

# === Agentes PydanticAI ===

class SummaryAgent:
    """Agente especializado em sumarizar conversas"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.agent = Agent(
            model=model,
            result_type=ConversationSummary,
            system_prompt="""
            Você é um assistente especializado em sumarizar conversas de atendimento ao cliente.
            Analise a conversa e extraia as informações mais importantes de forma estruturada.
            Seja conciso mas completo.
            """
        )
    
    async def summarize(self, messages: list[dict]) -> ConversationSummary:
        """Sumariza lista de mensagens"""
        formatted = "\n".join([
            f"[{m['sender_type']}] {m['content']}" 
            for m in messages
        ])
        
        result = await self.agent.run(
            f"Analise e resuma esta conversa:\n\n{formatted}"
        )
        
        return result.data

class AnalysisAgent:
    """Agente especializado em análise de conversas"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.agent = Agent(
            model=model,
            result_type=ConversationAnalysis,
            system_prompt="""
            Você é um supervisor de atendimento ao cliente experiente.
            Analise conversas para determinar sentimento, urgência, complexidade
            e se o bot pode continuar ou precisa escalar para humano.
            """
        )
    
    async def analyze(self, messages: list[dict]) -> ConversationAnalysis:
        """Analisa conversa completa"""
        formatted = "\n".join([
            f"[{m['sender_type']}] {m['content']}" 
            for m in messages
        ])
        
        result = await self.agent.run(
            f"Analise esta conversa de atendimento:\n\n{formatted}"
        )
        
        return result.data

class SuggestionAgent:
    """Agente que sugere respostas"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.agent = Agent(
            model=model,
            result_type=ResponseSuggestion,
            system_prompt="""
            Você é um atendente experiente que sugere respostas excelentes.
            Considere o contexto completo, seja empático, profissional e resolva o problema.
            Ofereça alternativas quando apropriado.
            """
        )
    
    async def suggest(
        self, 
        conversation_context: str,
        customer_last_message: str
    ) -> ResponseSuggestion:
        """Sugere resposta baseada no contexto"""
        result = await self.agent.run(
            f"""
            Contexto da conversa:
            {conversation_context}
            
            Última mensagem do cliente:
            {customer_last_message}
            
            Sugira uma resposta apropriada.
            """
        )
        
        return result.data

class SentimentAgent:
    """Agente de análise de sentimento"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.agent = Agent(
            model=model,
            result_type=SentimentAnalysis,
            system_prompt="""
            Você é um psicólogo especializado em análise de sentimento em atendimento.
            Avalie o sentimento emocional do cliente ao longo da conversa.
            Identifique frustrações, satisfação, e recomende a melhor abordagem.
            """
        )
    
    async def analyze_sentiment(
        self, 
        customer_messages: list[dict]
    ) -> SentimentAnalysis:
        """Analisa sentimento das mensagens do cliente"""
        formatted = "\n".join([
            f"[{m['timestamp']}] {m['content']}" 
            for m in customer_messages
        ])
        
        result = await self.agent.run(
            f"Analise o sentimento nestas mensagens do cliente:\n\n{formatted}"
        )
        
        return result.data

class DraftAgent:
    """Agente para criar rascunhos"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        self.agent = Agent(
            model=model,
            result_type=DraftResponse,
            system_prompt="""
            Você cria rascunhos de respostas para atendimento ao cliente.
            Personalize templates mantendo tom profissional e empático.
            Seja específico e relevante ao contexto.
            """
        )
    
    async def create_draft(
        self,
        template_type: str,
        conversation_context: str
    ) -> DraftResponse:
        """Cria rascunho baseado em template"""
        
        templates = {
            "apology": "Pedimos desculpas pelo inconveniente causado...",
            "followup": "Retornando sobre seu caso...",
            "resolved": "Ficamos felizes em informar que seu problema foi resolvido...",
            "default": "Obrigado por entrar em contato conosco..."
        }
        
        base_template = templates.get(template_type, templates["default"])
        
        result = await self.agent.run(
            f"""
            Template base: {base_template}
            
            Contexto da conversa:
            {conversation_context}
            
            Crie um rascunho personalizado baseado neste template e contexto.
            """
        )
        
        return result.data

# === Command Handler Refatorado ===

class CommandHandler:
    """Executa comandos usando agentes especializados"""
    
    def __init__(self, db, langgraph_agent, composio_toolset):
        self.db = db
        self.agent = langgraph_agent
        self.composio = composio_toolset
        
        # Inicializa agentes especializados
        self.summary_agent = SummaryAgent()
        self.analysis_agent = AnalysisAgent()
        self.suggestion_agent = SuggestionAgent()
        self.sentiment_agent = SentimentAgent()
        self.draft_agent = DraftAgent()
    
    async def handle(self, conversation_id: str, command: str, args: dict) -> dict:
        """Executa comando e retorna resultado"""
        
        handlers = {
            BotCommand.BOT_TAKEOVER: self._handle_bot_takeover,
            BotCommand.BOT_PAUSE: self._handle_bot_pause,
            BotCommand.BOT_RESUME: self._handle_bot_resume,
            BotCommand.BOT_SUMMARY: self._handle_summary,
            BotCommand.BOT_CONTEXT: self._handle_context,
            BotCommand.BOT_STATUS: self._handle_status,
            BotCommand.BOT_ANALYZE: self._handle_analyze,
            BotCommand.BOT_SUGGEST: self._handle_suggest,
            BotCommand.BOT_DRAFT: self._handle_draft,
            BotCommand.BOT_ESCALATE: self._handle_escalate,
            BotCommand.BOT_TRANSFER: self._handle_transfer,
            BotCommand.BOT_HELP: self._handle_help,
            BotCommand.BOT_HISTORY: self._handle_history,
            BotCommand.BOT_SENTIMENT: self._handle_sentiment,
        }
        
        handler = handlers.get(command)
        if not handler:
            return {"error": f"Comando desconhecido: {command}"}
        
        return await handler(conversation_id, args)
    
    # === Handlers usando PydanticAI ===
    
    async def _handle_summary(self, conv_id: str, args: dict) -> dict:
        """@summary - Resume conversa usando SummaryAgent"""
        last_n = int(args.get("last", 0)) or None
        
        messages = self.db.get_messages(conv_id, limit=last_n)
        message_dicts = [
            {
                "sender_type": m.sender_type,
                "content": m.content,
                "timestamp": m.timestamp
            }
            for m in messages
        ]
        
        # Usa agente especializado
        summary = await self.summary_agent.summarize(message_dicts)
        
        return {
            "success": True,
            "summary": summary.model_dump(),
            "messages_analyzed": len(messages),
            "formatted": f"""
📝 **Resumo da Conversa**

**Tópico:** {summary.main_topic}

**Problema do Cliente:** {summary.customer_issue}

**Status Atual:** {summary.current_status}

**Próximos Passos:**
{chr(10).join(f"  • {step}" for step in summary.next_steps)}

**Sentimento:** {summary.sentiment_score:.2f} | **Urgência:** {summary.urgency}
            """
        }
    
    async def _handle_analyze(self, conv_id: str, args: dict) -> dict:
        """@analyze - Analisa conversa usando AnalysisAgent"""
        messages = self.db.get_messages(conv_id)
        message_dicts = [
            {
                "sender_type": m.sender_type,
                "content": m.content,
                "timestamp": m.timestamp
            }
            for m in messages
        ]
        
        # Usa agente especializado
        analysis = await self.analysis_agent.analyze(message_dicts)
        
        return {
            "success": True,
            "analysis": analysis.model_dump(),
            "formatted": f"""
📊 **Análise da Conversa**

**Sentimento:** {analysis.sentiment}
**Urgência:** {analysis.urgency}
**Complexidade:** {analysis.complexity}

**Bot pode resolver?** {"✅ Sim" if analysis.bot_can_handle else "❌ Não"}
{f"**Razão:** {analysis.escalation_reason}" if analysis.escalation_reason else ""}

**Ações Recomendadas:**
{chr(10).join(f"  • {action}" for action in analysis.recommended_actions)}
            """
        }
    
    async def _handle_suggest(self, conv_id: str, args: dict) -> dict:
        """@suggest - Sugere resposta usando SuggestionAgent"""
        messages = self.db.get_messages(conv_id)
        
        # Separa contexto e última mensagem do cliente
        customer_messages = [m for m in messages if m.sender_type == "customer"]
        last_customer_msg = customer_messages[-1].content if customer_messages else ""
        
        context = "\n".join([
            f"[{m.sender_type}] {m.content}"
            for m in messages[:-1]  # Tudo exceto última
        ])
        
        # Usa agente especializado
        suggestion = await self.suggestion_agent.suggest(context, last_customer_msg)
        
        return {
            "success": True,
            "suggestion": suggestion.model_dump(),
            "formatted": f"""
💡 **Sugestão de Resposta** (Confiança: {suggestion.confidence:.0%})

**Tom:** {suggestion.tone}

**Resposta Sugerida:**
{suggestion.suggested_response}

**Abordagens Alternativas:**
{chr(10).join(f"  • {alt}" for alt in suggestion.alternative_approaches)}

⚠️ _Esta é apenas uma sugestão. Revise antes de enviar._
            """
        }
    
    async def _handle_sentiment(self, conv_id: str, args: dict) -> dict:
        """@sentiment - Analisa sentimento usando SentimentAgent"""
        messages = self.db.get_messages(conv_id, sender_type="customer")
        message_dicts = [
            {
                "content": m.content,
                "timestamp": m.timestamp
            }
            for m in messages
        ]
        
        # Usa agente especializado
        sentiment = await self.sentiment_agent.analyze_sentiment(message_dicts)
        
        return {
            "success": True,
            "sentiment": sentiment.model_dump(),
            "formatted": f"""
🎭 **Análise de Sentimento**

**Sentimento Geral:** {sentiment.overall_sentiment:.2f} ({sentiment.customer_satisfaction})
**Tendência:** {sentiment.sentiment_trend}

**Palavras-chave Emocionais:**
{', '.join(sentiment.emotional_keywords)}

**Recomendações de Abordagem:**
{chr(10).join(f"  • {rec}" for rec in sentiment.approach_recommendations)}
            """,
            "messages_analyzed": len(messages)
        }
    
    async def _handle_draft(self, conv_id: str, args: dict) -> dict:
        """@draft - Cria rascunho usando DraftAgent"""
        template_type = args.get("type", "default")
        
        messages = self.db.get_messages(conv_id)
        context = "\n".join([
            f"[{m.sender_type}] {m.content}"
            for m in messages
        ])
        
        # Usa agente especializado
        draft = await self.draft_agent.create_draft(template_type, context)
        
        return {
            "success": True,
            "draft": draft.model_dump(),
            "formatted": f"""
✏️ **Rascunho de Resposta**

**Template:** {draft.template_used}

**Rascunho:**
{draft.draft}

**Personalizações Aplicadas:**
{chr(10).join(f"  • {note}" for note in draft.personalization_notes)}

**Sugestões de Edição:**
{chr(10).join(f"  • {edit}" for edit in draft.suggested_edits)}
            """
        }
    
    # === Handlers simples (não precisam de agentes) ===
    
    async def _handle_bot_takeover(self, conv_id: str, args: dict) -> dict:
        """@bot - Bot assume conversa"""
        conv = self.db.get_conversation(conv_id)
        
        # Usa summary agent para contexto
        messages = self.db.get_messages_since(conv.id, conv.last_bot_message_id)
        message_dicts = [
            {"sender_type": m.sender_type, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ]
        
        summary = await self.summary_agent.summarize(message_dicts)
        instruction = args.get("instruction", "Continue a conversa")
        
        # Reativa bot
        conv.is_hil = False
        
        result = self.agent.invoke({
            "messages": [
                {"role": "system", "content": f"Resumo: {summary.model_dump_json()}"},
                {"role": "user", "content": instruction}
            ],
            "composio_entity_id": self._get_composio_entity(conv.customer_id)
        })
        
        self.db.save(conv)
        
        return {
            "success": True,
            "bot_response": result.get("final_response"),
            "status": "bot_active"
        }
    
    async def _handle_bot_pause(self, conv_id: str, args: dict) -> dict:
        """@pause - Pausa bot"""
        conv = self.db.get_conversation(conv_id)
        conv.is_hil = True
        self.db.save(conv)
        
        return {
            "success": True,
            "message": "Bot pausado. Use @resume para reativar.",
            "status": "human_active"
        }
    
    async def _handle_bot_resume(self, conv_id: str, args: dict) -> dict:
        """@resume - Resume bot"""
        return await self._handle_bot_takeover(conv_id, {"instruction": "Continue"})
    
    async def _handle_context(self, conv_id: str, args: dict) -> dict:
        """@context - Mostra contexto completo"""
        conv = self.db.get_conversation(conv_id)
        messages = self.db.get_messages(conv_id)
        
        return {
            "success": True,
            "conversation_id": conv_id,
            "customer_id": conv.customer_id,
            "is_hil": conv.is_hil,
            "started_at": conv.created_at,
            "last_activity": conv.last_activity_at,
            "message_count": len(messages),
            "messages": [
                {
                    "sender": m.sender_type,
                    "content": m.content,
                    "timestamp": m.timestamp
                }
                for m in messages[-10:]
            ]
        }
    
    async def _handle_status(self, conv_id: str, args: dict) -> dict:
        """@status - Status da conversa"""
        conv = self.db.get_conversation(conv_id)
        
        status = "🤖 Bot Ativo" if not conv.is_hil else "👤 Humano Ativo"
        
        return {
            "success": True,
            "status": status,
            "is_hil": conv.is_hil,
            "last_activity": conv.last_activity_at,
            "inactive_minutes": (now() - conv.last_activity_at).total_seconds() / 60
        }
    
    async def _handle_escalate(self, conv_id: str, args: dict) -> dict:
        """@escalate - Escala para supervisor"""
        conv = self.db.get_conversation(conv_id)
        reason = args.get("reason", "Requisitado pelo atendente")
        
        # Usa summary agent
        messages = self.db.get_messages(conv_id)
        message_dicts = [
            {"sender_type": m.sender_type, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ]
        summary = await self.summary_agent.summarize(message_dicts)
        
        # Notifica supervisor
        self._notify_supervisor(conv_id, reason, summary)
        
        conv.is_hil = True
        conv.escalated = True
        self.db.save(conv)
        
        return {
            "success": True,
            "message": "Escalado para supervisor",
            "reason": reason,
            "summary": summary.model_dump()
        }
    
    async def _handle_transfer(self, conv_id: str, args: dict) -> dict:
        """@transfer - Transfere departamento"""
        department = args.get("department", "geral")
        reason = args.get("reason", "Transferência solicitada")
        
        conv = self.db.get_conversation(conv_id)
        
        # Usa summary agent
        messages = self.db.get_messages(conv_id)
        message_dicts = [
            {"sender_type": m.sender_type, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ]
        summary = await self.summary_agent.summarize(message_dicts)
        
        self._notify_department(department, conv_id, reason, summary)
        
        conv.department = department
        self.db.save(conv)
        
        return {
            "success": True,
            "message": f"Transferido para {department}",
            "reason": reason,
            "summary": summary.model_dump()
        }
    
    async def _handle_help(self, conv_id: str, args: dict) -> dict:
        """@help - Lista comandos"""
        return {
            "success": True,
            "commands": {
                "Controle": {
                    "@bot [instrução]": "Bot assume conversa",
                    "@pause": "Pausa bot",
                    "@resume": "Resume bot",
                },
                "Informações": {
                    "@summary [last:N]": "Resume conversa (PydanticAI)",
                    "@context": "Contexto completo",
                    "@status": "Status atual",
                },
                "Ações": {
                    "@analyze": "Analisa conversa (PydanticAI)",
                    "@suggest": "Sugere resposta (PydanticAI)",
                    "@draft [type:X]": "Cria rascunho (PydanticAI)",
                },
                "Escalação": {
                    "@escalate [reason:X]": "Escala para supervisor",
                    "@transfer department:X": "Transfere departamento",
                },
                "Utilidades": {
                    "@help": "Esta mensagem",
                    "@history [last:N]": "Histórico",
                    "@sentiment": "Análise de sentimento (PydanticAI)",
                }
            }
        }
    
    async def _handle_history(self, conv_id: str, args: dict) -> dict:
        """@history - Histórico"""
        last_n = int(args.get("last", 20))
        messages = self.db.get_messages(conv_id, limit=last_n)
        
        return {
            "success": True,
            "history": [
                {
                    "time": m.timestamp,
                    "sender": m.sender_type,
                    "message": m.content[:100] + "..." if len(m.content) > 100 else m.content
                }
                for m in messages
            ]
        }
    
    def _get_composio_entity(self, customer_id):
        """Obtém entity ID do Composio"""
        pass
    
    def _notify_supervisor(self, conv_id, reason, summary):
        """Notifica supervisor"""
        pass
    
    def _notify_department(self, department, conv_id, reason, summary):
        """Notifica departamento"""
        pass
```

### 3.2 Integração Assíncrona com Webhook

```python
import asyncio

async def on_message_received_async(conversation_id, message):
    """Versão assíncrona do webhook handler"""
    conv = db.get_conversation(conversation_id)
    db.save_message(message)
    
    if message.sender_type == 'human_agent':
        command, args = CommandParser.parse(message.content)
        
        if command:
            # Executa comando (agora async)
            command_handler = CommandHandler(db, langgraph_agent, composio_toolset)
            result = await command_handler.handle(conversation_id, command, args)
            
            # Envia resultado formatado
            send_to_agent(conversation_id, result.get("formatted", result))
        else:
            conv.is_hil = True
            iniciar_timeout(conversation_id)
    
    elif message.sender_type == 'customer':
        if not conv.is_hil:
            # Bot processa normalmente
            pass
    
    db.save(conv)

# Wrapper síncrono se necessário
def on_message_received(conversation_id, message):
    """Wrapper síncrono para compatibilidade"""
    asyncio.run(on_message_received_async(conversation_id, message))
```

### 3.3 Vantagens do PydanticAI

✅ **Type Safety**: Outputs validados e estruturados  
✅ **Agentes Especializados**: Cada tarefa tem seu próprio agente otimizado  
✅ **Testabilidade**: Fácil mockar agentes individuais  
✅ **Reutilizável**: Agentes podem ser usados em outros contextos  
✅ **Observabilidade**: PydanticAI tem tracing built-in  
✅ **Validação**: Garante que LLM retorne estrutura esperada  

### 3.4 Exemplo de Uso

```python
# Teste unitário de um agente
async def test_summary_agent():
    agent = SummaryAgent(model="openai:gpt-4")
    
    messages = [
        {"sender_type": "customer", "content": "Onde está meu pedido #123?"},
        {"sender_type": "ai_agent", "content": "Seu pedido está em trânsito."},
        {"sender_type": "customer", "content": "Quando chega?"},
    ]
    
    summary = await agent.summarize(messages)
    
    assert isinstance(summary, ConversationSummary)
    assert summary.urgency in ["low", "medium", "high"]
    assert -1 <= summary.sentiment_score <= 1
    assert len(summary.next_steps) > 0
```

## 4. Arquitetura em Camadas com PydanticAIotCommand.BOT_RESUME: self._handle_bot_resume,
            BotCommand.BOT_SUMMARY: self._handle_summary,
            BotCommand.BOT_CONTEXT: self._handle_context,
            BotCommand.BOT_STATUS: self._handle_status,
            BotCommand.BOT_ANALYZE: self._handle_analyze,
            BotCommand.BOT_SUGGEST: self._handle_suggest,
            BotCommand.BOT_DRAFT: self._handle_draft,
            BotCommand.BOT_ESCALATE: self._handle_escalate,
            BotCommand.BOT_TRANSFER: self._handle_transfer,
            BotCommand.BOT_HELP: self._handle_help,
            BotCommand.BOT_HISTORY: self._handle_history,
            BotCommand.BOT_SENTIMENT: self._handle_sentiment,
        }
        
        handler = handlers.get(command)
        if not handler:
            return {"error": f"Comando desconhecido: {command}"}
        
        return handler(conversation_id, args)
    
    # === Controle do Bot ===
    
    def _handle_bot_takeover(self, conv_id: str, args: dict) -> dict:
        """@bot - Bot assume conversa"""
        conv = self.db.get_conversation(conv_id)
        
        # Sumariza desde último bot
        context = self._summarize_context(conv)
        instruction = args.get("instruction", "Continue a conversa")
        
        # Reativa bot
        conv.is_hil = False
        
        result = self.agent.invoke({
            "messages": context + [{"role": "system", "content": instruction}],
            "composio_entity_id": self._get_composio_entity(conv.customer_id)
        })
        
        self.db.save(conv)
        
        return {
            "success": True,
            "bot_response": result.get("final_response"),
            "status": "bot_active"
        }
    
    def _handle_bot_pause(self, conv_id: str, args: dict) -> dict:
        """@pause - Pausa bot"""
        conv = self.db.get_conversation(conv_id)
        conv.is_hil = True
        self.db.save(conv)
        
        return {
            "success": True,
            "message": "Bot pausado. Use @resume para reativar.",
            "status": "human_active"
        }
    
    def _handle_bot_resume(self, conv_id: str, args: dict) -> dict:
        """@resume - Resume bot sem instrução específica"""
        return self._handle_bot_takeover(conv_id, {"instruction": "Continue"})
    
    # === Informações ===
    
    def _handle_summary(self, conv_id: str, args: dict) -> dict:
        """@summary - Resume conversa"""
        last_n = int(args.get("last", 0)) or None
        
        messages = self.db.get_messages(conv_id, limit=last_n)
        
        summary = self.llm.invoke(f"""
        Resuma a seguinte conversa de atendimento ao cliente:
        
        {self._format_messages(messages)}
        
        Forneça:
        1. Tópico principal
        2. Problema/pedido do cliente
        3. Status atual
        4. Próximos passos recomendados
        """)
        
        return {
            "success": True,
            "summary": summary,
            "messages_analyzed": len(messages)
        }
    
    def _handle_context(self, conv_id: str, args: dict) -> dict:
        """@context - Mostra contexto completo"""
        conv = self.db.get_conversation(conv_id)
        messages = self.db.get_messages(conv_id)
        
        return {
            "success": True,
            "conversation_id": conv_id,
            "customer_id": conv.customer_id,
            "is_hil": conv.is_hil,
            "started_at": conv.created_at,
            "last_activity": conv.last_activity_at,
            "message_count": len(messages),
            "messages": [
                {
                    "sender": m.sender_type,
                    "content": m.content,
                    "timestamp": m.timestamp
                }
                for m in messages[-10:]  # Últimas 10
            ]
        }
    
    def _handle_status(self, conv_id: str, args: dict) -> dict:
        """@status - Status da conversa"""
        conv = self.db.get_conversation(conv_id)
        
        status = "🤖 Bot Ativo" if not conv.is_hil else "👤 Humano Ativo"
        
        return {
            "success": True,
            "status": status,
            "is_hil": conv.is_hil,
            "last_activity": conv.last_activity_at,
            "inactive_minutes": (now() - conv.last_activity_at).total_seconds() / 60
        }
    
    # === Ações do Bot ===
    
    def _handle_analyze(self, conv_id: str, args: dict) -> dict:
        """@analyze - Bot analisa e sugere ações"""
        messages = self.db.get_messages(conv_id)
        
        analysis = self.llm.invoke(f"""
        Analise esta conversa de atendimento:
        
        {self._format_messages(messages)}
        
        Forneça:
        1. Sentimento do cliente (satisfeito/neutro/insatisfeito)
        2. Urgência (baixa/média/alta)
        3. Complexidade do problema
        4. Ações recomendadas
        5. Se o bot pode resolver ou precisa escalar
        """)
        
        return {
            "success": True,
            "analysis": analysis
        }
    
    def _handle_suggest(self, conv_id: str, args: dict) -> dict:
        """@suggest - Bot sugere resposta (não envia)"""
        conv = self.db.get_conversation(conv_id)
        context = self._summarize_context(conv)
        
        suggestion = self.llm.invoke(f"""
        Com base neste contexto:
        {context}
        
        Sugira uma resposta apropriada para o cliente.
        A resposta deve ser profissional, empática e resolver o problema.
        """)
        
        return {
            "success": True,
            "suggestion": suggestion,
            "note": "Esta é apenas uma sugestão. Revise antes de enviar."
        }
    
    def _handle_draft(self, conv_id: str, args: dict) -> dict:
        """@draft - Cria rascunho de resposta"""
        template_type = args.get("type", "default")
        
        templates = {
            "apology": "Pedimos desculpas pelo inconveniente...",
            "followup": "Retornando sobre seu caso...",
            "resolved": "Ficamos felizes em resolver...",
            "default": "Obrigado por entrar em contato..."
        }
        
        base = templates.get(template_type, templates["default"])
        
        # LLM personaliza template
        draft = self.llm.invoke(f"""
        Crie uma resposta baseada neste template:
        {base}
        
        Contexto da conversa:
        {self._summarize_context(self.db.get_conversation(conv_id))}
        
        Personalize mantendo tom profissional e empático.
        """)
        
        return {
            "success": True,
            "draft": draft,
            "template_used": template_type
        }
    
    # === Escalação ===
    
    def _handle_escalate(self, conv_id: str, args: dict) -> dict:
        """@escalate - Escala para supervisor"""
        conv = self.db.get_conversation(conv_id)
        reason = args.get("reason", "Requisitado pelo atendente")
        
        # Cria ticket de escalação
        summary = self._handle_summary(conv_id, {})["summary"]
        
        # Notifica supervisor
        self._notify_supervisor(conv_id, reason, summary)
        
        conv.is_hil = True
        conv.escalated = True
        self.db.save(conv)
        
        return {
            "success": True,
            "message": "Escalado para supervisor",
            "reason": reason
        }
    
    def _handle_transfer(self, conv_id: str, args: dict) -> dict:
        """@transfer - Transfere para outro departamento"""
        department = args.get("department", "geral")
        reason = args.get("reason", "Transferência solicitada")
        
        conv = self.db.get_conversation(conv_id)
        summary = self._handle_summary(conv_id, {})["summary"]
        
        # Notifica novo departamento
        self._notify_department(department, conv_id, reason, summary)
        
        conv.department = department
        self.db.save(conv)
        
        return {
            "success": True,
            "message": f"Transferido para {department}",
            "reason": reason
        }
    
    # === Utilidades ===
    
    def _handle_help(self, conv_id: str, args: dict) -> dict:
        """@help - Lista comandos"""
        return {
            "success": True,
            "commands": {
                "Controle": {
                    "@bot [instrução]": "Bot assume conversa",
                    "@pause": "Pausa bot",
                    "@resume": "Resume bot",
                },
                "Informações": {
                    "@summary [last:N]": "Resume conversa",
                    "@context": "Contexto completo",
                    "@status": "Status atual",
                },
                "Ações": {
                    "@analyze": "Analisa conversa",
                    "@suggest": "Sugere resposta",
                    "@draft [type:X]": "Cria rascunho",
                },
                "Escalação": {
                    "@escalate [reason:X]": "Escala para supervisor",
                    "@transfer department:X": "Transfere departamento",
                },
                "Utilidades": {
                    "@help": "Esta mensagem",
                    "@history [last:N]": "Histórico",
                    "@sentiment": "Análise de sentimento",
                }
            }
        }
    
    def _handle_history(self, conv_id: str, args: dict) -> dict:
        """@history - Histórico de interações"""
        last_n = int(args.get("last", 20))
        messages = self.db.get_messages(conv_id, limit=last_n)
        
        return {
            "success": True,
            "history": [
                {
                    "time": m.timestamp,
                    "sender": m.sender_type,
                    "message": m.content[:100] + "..." if len(m.content) > 100 else m.content
                }
                for m in messages
            ]
        }
    
    def _handle_sentiment(self, conv_id: str, args: dict) -> dict:
        """@sentiment - Análise de sentimento"""
        messages = self.db.get_messages(conv_id, sender_type="customer")
        
        sentiment_analysis = self.llm.invoke(f"""
        Analise o sentimento destas mensagens do cliente:
        
        {self._format_messages(messages)}
        
        Retorne:
        1. Sentimento geral (score -1 a 1)
        2. Evolução do sentimento (melhorando/piorando/estável)
        3. Palavras-chave emocionais
        4. Recomendações de abordagem
        """)
        
        return {
            "success": True,
            "sentiment": sentiment_analysis,
            "messages_analyzed": len(messages)
        }
    
    # === Helpers ===
    
    def _summarize_context(self, conv):
        """Helper para sumarizar contexto"""
        messages = self.db.get_messages_since(conv.id, conv.last_bot_message_id)
        
        if len(messages) <= 5:
            return [{"role": "user", "content": m.content} for m in messages]
        
        summary = self.llm.summarize(messages)
        return [{"role": "system", "content": f"Resumo: {summary}"}]
    
    def _format_messages(self, messages):
        """Formata mensagens para LLM"""
        return "\n".join([
            f"[{m.sender_type}] {m.content}" 
            for m in messages
        ])
    
    def _get_composio_entity(self, customer_id):
        """Obtém entity ID do Composio"""
        # Implementação anterior
        pass
    
    def _notify_supervisor(self, conv_id, reason, summary):
        """Notifica supervisor sobre escalação"""
        # Implementação específica (email, Slack, etc)
        pass
    
    def _notify_department(self, department, conv_id, reason, summary):
        """Notifica novo departamento"""
        # Implementação específica
        pass
```

### 3.2 Integração com Webhook

```python
def on_message_received(conversation_id, message):
    conv = db.get_conversation(conversation_id)
    db.save_message(message)
    
    # Se humano enviou
    if message.sender_type == 'human_agent':
        # Parse comando
        command, args = CommandParser.parse(message.content)
        
        if command:
            # Executa comando
            command_handler = CommandHandler(db, langgraph_agent, composio_toolset, llm)
            result = command_handler.handle(conversation_id, command, args)
            
            # Envia resultado para o atendente
            send_to_agent(conversation_id, result)
        else:
            # Mensagem normal, humano assume
            conv.is_hil = True
            iniciar_timeout(conversation_id)
    
    # Se cliente enviou
    elif message.sender_type == 'customer':
        if not conv.is_hil:
            # Bot processa normalmente
            # ... (código anterior)
            pass
    
    db.save(conv)
```

### 3.3 Exemplos de Uso

```
Atendente: @status
Sistema: 🤖 Bot Ativo | Última atividade: há 2 minutos

Atendente: @summary last:5
Sistema: 
📝 Resumo das últimas 5 mensagens:
- Cliente perguntou sobre pedido #12345
- Bot consultou sistema e informou status "em trânsito"
- Cliente quer saber prazo de entrega
- Bot estimou 2-3 dias úteis
- Cliente agradeceu

Atendente: @analyze
Sistema:
📊 Análise:
- Sentimento: Neutro/Positivo (0.6)
- Urgência: Baixa
- Complexidade: Simples
- Recomendação: Bot pode continuar, apenas acompanhamento de rotina

Atendente: @bot Por favor, ofereça um cupom de desconto de 10% para próxima compra
Sistema: ✅ Bot assumiu conversa

Bot → Cliente: Obrigado pela sua paciência! Como agradecimento, preparei um cupom...

Atendente: @pause
Sistema: ⏸️ Bot pausado

Atendente: [conversa com cliente...]

Atendente: @suggest
Sistema:
💡 Sugestão de resposta:
"Entendo sua preocupação com o atraso. Vou verificar com nossa equipe de logística 
e retorno em até 1 hora com uma atualização detalhada."

Atendente: @transfer department:logistica reason:precisa rastreamento detalhado
Sistema: ✅ Transferido para logistica

Atendente: @escalate reason:cliente solicita cancelamento de pedido já enviado
Sistema: ✅ Escalado para supervisor | Ticket #5678 criado
```

## 4. Arquitetura em Camadas com Comandos

### Arquitetura em Camadas

```
┌─────────────────────────────────────────────┐
│   Orquestrador (HIL Manager)               │  ← Gerencia is_hil
│   - Webhook Handler (async)                 │
│   - Command Parser                          │
│   - Command Router                          │
└──────────┬──────────────────────────────────┘
           │
           ├─→ is_hil = false
           │   ┌────────────────────────────────┐
           │   │   LangGraph Agent              │  ← Raciocínio complexo
           │   │   - Entender                   │
           │   │   - Planejar                   │
           │   │   - Executar (Composio)        │
           │   │   - Validar                    │
           │   │   - Responder/Escalar          │
           │   └────────────────────────────────┘
           │
           ├─→ Comandos (@summary, @analyze...)
           │   ┌────────────────────────────────┐
           │   │   PydanticAI Agents            │  ← Tarefas especializadas
           │   │   - SummaryAgent               │
           │   │   - AnalysisAgent              │
           │   │   - SuggestionAgent            │
           │   │   - SentimentAgent             │
           │   │   - DraftAgent                 │
           │   └────────────────────────────────┘
           │
           └─→ is_hil = true
               ┌────────────────────────────────┐
               │   Timeout Monitor              │
               │   - Aguarda humano             │
               │   - Detecta @bot               │
               │   - Verifica timeout           │
               └────────────────────────────────┘
```

### Stack Completo

```python
# main.py - Aplicação completa

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio

app = FastAPI()

# === Inicialização ===

# Database
db = Database()

# LangGraph Agent (bot principal)
langgraph_agent = build_bot_graph()

# Composio (integrações)
composio_toolset = ComposioToolSet()

# PydanticAI Agents (comandos)
summary_agent = SummaryAgent()
analysis_agent = AnalysisAgent()
suggestion_agent = SuggestionAgent()
sentiment_agent = SentimentAgent()
draft_agent = DraftAgent()

# Command Handler
command_handler = CommandHandler(
    db=db,
    langgraph_agent=langgraph_agent,
    composio_toolset=composio_toolset
)

# === Models ===

class IncomingMessage(BaseModel):
    conversation_id: str
    sender_type: str  # 'customer' | 'human_agent'
    sender_id: str
    content: str

# === Endpoints ===

@app.post("/webhook/message")
async def receive_message(message: IncomingMessage, background_tasks: BackgroundTasks):
    """
    Webhook que recebe mensagens de qualquer canal
    (WhatsApp, Telegram, Web Chat, etc)
    """
    
    # Processa de forma assíncrona
    background_tasks.add_task(process_message, message)
    
    return {"status": "received"}

async def process_message(message: IncomingMessage):
    """Processa mensagem de forma assíncrona"""
    
    conv = db.get_or_create_conversation(message.conversation_id)
    
    # Salva mensagem
    db.save_message({
        "conversation_id": message.conversation_id,
        "sender_type": message.sender_type,
        "sender_id": message.sender_id,
        "content": message.content,
        "timestamp": now()
    })
    
    # Roteamento baseado em sender
    if message.sender_type == "human_agent":
        await handle_human_agent_message(conv, message)
    elif message.sender_type == "customer":
        await handle_customer_message(conv, message)
    
    # Atualiza timestamp
    conv.last_activity_at = now()
    db.save(conv)

async def handle_human_agent_message(conv, message):
    """Processa mensagem do atendente humano"""
    
    # Tenta parsear comando
    command, args = CommandParser.parse(message.content)
    
    if command:
        # É um comando - executa via PydanticAI agents
        result = await command_handler.handle(
            conv.id, 
            command, 
            args
        )
        
        # Envia resultado formatado para o atendente
        await send_to_agent(conv.id, result.get("formatted", result))
        
    else:
        # Mensagem normal - humano assume controle
        conv.is_hil = True
        db.save(conv)
        
        # Inicia monitoramento de timeout
        await start_timeout_monitor(conv.id)
        
        # Encaminha mensagem para o cliente
        await send_to_customer(conv.customer_id, message.content)

async def handle_customer_message(conv, message):
    """Processa mensagem do cliente"""
    
    if conv.is_hil:
        # Humano no controle - apenas encaminha para atendente
        await send_to_agent(conv.id, {
            "type": "customer_message",
            "content": message.content,
            "timestamp": now()
        })
    else:
        # Bot no controle - processa via LangGraph
        try:
            history = db.get_conversation_history(conv.id)
            entity_id = get_or_create_composio_entity(conv.customer_id)
            
            result = langgraph_agent.invoke({
                "messages": history + [
                    {"role": "user", "content": message.content}
                ],
                "composio_entity_id": entity_id
            })
            
            # Verifica se bot escalou
            if result.get("should_escalate"):
                # Bot decidiu escalar
                conv.is_hil = True
                db.save(conv)
                
                # Notifica atendente
                await notify_human_agent(
                    conv.id,
                    escalation_reason=result.get("escalation_reason"),
                    summary=result.get("summary")
                )
            else:
                # Bot respondeu normalmente
                await send_to_customer(
                    conv.customer_id,
                    result["final_response"]
                )
                
                # Salva ID da última mensagem do bot
                conv.last_bot_message_id = result.get("message_id")
                db.save(conv)
                
        except Exception as e:
            # Erro no bot - escala para humano
            conv.is_hil = True
            db.save(conv)
            
            await notify_human_agent(
                conv.id,
                escalation_reason=f"Erro no bot: {str(e)}",
                error=True
            )

# === Timeout Monitor (Job Cron) ===

@app.on_event("startup")
async def start_background_jobs():
    """Inicia jobs em background"""
    asyncio.create_task(timeout_monitor_loop())

async def timeout_monitor_loop():
    """Loop que verifica timeouts a cada 5 minutos"""
    while True:
        await asyncio.sleep(300)  # 5 minutos
        await check_all_timeouts()

async def check_all_timeouts():
    """Verifica conversas com timeout"""
    
    stalled_convs = db.query("""
        SELECT * FROM conversations 
        WHERE is_hil = true 
        AND last_activity_at < NOW() - INTERVAL '30 minutes'
        AND NOT closed
    """)
    
    for conv in stalled_convs:
        # Sumariza conversa antes de encerrar
        messages = db.get_messages(conv.id)
        message_dicts = [
            {"sender_type": m.sender_type, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ]
        
        summary = await summary_agent.summarize(message_dicts)
        
        # Envia mensagem de encerramento
        await send_to_customer(
            conv.customer_id,
            "Detectamos inatividade e encerramos este atendimento. "
            "Se precisar de ajuda, inicie uma nova conversa."
        )
        
        # Arquiva conversa
        db.archive_conversation(
            conv.id,
            reason="timeout",
            summary=summary.model_dump()
        )
        
        # Notifica atendente
        await notify_human_agent(
            conv.id,
            notification_type="timeout_closed",
            summary=summary
        )

# === Helper Functions ===

async def send_to_customer(customer_id: str, content: str):
    """Envia mensagem para o cliente via canal apropriado"""
    # Implementação depende do canal (WhatsApp, Telegram, etc)
    pass

async def send_to_agent(conversation_id: str, content: dict | str):
    """Envia mensagem/resultado para o atendente"""
    # Implementação via dashboard, Slack, etc
    pass

async def notify_human_agent(conversation_id: str, **kwargs):
    """Notifica atendente sobre eventos importantes"""
    # Implementação via webhook, Slack, email, etc
    pass

async def start_timeout_monitor(conversation_id: str):
    """Inicia monitoramento de timeout para conversa"""
    # Pode usar Redis com TTL ou job scheduler
    pass

def get_or_create_composio_entity(customer_id: str) -> str:
    """Mapeia customer para Composio entity"""
    entity = db.get_composio_entity(customer_id)
    
    if not entity:
        entity_id = composio_toolset.create_entity(
            name=customer_id,
            metadata={"customer_id": customer_id}
        )
        db.save_composio_entity(customer_id, entity_id)
        return entity_id
    
    return entity.composio_entity_id

def now():
    """Helper para timestamp"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)

# === Observability ===

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agents": {
            "langgraph": "ok",
            "summary": "ok",
            "analysis": "ok",
            "suggestion": "ok",
            "sentiment": "ok",
            "draft": "ok"
        }
    }

@app.get("/stats")
async def get_stats():
    """Estatísticas do sistema"""
    return {
        "active_conversations": db.count_active_conversations(),
        "bot_active": db.count_conversations(is_hil=False),
        "human_active": db.count_conversations(is_hil=True),
        "avg_response_time": db.get_avg_response_time(),
        "escalation_rate": db.get_escalation_rate()
    }
```

### Testes Unitários (PydanticAI Agents)

```python
import pytest
from unittest.mock import AsyncMock, Mock

@pytest.mark.asyncio
async def test_summary_agent():
    """Testa SummaryAgent isoladamente"""
    agent = SummaryAgent(model="openai:gpt-4")
    
    messages = [
        {"sender_type": "customer", "content": "Meu pedido #123 não chegou"},
        {"sender_type": "ai_agent", "content": "Verificando seu pedido..."},
        {"sender_type": "ai_agent", "content": "Está em trânsito, chega amanhã"},
        {"sender_type": "customer", "content": "Ok, obrigado"}
    ]
    
    summary = await agent.summarize(messages)
    
    # Validações
    assert isinstance(summary, ConversationSummary)
    assert "pedido" in summary.main_topic.lower() or "entrega" in summary.main_topic.lower()
    assert summary.urgency in ["low", "medium", "high"]
    assert -1 <= summary.sentiment_score <= 1
    assert len(summary.next_steps) > 0

@pytest.mark.asyncio
async def test_analysis_agent_recommends_escalation():
    """Testa AnalysisAgent com caso que deve escalar"""
    agent = AnalysisAgent(model="openai:gpt-4")
    
    messages = [
        {"sender_type": "customer", "content": "QUERO MEU DINHEIRO DE VOLTA AGORA!!!"},
        {"sender_type": "ai_agent", "content": "Vou processar seu reembolso..."},
        {"sender_type": "customer", "content": "Já tentei 3 vezes e nada!"},
    ]
    
    analysis = await agent.analyze(messages)
    
    assert analysis.sentiment == "dissatisfied"
    assert analysis.urgency == "high"
    assert not analysis.bot_can_handle
    assert analysis.escalation_reason is not None

@pytest.mark.asyncio
async def test_suggestion_agent():
    """Testa SuggestionAgent"""
    agent = SuggestionAgent(model="openai:gpt-4")
    
    context = "[customer] Quanto custa o frete?\n[bot] Depende do CEP"
    last_message = "Meu CEP é 01310-100"
    
    suggestion = await agent.suggest(context, last_message)
    
    assert isinstance(suggestion, ResponseSuggestion)
    assert len(suggestion.suggested_response) > 0
    assert suggestion.tone in ["professional", "empathetic", "apologetic", "casual"]
    assert 0 <= suggestion.confidence <= 1
    assert len(suggestion.alternative_approaches) > 0

@pytest.mark.asyncio
async def test_sentiment_agent_detects_frustration():
    """Testa SentimentAgent com cliente frustrado"""
    agent = SentimentAgent(model="openai:gpt-4")
    
    messages = [
        {"content": "Olá, gostaria de saber sobre meu pedido", "timestamp": "2025-01-01T10:00:00"},
        {"content": "Ainda não recebi resposta...", "timestamp": "2025-01-01T10:30:00"},
        {"content": "Isso é um absurdo!", "timestamp": "2025-01-01T11:00:00"},
    ]
    
    sentiment = await agent.analyze_sentiment(messages)
    
    assert sentiment.overall_sentiment < 0  # Negativo
    assert sentiment.sentiment_trend == "worsening"
    assert sentiment.customer_satisfaction in ["dissatisfied", "very_dissatisfied"]
    assert len(sentiment.emotional_keywords) > 0

@pytest.mark.asyncio
async def test_draft_agent():
    """Testa DraftAgent"""
    agent = DraftAgent(model="openai:gpt-4")
    
    context = "[customer] Recebi produto errado\n[bot] Lamento pelo erro"
    
    draft = await agent.create_draft("apology", context)
    
    assert isinstance(draft, DraftResponse)
    assert len(draft.draft) > 0
    assert draft.template_used == "apology"
    assert len(draft.personalization_notes) > 0
```

### Testes de Integração (Comandos)

```python
@pytest.mark.asyncio
async def test_command_handler_summary():
    """Testa comando @summary end-to-end"""
    
    # Mock database
    mock_db = Mock()
    mock_db.get_messages.return_value = [
        Mock(sender_type="customer", content="Oi", timestamp="2025-01-01T10:00:00"),
        Mock(sender_type="ai_agent", content="Olá!", timestamp="2025-01-01T10:00:05"),
    ]
    
    handler = CommandHandler(mock_db, None, None)
    
    result = await handler.handle("conv_123", "@summary", {"last": 10})
    
    assert result["success"] == True
    assert "summary" in result
    assert "formatted" in result
    assert "messages_analyzed" in result

@pytest.mark.asyncio
async def test_command_handler_bot_takeover():
    """Testa comando @bot"""
    
    mock_db = Mock()
    mock_conv = Mock(id="conv_123", customer_id="cust_456", is_hil=True)
    mock_db.get_conversation.return_value = mock_conv
    mock_db.get_messages_since.return_value = []
    
    mock_agent = Mock()
    mock_agent.invoke.return_value = {"final_response": "Olá! Como posso ajudar?"}
    
    handler = CommandHandler(mock_db, mock_agent, None)
    
    result = await handler.handle("conv_123", "@bot", {"instruction": "Seja educado"})
    
    assert result["success"] == True
    assert result["status"] == "bot_active"
    assert mock_conv.is_hil == False  # Bot reativado

@pytest.mark.asyncio
async def test_command_handler_analyze():
    """Testa comando @analyze"""
    
    mock_db = Mock()
    mock_db.get_messages.return_value = [
        Mock(sender_type="customer", content="Problema urgente!", timestamp="2025-01-01"),
    ]
    
    handler = CommandHandler(mock_db, None, None)
    
    result = await handler.handle("conv_123", "@analyze", {})
    
    assert result["success"] == True
    assert "analysis" in result
    assert "formatted" in result
```

### Testes E2E (Fluxo Completo)

```python
@pytest.mark.asyncio
async def test_full_conversation_flow():
    """Testa fluxo completo: Cliente → Bot → Humano → @bot → Bot"""
    
    # Setup
    app_client = TestClient(app)
    conv_id = "test_conv_123"
    
    # 1. Cliente inicia conversa
    response = app_client.post("/webhook/message", json={
        "conversation_id": conv_id,
        "sender_type": "customer",
        "sender_id": "cust_456",
        "content": "Olá, preciso de ajuda"
    })
    assert response.status_code == 200
    
    # Verifica: Bot respondeu
    await asyncio.sleep(1)  # Aguarda processamento async
    conv = db.get_conversation(conv_id)
    assert conv.is_hil == False
    
    # 2. Humano intervém
    response = app_client.post("/webhook/message", json={
        "conversation_id": conv_id,
        "sender_type": "human_agent",
        "sender_id": "agent_789",
        "content": "Vou assumir daqui"
    })
    
    # Verifica: Humano assumiu
    await asyncio.sleep(1)
    conv = db.get_conversation(conv_id)
    assert conv.is_hil == True
    
    # 3. Humano pede análise
    response = app_client.post("/webhook/message", json={
        "conversation_id": conv_id,
        "sender_type": "human_agent",
        "sender_id": "agent_789",
        "content": "@analyze"
    })
    
    # Verifica: Análise retornada
    # (verificar via logs ou mock)
    
    # 4. Humano reativa bot
    response = app_client.post("/webhook/message", json={
        "conversation_id": conv_id,
        "sender_type": "human_agent",
        "sender_id": "agent_789",
        "content": "@bot Continue a conversa"
    })
    
    # Verifica: Bot reativado
    await asyncio.sleep(1)
    conv = db.get_conversation(conv_id)
    assert conv.is_hil == False

@pytest.mark.asyncio
async def test_timeout_flow():
    """Testa fluxo de timeout"""
    
    # Setup conversa com humano ativo
    conv_id = "timeout_test"
    db.create_conversation(conv_id, is_hil=True)
    
    # Simula inatividade de 31 minutos
    conv = db.get_conversation(conv_id)
    conv.last_activity_at = now() - timedelta(minutes=31)
    db.save(conv)
    
    # Executa check de timeout
    await check_all_timeouts()
    
    # Verifica: Conversa encerrada
    conv = db.get_conversation(conv_id)
    assert conv.closed == True
    assert "timeout" in conv.close_reason
```

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/conversations
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - COMPOSIO_API_KEY=${COMPOSIO_API_KEY}
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=conversations
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  worker:
    build: .
    command: python worker.py
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/conversations
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

### Observability com Logging

```python
# logging_config.py
import logging
import structlog
from datetime import datetime

def setup_logging():
    """Configura logging estruturado"""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
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

# Uso nos handlers
async def process_message(message: IncomingMessage):
    logger.info(
        "message_received",
        conversation_id=message.conversation_id,
        sender_type=message.sender_type,
        content_length=len(message.content)
    )
    
    try:
        # ... processamento
        
        logger.info(
            "message_processed",
            conversation_id=message.conversation_id,
            duration_ms=(end - start) * 1000
        )
    except Exception as e:
        logger.error(
            "message_processing_failed",
            conversation_id=message.conversation_id,
            error=str(e),
            exc_info=True
        )
        raise
```

### Métricas (Prometheus)

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Contadores
messages_received = Counter(
    'messages_received_total',
    'Total messages received',
    ['sender_type']
)

commands_executed = Counter(
    'commands_executed_total',
    'Total commands executed',
    ['command']
)

escalations = Counter(
    'escalations_total',
    'Total escalations to human',
    ['reason']
)

# Histogramas (latência)
message_processing_duration = Histogram(
    'message_processing_seconds',
    'Time to process message',
    ['sender_type']
)

command_execution_duration = Histogram(
    'command_execution_seconds',
    'Time to execute command',
    ['command']
)

# Gauges (estado atual)
active_conversations = Gauge(
    'active_conversations',
    'Currently active conversations'
)

bot_active_conversations = Gauge(
    'bot_active_conversations',
    'Conversations with bot active'
)

human_active_conversations = Gauge(
    'human_active_conversations',
    'Conversations with human active'
)

# Uso
async def process_message(message: IncomingMessage):
    messages_received.labels(sender_type=message.sender_type).inc()
    
    with message_processing_duration.labels(sender_type=message.sender_type).time():
        # ... processamento
        pass

@app.get("/metrics")
async def metrics():
    """Endpoint para Prometheus scraping"""
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")
```

### Dashboard Grafana (exemplo)

```json
{
  "dashboard": {
    "title": "Conversation Management",
    "panels": [
      {
        "title": "Active Conversations",
        "targets": [
          {
            "expr": "active_conversations"
          },
          {
            "expr": "bot_active_conversations"
          },
          {
            "expr": "human_active_conversations"
          }
        ]
      },
      {
        "title": "Message Processing Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(message_processing_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Escalation Rate",
        "targets": [
          {
            "expr": "rate(escalations_total[5m])"
          }
        ]
      },
      {
        "title": "Commands Usage",
        "targets": [
          {
            "expr": "sum by (command) (rate(commands_executed_total[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Tracing (OpenTelemetry)

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Uso
async def process_message(message: IncomingMessage):
    with tracer.start_as_current_span("process_message") as span:
        span.set_attribute("conversation_id", message.conversation_id)
        span.set_attribute("sender_type", message.sender_type)
        
        # Parse command
        with tracer.start_as_current_span("parse_command"):
            command, args = CommandParser.parse(message.content)
        
        if command:
            # Execute command
            with tracer.start_as_current_span("execute_command") as cmd_span:
                cmd_span.set_attribute("command", command)
                result = await command_handler.handle(conv_id, command, args)
```

## 7. Environment Variables

```python
def on_message_received(conversation_id, message):
    conv = db.get_conversation(conversation_id)
    
    # Se humano enviou → ativa HIL
    if message.sender_type == 'human_agent':
        if message.content.startswith('@bot'):
            # Reativa bot
            conv.is_hil = False
            contexto = sumarizar_desde_ultimo_bot(conversation_id)
            
            # Invoca LangGraph com contexto
            langgraph_agent.invoke({
                "messages": contexto,
                "instruction": message.content.replace('@bot', '').strip()
            })
        else:
            # Humano assume
            conv.is_hil = True
            iniciar_timeout(conversation_id)
    
    # Se cliente enviou
    elif message.sender_type == 'customer':
        if not conv.is_hil:
            # Delega para LangGraph processar
            result = langgraph_agent.invoke({
                "messages": [message]
            })
            
            # Se LangGraph escalou
            if result.get('escalate_to_human'):
                conv.is_hil = True
                iniciar_timeout(conversation_id)
        else:
            # Humano no loop, bot fica quieto
            pass
    
    db.save(conv)
```

## 4. Definição do LangGraph Agent

### Estrutura do Grafo com Composio

```python
from langgraph.graph import StateGraph, END
from composio_langgraph import ComposioToolSet, Action, App
from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    intent: str
    plan: List[str]
    execution_result: dict
    should_escalate: bool
    final_response: str
    composio_entity_id: str  # Identifica o usuário no Composio

def build_bot_graph():
    workflow = StateGraph(AgentState)
    
    # Inicializa Composio Toolset
    composio_toolset = ComposioToolSet()
    
    # Nós
    workflow.add_node("entender", entender_intent)
    workflow.add_node("planejar", criar_plano)
    workflow.add_node("executar", lambda state: executar_acoes(state, composio_toolset))
    workflow.add_node("validar", validar_resultado)
    workflow.add_node("responder", gerar_resposta)
    workflow.add_node("escalar", escalar_para_humano)
    
    # Fluxo principal
    workflow.set_entry_point("entender")
    workflow.add_edge("entender", "planejar")
    workflow.add_edge("planejar", "executar")
    
    # Decisão pós-execução
    workflow.add_conditional_edges(
        "executar",
        decidir_proximo_passo,
        {
            "validar": "validar",
            "escalar": "escalar"
        }
    )
    
    # Validação pode retry
    workflow.add_conditional_edges(
        "validar",
        verificar_validacao,
        {
            "responder": "responder",
            "planejar": "planejar"  # retry
        }
    )
    
    workflow.add_edge("responder", END)
    workflow.add_edge("escalar", END)
    
    return workflow.compile()

# Implementação dos nós
def entender_intent(state: AgentState) -> AgentState:
    """Usa LLM para classificar intenção do usuário"""
    messages = state["messages"]
    
    intent = llm.invoke([
        {"role": "system", "content": "Classifique a intenção do usuário"},
        {"role": "user", "content": messages[-1]["content"]}
    ])
    
    return {"intent": intent}

def criar_plano(state: AgentState) -> AgentState:
    """Cria plano usando Composio Actions disponíveis"""
    from composio import ComposioToolSet as ComposioClient
    
    intent = state["intent"]
    
    # Lista actions disponíveis do Composio
    client = ComposioClient()
    available_actions = client.get_actions(
        apps=[App.GMAIL, App.SALESFORCE, App.SLACK, App.ZENDESK]
    )
    
    # LLM escolhe quais actions usar
    planner_prompt = f"""
    Intenção do usuário: {intent}
    
    Actions disponíveis via Composio:
    {[action.name for action in available_actions]}
    
    Crie um plano de ação sequencial usando essas actions.
    Retorne como JSON: [{{"action": "...", "params": {{...}}}}]
    """
    
    plan = llm.invoke(planner_prompt)
    
    return {"plan": plan}

def executar_acoes(state: AgentState) -> AgentState:
    """Executa ferramentas via Composio"""
    from composio_langgraph import ComposioToolSet, Action
    
    plan = state["plan"]
    results = {}
    
    # Inicializa Composio
    composio_toolset = ComposioToolSet()
    
    for action in plan:
        try:
            if action == "buscar_pedido":
                # Usa integração com CRM/ERP via Composio
                result = composio_toolset.execute_action(
                    action=Action.SALESFORCE_GET_RECORD,
                    params={"record_id": state.get("order_id")},
                    entity_id=state.get("customer_id")
                )
                results["pedido"] = result
                
            elif action == "consultar_estoque":
                # Integração com sistema de inventário
                result = composio_toolset.execute_action(
                    action=Action.SHOPIFY_GET_INVENTORY,
                    params={"product_id": state.get("product_id")},
                    entity_id=state.get("customer_id")
                )
                results["estoque"] = result
                
            elif action == "enviar_email":
                # Composio gerencia autenticação do Gmail
                result = composio_toolset.execute_action(
                    action=Action.GMAIL_SEND_EMAIL,
                    params={
                        "to": state.get("email"),
                        "subject": state.get("subject"),
                        "body": state.get("body")
                    },
                    entity_id=state.get("customer_id")
                )
                results["email_sent"] = result
                
            elif action == "criar_ticket":
                # Integração com Zendesk/Jira
                result = composio_toolset.execute_action(
                    action=Action.ZENDESK_CREATE_TICKET,
                    params={
                        "subject": state.get("issue_subject"),
                        "description": state.get("issue_description"),
                        "priority": "high"
                    },
                    entity_id=state.get("customer_id")
                )
                results["ticket"] = result
                
        except Exception as e:
            results["error"] = str(e)
            break
    
    return {"execution_result": results}

def decidir_proximo_passo(state: AgentState) -> str:
    """Decide se continua ou escala"""
    result = state["execution_result"]
    
    # Condições para escalação
    if result.get("error"):
        return "escalar"
    if result.get("requires_human_approval"):
        return "escalar"
    
    return "validar"

def validar_resultado(state: AgentState) -> AgentState:
    """Valida se resultado satisfaz a intenção"""
    intent = state["intent"]
    result = state["execution_result"]
    
    # Validação pode ser por LLM ou regras
    validation = llm.invoke(f"""
    Intenção: {intent}
    Resultado: {result}
    
    O resultado satisfaz a intenção? Retorne sim/não e reasoning.
    """)
    
    return {"validation": validation}

def verificar_validacao(state: AgentState) -> str:
    """Decide se responde ou retenta"""
    validation = state.get("validation", {})
    
    if validation.get("success"):
        return "responder"
    else:
        return "planejar"  # retry

def gerar_resposta(state: AgentState) -> AgentState:
    """Gera resposta final para o usuário"""
    context = {
        "intent": state["intent"],
        "result": state["execution_result"]
    }
    
    response = llm.invoke(f"""
    Com base no contexto: {context}
    
    Gere uma resposta amigável e clara para o usuário.
    """)
    
    return {"final_response": response}

def escalar_para_humano(state: AgentState) -> AgentState:
    """Prepara escalação"""
    return {
        "should_escalate": True,
        "escalation_reason": state.get("execution_result", {}).get("error")
    }
```

### Integração com Webhook

```python
# Inicializa LangGraph uma vez
bot_graph = build_bot_graph()

def on_message_received(conversation_id, message):
    conv = db.get_conversation(conversation_id)
    
    if message.sender_type == 'human_agent':
        if message.content.startswith('@bot'):
            conv.is_hil = False
            context_messages = sumarizar_desde_ultimo_bot(conversation_id)
            
            # Invoca LangGraph
            result = bot_graph.invoke({
                "messages": context_messages + [
                    {"role": "user", "content": message.content.replace('@bot', '')}
                ]
            })
            
            send_message(conv.customer_id, result["final_response"])
        else:
            conv.is_hil = True
    
    elif message.sender_type == 'customer':
        if not conv.is_hil:
            # Invoca LangGraph
            result = bot_graph.invoke({
                "messages": get_conversation_history(conversation_id) + [
                    {"role": "user", "content": message.content}
                ]
            })
            
            # Verifica se escalou
            if result.get("should_escalate"):
                conv.is_hil = True
                notify_human_agent(conv.id, result.get("escalation_reason"))
            else:
                send_message(conv.customer_id, result["final_response"])
    
    db.save(conv)
```

```python
# Roda a cada 5 minutos
def check_timeouts():
    conversas_hil = db.query("""
        SELECT * FROM conversations 
        WHERE is_hil = true 
        AND last_activity_at < NOW() - INTERVAL '30 minutes'
    """)
    
    for conv in conversas_hil:
        bot.enviar_mensagem(
            conv.customer_id,
            "Desculpe, encerramos por inatividade."
        )
        arquivar_conversa(conv.id)
```

## 5. Monitor de Timeout (Job Separado)

```python
def sumarizar_desde_ultimo_bot(conversation_id):
    conv = db.get_conversation(conversation_id)
    
    mensagens = db.query("""
        SELECT * FROM messages 
        WHERE conversation_id = ? 
        AND timestamp > (
            SELECT timestamp FROM messages 
            WHERE id = ? 
        )
        ORDER BY timestamp ASC
    """, conversation_id, conv.last_bot_message_id)
    
    # Concatena mensagens ou usa LLM para sumarizar
    if len(mensagens) < 5:
        return "\n".join(m.content for m in mensagens)
    else:
        return llm.sumarizar(mensagens)
```

## 6. Função de Sumarização

### Cliente inicia conversa
```
1. Cliente envia mensagem
2. Cria conversation com is_hil=false
3. Bot responde
```

### Humano assume
```
1. Humano envia mensagem
2. Atualiza is_hil=true
3. Inicia timer de timeout
4. Bot para de responder
```

### Humano reativa bot
```
1. Humano digita "@bot [instrução]"
2. Sistema detecta @bot
3. Busca mensagens desde último bot
4. Sumariza contexto
5. Atualiza is_hil=false
6. Bot responde com contexto
```

### Timeout
```
1. Job detecta inatividade > 30min com is_hil=true
2. Bot envia mensagem de encerramento
3. Arquiva conversa
```

## 7. Fluxos Simplificados

```python
class ConversationManager:
    def __init__(self, db, bot, llm):
        self.db = db
        self.bot = bot
        self.llm = llm
    
    def handle_webhook(self, message):
        conv = self.db.get_or_create_conversation(message.customer_id)
        self.db.save_message(message)
        
        # Detecta tipo de mensagem
        if message.is_from_human_agent():
            self._handle_human_message(conv, message)
        elif message.is_from_customer():
            self._handle_customer_message(conv, message)
        
        # Atualiza timestamp
        conv.last_activity_at = now()
        self.db.save(conv)
    
    def _handle_human_message(self, conv, message):
        if message.starts_with('@bot'):
            # Reativa bot
            context = self._summarize_context(conv)
            instruction = message.content.replace('@bot', '').strip()
            
            conv.is_hil = False
            response = self.bot.respond(context, instruction)
            conv.last_bot_message_id = response.id
        else:
            # Humano assume
            conv.is_hil = True
    
    def _handle_customer_message(self, conv, message):
        if not conv.is_hil:
            # Bot responde
            response = self.bot.respond(message)
            conv.last_bot_message_id = response.id
        # else: humano no loop, bot ignora
    
    def _summarize_context(self, conv):
        messages = self.db.get_messages_since(
            conv.id, 
            conv.last_bot_message_id
        )
        
        if len(messages) <= 5:
            return '\n'.join(m.content for m in messages)
        
        return self.llm.summarize(messages)
    
    def check_timeouts(self):
        stalled = self.db.query("""
            SELECT * FROM conversations 
            WHERE is_hil = true 
            AND last_activity_at < NOW() - INTERVAL '30 minutes'
        """)
        
        for conv in stalled:
            self.bot.send(
                conv.customer_id,
                "Desculpe, encerramos por inatividade."
            )
            self.db.archive(conv.id)
```

## 8. Pseudocódigo Completo com LangGraph

```python
HUMAN_TIMEOUT_MINUTES = 30
SUMMARIZE_THRESHOLD = 5  # Se > 5 mensagens, usa LLM
BOT_COMMAND = '@bot'
```

## 10. Composio: Superpoderes para o Bot

### O que é Composio?

Composio é uma plataforma de integração que fornece:
- **150+ integrações** prontas (Gmail, Slack, Salesforce, Shopify, etc)
- **Gerenciamento de autenticação** (OAuth, API keys) por usuário
- **Actions padronizadas** para cada app
- **Execução segura** com rate limiting e retry

### Integrações Comuns para Atendimento

```python
from composio_langgraph import ComposioToolSet, Action, App

# Exemplos de actions disponíveis:

# CRM & Vendas
Action.SALESFORCE_GET_RECORD
Action.SALESFORCE_UPDATE_RECORD
Action.HUBSPOT_GET_CONTACT
Action.PIPEDRIVE_GET_DEAL

# E-commerce
Action.SHOPIFY_GET_ORDER
Action.SHOPIFY_GET_INVENTORY
Action.WOOCOMMERCE_GET_PRODUCT

# Comunicação
Action.GMAIL_SEND_EMAIL
Action.SLACK_SEND_MESSAGE
Action.DISCORD_SEND_MESSAGE
Action.WHATSAPP_SEND_MESSAGE

# Suporte
Action.ZENDESK_CREATE_TICKET
Action.ZENDESK_GET_TICKET
Action.JIRA_CREATE_ISSUE
Action.INTERCOM_CREATE_CONVERSATION

# Pagamentos
Action.STRIPE_GET_PAYMENT
Action.STRIPE_CREATE_REFUND
Action.PAYPAL_GET_TRANSACTION

# Produtividade
Action.GOOGLE_CALENDAR_CREATE_EVENT
Action.NOTION_CREATE_PAGE
Action.AIRTABLE_CREATE_RECORD
```

### Setup Composio

```python
# 1. Instalar
# pip install composio-langgraph

# 2. Autenticar (uma vez)
from composio import ComposioToolSet

toolset = ComposioToolSet()

# 3. Conectar apps (por usuário/entity)
# O bot redireciona usuário para OAuth flow
entity_id = "customer_12345"

# Conecta Gmail do usuário
connection = toolset.initiate_connection(
    app=App.GMAIL,
    entity_id=entity_id,
    redirect_url="https://yourapp.com/oauth/callback"
)

# 4. Usuário autoriza e Composio armazena tokens
# 5. Bot pode agora executar actions em nome do usuário
```

### Exemplo Completo: Bot com Composio

```python
from langgraph.graph import StateGraph, END
from composio_langgraph import ComposioToolSet, Action, App

def build_customer_support_bot():
    """Bot de suporte com poderes Composio"""
    
    toolset = ComposioToolSet()
    workflow = StateGraph(AgentState)
    
    def handle_order_inquiry(state: AgentState):
        """Consulta pedido no Shopify"""
        order_id = extract_order_id(state["messages"][-1]["content"])
        
        result = toolset.execute_action(
            action=Action.SHOPIFY_GET_ORDER,
            params={"order_id": order_id},
            entity_id=state["composio_entity_id"]
        )
        
        return {"execution_result": {"order": result}}
    
    def send_email_confirmation(state: AgentState):
        """Envia email via Gmail do agente"""
        
        result = toolset.execute_action(
            action=Action.GMAIL_SEND_EMAIL,
            params={
                "to": state["customer_email"],
                "subject": "Confirmação de Pedido",
                "body": generate_email_body(state)
            },
            entity_id="support_agent_entity"  # Usa conta do bot
        )
        
        return {"execution_result": {"email_sent": True}}
    
    def create_support_ticket(state: AgentState):
        """Cria ticket no Zendesk se necessário"""
        
        result = toolset.execute_action(
            action=Action.ZENDESK_CREATE_TICKET,
            params={
                "subject": state["issue_summary"],
                "description": state["issue_details"],
                "priority": "high",
                "tags": ["bot_escalation"]
            },
            entity_id="support_team_entity"
        )
        
        return {"execution_result": {"ticket": result}}
    
    # Adiciona nós ao grafo
    workflow.add_node("check_order", handle_order_inquiry)
    workflow.add_node("send_confirmation", send_email_confirmation)
    workflow.add_node("escalate_ticket", create_support_ticket)
    
    # ... resto do grafo
    
    return workflow.compile()
```

### Gerenciamento de Entity IDs

```python
class ConversationManager:
    def __init__(self, db, langgraph_agent, composio_toolset):
        self.db = db
        self.agent = langgraph_agent
        self.composio = composio_toolset
    
    def _handle_customer_message(self, conv, message):
        if not conv.is_hil:
            # Garante que customer tem entity_id no Composio
            entity_id = self._get_or_create_composio_entity(conv.customer_id)
            
            history = self.db.get_conversation_history(conv.id)
            
            result = self.agent.invoke({
                "messages": history + [{"role": "user", "content": message.content}],
                "composio_entity_id": entity_id
            })
            
            if result.get('should_escalate'):
                conv.is_hil = True
                self._notify_human(conv.id, result.get('escalation_reason'))
            else:
                self._send_response(conv, result['final_response'])
    
    def _get_or_create_composio_entity(self, customer_id):
        """Mapeia customer para Composio entity"""
        entity = self.db.get_composio_entity(customer_id)
        
        if not entity:
            # Cria entity no Composio
            entity_id = self.composio.create_entity(
                name=customer_id,
                metadata={"customer_id": customer_id}
            )
            
            self.db.save_composio_entity(customer_id, entity_id)
            return entity_id
        
        return entity.composio_entity_id
```

### Benefícios do Composio

✅ **Autenticação gerenciada**: Não precisa lidar com OAuth flows  
✅ **Rate limiting automático**: Composio gerencia limites das APIs  
✅ **Retry inteligente**: Falhas transitórias são retentadas  
✅ **Multi-tenant**: Cada customer pode ter suas próprias conexões  
✅ **Segurança**: Tokens nunca expostos ao seu código  
✅ **Logs & Monitoring**: Dashboard do Composio mostra todas executions  
✅ **Facilita testing**: Mock Composio actions facilmente

### Exemplo de Actions Compostas

```python
def process_refund_request(state: AgentState):
    """Workflow completo de reembolso usando múltiplas integrações"""
    
    toolset = ComposioToolSet()
    entity_id = state["composio_entity_id"]
    
    # 1. Busca pedido no Shopify
    order = toolset.execute_action(
        action=Action.SHOPIFY_GET_ORDER,
        params={"order_id": state["order_id"]},
        entity_id=entity_id
    )
    
    # 2. Valida elegibilidade
    if not is_refund_eligible(order):
        return {"should_escalate": True, "reason": "fora_da_politica"}
    
    # 3. Processa reembolso no Stripe
    refund = toolset.execute_action(
        action=Action.STRIPE_CREATE_REFUND,
        params={"payment_intent": order["payment_id"]},
        entity_id=entity_id
    )
    
    # 4. Atualiza pedido no Shopify
    toolset.execute_action(
        action=Action.SHOPIFY_UPDATE_ORDER,
        params={
            "order_id": state["order_id"],
            "financial_status": "refunded"
        },
        entity_id=entity_id
    )
    
    # 5. Envia email de confirmação
    toolset.execute_action(
        action=Action.GMAIL_SEND_EMAIL,
        params={
            "to": order["customer"]["email"],
            "subject": "Reembolso Processado",
            "body": f"Seu reembolso de {refund['amount']} foi processado."
        },
        entity_id="support_agent_entity"
    )
    
    # 6. Registra no Zendesk
    toolset.execute_action(
        action=Action.ZENDESK_ADD_COMMENT,
        params={
            "ticket_id": state.get("ticket_id"),
            "comment": f"Reembolso processado automaticamente: {refund['id']}"
        },
        entity_id="support_team_entity"
    )
    
    return {
        "execution_result": {
            "refund": refund,
            "success": True
        }
    }
```

## 11. Vantagens do LangGraph + Composio

### ✅ Separação de Responsabilidades
- **HIL Manager**: Apenas roteamento (is_hil true/false)
- **LangGraph**: Toda complexidade do agente (entender, planejar, executar)

### ✅ Testabilidade
```python
# Testa LangGraph isoladamente
def test_bot_graph():
    result = bot_graph.invoke({
        "messages": [{"role": "user", "content": "Onde está meu pedido #123?"}]
    })
    assert result["intent"] == "track_order"
    assert "pedido" in result["execution_result"]

# Testa HIL Manager isoladamente
def test_hil_routing():
    manager = ConversationManager(mock_db, mock_agent, mock_llm)
    # ...
```

### ✅ Observabilidade
LangGraph tem built-in tracing:
```python
from langgraph.checkpoint import MemorySaver

checkpointer = MemorySaver()
bot_graph = build_bot_graph().compile(checkpointer=checkpointer)

# Cada execução gera trace completo
result = bot_graph.invoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": conversation_id}}
)
```

### ✅ Flexibilidade
- Adicionar novos nós no grafo não afeta HIL Manager
- Mudar lógica de escalação fica contido no LangGraph
- Pode ter múltiplos grafos (bot_vendas, bot_suporte, etc)

### ✅ Escala
- LangGraph pode rodar em worker separado
- HIL Manager pode ser stateless
- Checkpoint permite pausar/resumir grafos

## 12. That's it!

**Arquitetura Final:**
- **HIL Manager** (simples) → Roteamento com `is_hil` flag
- **LangGraph Agent** (complexo) → Raciocínio e orquestração
- **Composio** (integração) → 150+ apps prontos para usar

**O bot agora pode:**
- ✅ Consultar CRMs (Salesforce, HubSpot)
- ✅ Verificar pedidos (Shopify, WooCommerce)
- ✅ Enviar emails/mensagens (Gmail, Slack, WhatsApp)
- ✅ Criar tickets (Zendesk, Jira)
- ✅ Processar pagamentos (Stripe, PayPal)
- ✅ E muito mais... tudo através de uma API unificada!

**Quando escalar para humano:**
- ❌ Action falhou após retries
- ❌ Fora da política da empresa
- ❌ Requer aprovação humana
- ❌ Múltiplos erros consecutivos
- ❌ Cliente solicita explicitamente

Simples, poderoso e escalável! 🚀