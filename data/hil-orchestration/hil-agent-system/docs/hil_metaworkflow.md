
🔄 Human-in-the-Loop (HIL) Integration
Architecture Overview
The HIL system operates as a meta-workflow layer above the agentic workflows:
┌─────────────────────────────────────────────────────────────────┐
│                    🎯 HIL Meta-Workflow Layer                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Conversation Orchestrator (with is_hil flag)             │  │
│  │                                                            │  │
│  │  if is_hil == false:                                       │  │
│  │    ├──> Execute Agent Workflow                            │  │
│  │    └──> Route to sink based on agent decision:            │  │
│  │         ├──> FINISH (conversation complete)               │  │
│  │         └──> HANDOVER (escalate to human)                 │  │
│  │                                                            │  │
│  │  if is_hil == true:                                        │  │
│  │    ├──> Route to Human Agent Queue                        │  │
│  │    ├──> Wait for human response                           │  │
│  │    └──> Update conversation state                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │   🤖 Agentic Workflow Layer        │
         │                                     │
         │   Agent Workflow Execution          │
         │   ├── Classify Intent               │
         │   ├── Process Request               │
         │   ├── Evaluate Confidence           │
         │   └── Decision:                     │
         │       ├── FINISH (success)          │
         │       └── HANDOVER (need human)     │
         └────────────────────────────────────┘
Conversation State Machine
┌──────────────┐
│   NEW        │  Conversation created
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ AI_PROCESSING│  is_hil=false, agent workflow executing
└──────┬───────┘
       │
       ├─────────────────┐
       │                 │
       ▼                 ▼
┌──────────────┐   ┌──────────────┐
│  FINISHED    │   │ HANDOVER     │  Agent requests human
└──────────────┘   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ PENDING_HUMAN│  is_hil=true
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ HUMAN_ACTIVE │  Human agent assigned
                   └──────┬───────┘
                          │
                          ├─────────────────┐
                          │                 │
                          ▼                 ▼
                   ┌──────────────┐   ┌──────────────┐
                   │  RESOLVED    │   │ BACK_TO_AI   │  Human re-routes to AI
                   └──────────────┘   └──────┬───────┘
                                             │
                                             └──> Back to AI_PROCESSING
Database Schema Extensions
sql-- =====================================
-- HIL System Tables
-- =====================================

-- Conversations (Extended)
ALTER TABLE conversations ADD COLUMN is_hil BOOLEAN DEFAULT false;
ALTER TABLE conversations ADD COLUMN hil_reason TEXT;
ALTER TABLE conversations ADD COLUMN assigned_agent_id UUID REFERENCES human_agents(id);
ALTER TABLE conversations ADD COLUMN handover_context JSONB;
ALTER TABLE conversations ADD COLUMN confidence_score FLOAT;

-- Human Agents
CREATE TABLE human_agents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK (status IN ('online', 'away', 'offline')),
  skills TEXT[] DEFAULT '{}',  -- e.g., ['returns', 'technical', 'billing']
  current_load INT DEFAULT 0,  -- Number of active conversations
  max_capacity INT DEFAULT 5,
  last_active TIMESTAMPTZ DEFAULT NOW(),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_human_agents_status ON human_agents(status);
CREATE INDEX idx_human_agents_skills ON human_agents USING GIN(skills);

-- Handover Events
CREATE TABLE handover_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  from_type TEXT NOT NULL CHECK (from_type IN ('ai', 'human')),
  to_type TEXT NOT NULL CHECK (to_type IN ('ai', 'human')),
  reason TEXT NOT NULL,
  context JSONB,
  agent_id UUID REFERENCES human_agents(id),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_handover_conversation ON handover_events(conversation_id);
CREATE INDEX idx_handover_created ON handover_events(created_at DESC);

-- Agent Assignment Queue
CREATE TABLE agent_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  priority INT DEFAULT 5,  -- 1 (highest) to 10 (lowest)
  required_skills TEXT[] DEFAULT '{}',
  context JSONB,
  assigned_to UUID REFERENCES human_agents(id),
  assigned_at TIMESTAMPTZ,
  status TEXT NOT NULL CHECK (status IN ('pending', 'assigned', 'completed', 'cancelled')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  CONSTRAINT unique_active_queue UNIQUE (conversation_id) WHERE status IN ('pending', 'assigned')
);

CREATE INDEX idx_queue_status ON agent_queue(status, priority, created_at);
CREATE INDEX idx_queue_assigned ON agent_queue(assigned_to) WHERE status = 'assigned';

-- Conversation Analytics
CREATE TABLE conversation_analytics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations(id),
  
  -- Timing metrics
  total_duration_seconds INT,
  ai_duration_seconds INT,
  human_duration_seconds INT,
  wait_time_seconds INT,  -- Time in queue
  
  -- Interaction metrics
  total_messages INT,
  ai_messages INT,
  human_messages INT,
  handover_count INT,
  
  -- Quality metrics
  customer_satisfaction FLOAT,  -- 1-5
  resolution_status TEXT CHECK (resolution_status IN ('resolved', 'unresolved', 'escalated')),
  
  -- Cost metrics
  ai_cost_usd FLOAT,
  human_cost_usd FLOAT,  -- Based on agent hourly rate
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_analytics_conversation ON conversation_analytics(conversation_id);
CREATE INDEX idx_analytics_created ON conversation_analytics(created_at DESC);
Agent Workflow with Sink Nodes
Extended Workflow Configuration:
yaml# config/workflows/customer_support_hil.yaml

name: customer_support_hil
version: 2.0.0
description: "Customer support with HIL capabilities"

# HIL Configuration
hil_config:
  enabled: true
  handover_triggers:
    - type: confidence_threshold
      threshold: 0.7  # Handover if confidence < 70%
    
    - type: explicit_request
      patterns: ["speak to human", "talk to agent", "escalate"]
    
    - type: agent_decision
      # Agent explicitly returns HANDOVER sink
    
    - type: error_threshold
      consecutive_errors: 3
    
    - type: complexity
      # Complex issues identified by intent classifier

  routing_rules:
    - skill: "returns"
      priority: 5
      max_wait_time: 300  # 5 minutes
    
    - skill: "technical"
      priority: 3
      max_wait_time: 180  # 3 minutes
    
    - skill: "billing"
      priority: 7
      max_wait_time: 600  # 10 minutes

workflow:
  nodes:
    # ===== Entry Point =====
    - id: classify_intent
      agent: simple_intent_classifier
      config:
        model_profile: fast
        output_schema:
          intent: string
          confidence: float
          complexity: string  # low, medium, high
          needs_human: boolean
    
    # ===== Confidence Check =====
    - id: check_confidence
      agent: simple_evaluator
      config:
        model_profile: fast
      prompt_template: |
        Evaluate if AI can handle this confidently:
        
        Intent: {{classify_intent.intent}}
        Confidence: {{classify_intent.confidence}}
        Complexity: {{classify_intent.complexity}}
        
        Return: {can_handle: true/false, reason: "..."}
    
    # ===== AI Handling Path =====
    - id: handle_with_ai
      agent: reasoning_support_agent
      config:
        model_profile: balanced
        tools: [shopify.*, gmail.send_email]
        max_iterations: 10
      condition:
        type: jmespath
        expression: "check_confidence.output.can_handle == true"
    
    # ===== Evaluation & Sinks =====
    - id: evaluate_result
      agent: simple_evaluator
      config:
        model_profile: fast
      prompt_template: |
        Evaluate the AI's handling of this request:
        
        Original request: {{customer_message}}
        AI response: {{handle_with_ai.answer}}
        Success: {{handle_with_ai.success}}
        
        Determine:
        1. Was the request fully resolved? (yes/no)
        2. Is customer likely satisfied? (yes/no)
        3. Should we handover to human? (yes/no)
        
        Return: {
          resolved: boolean,
          satisfaction_likely: boolean,
          needs_handover: boolean,
          reason: string
        }
    
    # ===== SINK: Finish (Success) =====
    - id: FINISH
      type: sink
      action: finish_conversation
      config:
        status: resolved
        update_analytics: true
      condition:
        type: jmespath
        expression: |
          evaluate_result.output.resolved == true && 
          evaluate_result.output.needs_handover == false
    
    # ===== SINK: Handover to Human =====
    - id: HANDOVER
      type: sink
      action: handover_to_human
      config:
        # Determine required skills based on intent
        required_skills: |
          {% if classify_intent.intent == 'return_request' %}['returns', 'order_management']
          {% elif classify_intent.intent == 'technical_issue' %}['technical']
          {% elif classify_intent.intent == 'billing' %}['billing', 'payments']
          {% else %}['general']
          {% endif %}
        
        # Priority calculation
        priority: |
          {% if classify_intent.complexity == 'high' %}3
          {% elif classify_intent.complexity == 'medium' %}5
          {% else %}7
          {% endif %}
        
        # Handover context for human agent
        handover_context:
          original_intent: "{{classify_intent.intent}}"
          confidence: "{{classify_intent.confidence}}"
          ai_attempted: true
          ai_result: "{{handle_with_ai}}"
          customer_history: "{{memory_context}}"
          reason: "{{evaluate_result.output.reason}}"
      
      condition:
        type: python_expr
        expression: |
          (check_confidence.output.can_handle == false) or
          (evaluate_result.output.needs_handover == true) or
          (classify_intent.output.needs_human == true)
  
  # ===== Workflow Edges =====
  edges:
    - from: classify_intent
      to: check_confidence
    
    - from: check_confidence
      to: handle_with_ai
      condition:
        type: jmespath
        expression: "output.can_handle == true"
    
    - from: check_confidence
      to: HANDOVER
      condition:
        type: jmespath
        expression: "output.can_handle == false"
    
    - from: handle_with_ai
      to: evaluate_result
    
    - from: evaluate_result
      to: FINISH
      condition:
        type: jmespath
        expression: "output.resolved == true && output.needs_handover == false"
    
    - from: evaluate_result
      to: HANDOVER
      condition:
        type: jmespath
        expression: "output.needs_handover == true"
HIL Orchestration Service
python# app/services/hil_orchestrator.py

from typing import Dict, Any, Optional
from enum import Enum

class ConversationStatus(str, Enum):
    NEW = "new"
    AI_PROCESSING = "ai_processing"
    FINISHED = "finished"
    HANDOVER = "handover"
    PENDING_HUMAN = "pending_human"
    HUMAN_ACTIVE = "human_active"
    RESOLVED = "resolved"
    BACK_TO_AI = "back_to_ai"


class HILOrchestrator:
    """
    Meta-orchestrator for Human-in-the-Loop workflows.
    Manages conversation state and routing between AI and humans.
    """
    
    def __init__(
        self,
        db_pool,
        agent_orchestrator: AgentOrchestrator,
        queue_manager: "QueueManager"
    ):
        self.db = db_pool
        self.agent_orchestrator = agent_orchestrator
        self.queue_manager = queue_manager
    
    async def handle_message(
        self,
        conversation_id: str,
        message: str,
        sender_type: str = "user"
    ) -> Dict[str, Any]:
        """
        Main entry point for handling messages.
        Routes based on conversation state (is_hil flag).
        """
        
        # Get conversation state
        conversation = await self._get_conversation(conversation_id)
        
        if conversation is None:
            # New conversation
            conversation = await self._create_conversation(conversation_id)
        
        # Store message
        await self._store_message(conversation_id, sender_type, message)
        
        # Route based on state
        if conversation["is_hil"]:
            # Human is handling
            return await self._handle_human_conversation(
                conversation_id,
                message,
                sender_type
            )
        else:
            # AI is handling
            return await self._handle_ai_conversation(
                conversation_id,
                message
            )
    
    async def _handle_ai_conversation(
        self,
        conversation_id: str,
        message: str
    ) -> Dict[str, Any]:
        """
        Handle conversation with AI workflow.
        Executes agent workflow and processes sink decisions.
        """
        
        # Update status
        await self._update_conversation_status(
            conversation_id,
            ConversationStatus.AI_PROCESSING
        )
        
        # Get conversation context
        context = await self._get_conversation_context(conversation_id)
        
        try:
            # Execute agent workflow
            result = await self.agent_orchestrator.execute_workflow(
                workflow_id="customer_support_hil",
                input_data={"customer_message": message},
                context=context
            )
            
            # Process sink decision
            sink_node = self._identify_sink_node(result)
            
            if sink_node == "FINISH":
                # AI successfully handled - finish conversation
                await self._finish_conversation(conversation_id, result)
                
                return {
                    "status": "finished",
                    "response": result["outputs"]["evaluate_result"]["response"],
                    "handled_by": "ai",
                    "confidence": result["outputs"]["classify_intent"]["confidence"]
                }
            
            elif sink_node == "HANDOVER":
                # AI requests handover - route to human
                await self._handover_to_human(conversation_id, result)
                
                return {
                    "status": "handover",
                    "message": "Your request has been escalated to a human agent. You'll be connected shortly.",
                    "handled_by": "ai",
                    "reason": result["outputs"]["evaluate_result"].get("reason", "Complex request")
                }
            
            else:
                # Unexpected - treat as error and handover
                logger.error("workflow_no_sink", conversation_id=conversation_id)
                await self._handover_to_human(
                    conversation_id,
                    result,
                    reason="Workflow completed without clear sink"
                )
                
                return {
                    "status": "handover",
                    "message": "Let me connect you with a human agent.",
                    "handled_by": "ai",
                    "reason": "Error in AI processing"
                }
        
        except Exception as e:
            logger.error(
                "ai_workflow_error",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True
            )
            
            # Error - handover to human
            await self._handover_to_human(
                conversation_id,
                None,
                reason=f"AI error: {str(e)}"
            )
            
            return {
                "status": "handover",
                "message": "I'm having trouble processing your request. Let me connect you with a human agent.",
                "handled_by": "ai",
                "reason": "System error"
            }
    
    async def _handle_human_conversation(
        self,
        conversation_id: str,
        message: str,
        sender_type: str
    ) -> Dict[str, Any]:
        """
        Handle conversation in human mode.
        """
        
        conversation = await self._get_conversation(conversation_id)
        
        if conversation["status"] == ConversationStatus.PENDING_HUMAN:
            # Still waiting for assignment
            return {
                "status": "pending_human",
                "message": "You're in the queue. A human agent will be with you shortly.",
                "queue_position": await self.queue_manager.get_position(conversation_id)
            }
        
        elif conversation["status"] == ConversationStatus.HUMAN_ACTIVE:
            # Human agent assigned
            if sender_type == "user":
                # Customer message - notify assigned agent
                await self._notify_human_agent(
                    conversation["assigned_agent_id"],
                    conversation_id,
                    message
                )
                
                return {
                    "status": "human_active",
                    "message": "Message sent to agent",
                    "agent_name": await self._get_agent_name(conversation["assigned_agent_id"])
                }
            
            elif sender_type == "agent":
                # Agent response - send to customer
                return {
                    "status": "human_active",
                    "response": message,
                    "handled_by": "human",
                    "agent_name": await self._get_agent_name(conversation["assigned_agent_id"])
                }
        
        return {"status": "error", "message": "Invalid conversation state"}
    
    async def _finish_conversation(
        self,
        conversation_id: str,
        workflow_result: Dict[str, Any]
    ):
        """
        Mark conversation as finished.
        Update analytics and state.
        """
        
        # Update conversation
        await self.db.execute("""
            UPDATE conversations
            SET 
                status = $1,
                closed_at = NOW(),
                updated_at = NOW()
            WHERE id = $2
        """, ConversationStatus.FINISHED, conversation_id)
        
        # Record analytics
        await self._record_analytics(
            conversation_id,
            workflow_result,
            resolved=True,
            handled_by="ai"
        )
        
        logger.info(
            "conversation_finished",
            conversation_id=conversation_id,
            handled_by="ai"
        )
    
    async def _handover_to_human(
        self,
        conversation_id: str,
        workflow_result: Optional[Dict[str, Any]],
        reason: str = None
    ):
        """
        Handover conversation to human agent.
        Creates queue entry and updates conversation state.
        """
        
        # Extract handover config from workflow
        if workflow_result:
            handover_config = workflow_result.get("outputs", {}).get("HANDOVER", {}).get("config", {})
            required_skills = handover_config.get("required_skills", ["general"])
            priority = handover_config.get("priority", 5)
            handover_context = handover_config.get("handover_context", {})
            reason = reason or handover_context.get("reason", "AI handover")
        else:
            required_skills = ["general"]
            priority = 5
            handover_context = {}
        
        # Update conversation
        await self.db.execute("""
            UPDATE conversations
            SET 
                is_hil = true,
                status = $1,
                hil_reason = $2,
                handover_context = $3,
                updated_at = NOW()
            WHERE id = $4
        """, ConversationStatus.PENDING_HUMAN, reason, json.dumps(handover_context), conversation_id)
        
        # Create handover event
        await self.db.execute("""
            INSERT INTO handover_events
            (conversation_id, from_type, to_type, reason, context)
            VALUES ($1, 'ai', 'human', $2, $3)
        """, conversation_id, reason, json.dumps(handover_context))
        
        # Add to queue
        await self.queue_manager.enqueue(
            conversation_id=conversation_id,
            priority=priority,
            required_skills=required_skills,
            context=handover_context
        )
        
        logger.info(
            "conversation_handover",
            conversation_id=conversation_id,
            reason=reason,
            priority=priority,
            required_skills=required_skills
        )
    
    def _identify_sink_node(self, workflow_result: Dict[str, Any]) -> Optional[str]:
        """
        Identify which sink node was reached in workflow.
        """
        
        executed_nodes = workflow_result.get("executed_nodes", [])
        
        if "FINISH" in executed_nodes:
            return "FINISH"
        elif "HANDOVER" in executed_nodes:
            return "HANDOVER"
        
        return None
    
    async def _record_analytics(
        self,
        conversation_id: str,
        workflow_result: Dict[str, Any],
        resolved: bool,
        handled_by: str
    ):
        """Record conversation analytics"""
        
        # Calculate metrics
        messages = await self.db.fetch(
            "SELECT * FROM messages WHERE conversation_id = $1",
            conversation_id
        )
        
        total_messages = len(messages)
        ai_messages = len([m for m in messages if m["sender_type"] == "agent"])
        human_messages = len([m for m in messages if m["sender_type"] == "user"])
        
        # Get timing
        conversation = await self._get_conversation(conversation_id)
        duration = (datetime.now() - conversation["created_at"]).total_seconds()
        
        # Get cost
        ai_cost = workflow_result.get("trace", {}).get("costs", {}).get("total_usd", 0.0)
        
        await self.db.execute("""
            INSERT INTO conversation_analytics
            (conversation_id, total_duration_seconds, ai_duration_seconds,
             total_messages, ai_messages, human_messages,
             resolution_status, ai_cost_usd)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, conversation_id, duration, duration if handled_by == "ai" else 0,
            total_messages, ai_messages, human_messages,
            "resolved" if resolved else "unresolved", ai_cost)
    
    async def _get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID"""
        row = await self.db.fetchrow(
            "SELECT * FROM conversations WHERE id = $1",
            conversation_id
        )
        return dict(row) if row else None
    
    async def _create_conversation(self, conversation_id: str) -> Dict:
        """Create new conversation"""
        await self.db.execute("""
            INSERT INTO conversations
            (id, status, is_hil)
            VALUES ($1, $2, false)
        """, conversation_id, ConversationStatus.NEW)
        
        return await self._get_conversation(conversation_id)
    
    async def _update_conversation_status(
        self,
        conversation_id: str,
        status: ConversationStatus
    ):
        """Update conversation status"""
        await self.db.execute("""
            UPDATE conversations
            SET status = $1, updated_at = NOW()
            WHERE id = $2
        """, status, conversation_id)
Queue Management System
python# app/services/queue_manager.py

class QueueManager:
    """
    Manages agent assignment queue.
    Routes conversations to available human agents based on skills and priority.
    """
    
    def __init__(self, db_pool, redis_client):
        self.db = db_pool
        self.redis = redis_client
    
    async def enqueue(
        self,
        conversation_id: str,
        priority: int,
        required_skills: List[str],
        context: Dict[str, Any]
    ) -> str:
        """
        Add conversation to queue.
        Returns queue entry ID.
        """
        
        queue_id = str(uuid.uuid4())
        
        await self.db.execute("""
            INSERT INTO agent_queue
            (id, conversation_id, priority, required_skills, context, status)
            VALUES ($1, $2, $3, $4, $5, 'pending')
        """, queue_id, conversation_id, priority, required_skills, json.dumps(context))
        
        # Try immediate assignment
        await self._try_assign(queue_id)
        
        return queue_id
    
    async def _try_assign(self, queue_id: str):
        """
        Try to assign queue entry to available agent.
        Uses skill matching and load balancing.
        """
        
        # Get queue entry
        entry = await self.db.fetchrow(
            "SELECT * FROM agent_queue WHERE id = $1",
            queue_id
        )
        
        if not entry or entry["status"] != "pending":
            return
        
        # Find available agents with required skills
        agents = await self.db.fetch("""
            SELECT *
            FROM human_agents
            WHERE status = 'online'
            AND current_load < max_capacity
            AND required_skills <@ skills  -- All required skills present
            ORDER BY current_load ASC, last_active DESC
            LIMIT 1
        """, entry["required_skills"])
        
        if not agents:
            # No available agents - notify via webhook/websocket
            await self._notify_no_agents(entry)
            return
        
        agent = agents[0]
        
        # Assign conversation
        async with self.db.transaction():
            # Update queue entry
            await self.db.execute("""
                UPDATE agent_queue
                SET 
                    assigned_to = $1,
                    assigned_at = NOW(),
                    status = 'assigned'
                WHERE id = $2
            """, agent["id"], queue_id)
            
            # Update conversation
            await self.db.execute("""
                UPDATE conversations
                SET 
                    assigned_agent_id = $1,
                    status = 'human_active',
                    updated_at = NOW()
                WHERE id = $2
            """, agent["id"], entry["conversation_id"])
            
            # Update agent load
            await self.db.execute("""
                UPDATE human_agents
                SET current_load = current_load + 1
                WHERE id = $1
            """, agent["id"])
        
        # Notify agent
        await self._notify_agent_assignment(agent["id"], entry["conversation_id"])
        
        logger.info(
            "conversation_assigned",
            conversation_id=entry["conversation_id"],
            agent_id=agent["id"],
            agent_name=agent["name"]
        )
    
    async def get_position(self, conversation_id: str) -> int:
        """Get position in queue"""
        
        entry = await self.db.fetchrow("""
            SELECT created_at, priority
            FROM agent_queue
            WHERE conversation_id = $1 AND status = 'pending'
        """, conversation_id)
        
        if not entry:
            return 0
        
        # Count entries ahead in queue
        position = await self.db.fetchval("""
            SELECT COUNT(*)
            FROM agent_queue
            WHERE status = 'pending'
            AND (
                priority < $1
                OR (priority = $1 AND created_at < $2)
            )
        """, entry["priority"], entry["created_at"])
        
        return position + 1
    
    async def _notify_agent_assignment(
        self,
        agent_id: str,
        conversation_id: str
    ):
        """Notify agent of new assignment via WebSocket/webhook"""
        
        # Publish to Redis for WebSocket delivery
        await self.redis.publish(
            f"agent:{agent_id}:assignments",
            json.dumps({
                "type": "new_assignment",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat()
            })
        )
API Endpoints for HIL
python# app/api/hil.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/v1/hil", tags=["hil"])

@router.post("/message")
async def handle_message(
    conversation_id: str,
    message: str,
    sender_type: str = "user",
    hil_orchestrator: HILOrchestrator = Depends(get_hil_orchestrator)
):
    """
    Main endpoint for handling messages.
    Routes to AI or human based on conversation state.
    """
    
    result = await hil_orchestrator.handle_message(
        conversation_id=conversation_id,
        message=message,
        sender_type=sender_type
    )
    
    return result


@router.post("/handover/{conversation_id}")
async def manual_handover(
    conversation_id: str,
    reason: str,
    hil_orchestrator: HILOrchestrator = Depends(get_hil_orchestrator)
):
    """
    Manually trigger handover to human.
    Used by agents or system rules.
    """
    
    await hil_orchestrator._handover_to_human(
        conversation_id=conversation_id,
        workflow_result=None,
        reason=reason
    )
    
    return {"status": "handover_initiated", "conversation_id": conversation_id}


@router.post("/resolve/{conversation_id}")
async def resolve_conversation(
    conversation_id: str,
    resolution_notes: str,
    agent_id: str,
    hil_orchestrator: HILOrchestrator = Depends(get_hil_orchestrator),
    db = Depends(get_db)
):
    """
    Human agent resolves conversation.
    """
    
    # Update conversation
    await db.execute("""
        UPDATE conversations
        SET 
            status = 'resolved',
            closed_at =