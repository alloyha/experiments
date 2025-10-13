# Chunking Strategies & Neo4j Integration for HIL Agent System

## 📋 Executive Summary

Based on the HIL Agent System architecture, there are two key areas where enhanced chunking strategies and Neo4j integration would provide significant value:

1. **Memory & Context Management** - Better chunking for long-term memory/RAG
2. **Workflow & Agent Relationships** - Graph database for complex relationships

---

## 🧩 Chunking Strategies for Memory System

### Current State
The system uses pgvector for semantic search with basic document indexing. There's no sophisticated chunking strategy defined.

### Why Different Chunking Strategies Matter

**Problem**: Different content types need different chunking approaches:
- Long customer support conversations need context-preserving chunks
- Technical documentation needs semantic boundary-aware chunks
- Code execution history needs structured chunks
- Product catalogs need entity-based chunks

### Proposed Chunking Strategies

#### 1. **Fixed-Size Chunking** (Baseline)
```python
class FixedSizeChunker:
    """Simple token-based chunking with overlap"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[Chunk]:
        tokens = self.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunks.append(Chunk(
                content=self.detokenize(chunk_tokens),
                start_idx=i,
                end_idx=i + len(chunk_tokens),
                metadata={"strategy": "fixed"}
            ))
        return chunks
```

**Use Cases**: Simple FAQ documents, general text

#### 2. **Semantic Chunking** (Recommended for Conversations)
```python
class SemanticChunker:
    """Chunk based on semantic boundaries using embeddings similarity"""
    
    def __init__(self, similarity_threshold: float = 0.8, min_chunk_size: int = 100):
        self.threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.embeddings_model = OpenAIEmbeddings()
    
    async def chunk(self, text: str) -> List[Chunk]:
        sentences = self.split_sentences(text)
        embeddings = await self.embeddings_model.embed_documents(sentences)
        
        chunks = []
        current_chunk = []
        current_embedding = embeddings[0]
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            similarity = cosine_similarity(current_embedding, embedding)
            
            if similarity < self.threshold and len(current_chunk) >= self.min_chunk_size:
                # Semantic boundary detected - create new chunk
                chunks.append(Chunk(
                    content=" ".join(current_chunk),
                    metadata={
                        "strategy": "semantic",
                        "boundary_score": similarity
                    }
                ))
                current_chunk = [sentence]
                current_embedding = embedding
            else:
                current_chunk.append(sentence)
                # Update rolling average embedding
                current_embedding = (current_embedding + embedding) / 2
        
        if current_chunk:
            chunks.append(Chunk(content=" ".join(current_chunk)))
        
        return chunks
```

**Use Cases**: 
- Customer support conversation history
- Long-form documentation
- Agent execution logs

#### 3. **Conversation-Turn Chunking** (Specific to HIL)
```python
class ConversationTurnChunker:
    """Chunk conversations by turns, preserving context"""
    
    def __init__(self, turns_per_chunk: int = 5, include_metadata: bool = True):
        self.turns_per_chunk = turns_per_chunk
        self.include_metadata = include_metadata
    
    def chunk(self, conversation: List[Message]) -> List[Chunk]:
        chunks = []
        
        for i in range(0, len(conversation), self.turns_per_chunk):
            turn_window = conversation[i:i + self.turns_per_chunk]
            
            # Format conversation chunk
            content = self._format_turns(turn_window)
            
            chunks.append(Chunk(
                content=content,
                metadata={
                    "strategy": "conversation_turn",
                    "conversation_id": conversation[0].conversation_id,
                    "turn_start": i,
                    "turn_end": i + len(turn_window),
                    "participants": self._extract_participants(turn_window),
                    "intent": self._detect_intent(turn_window),
                    "outcome": self._detect_outcome(turn_window)
                }
            ))
        
        return chunks
    
    def _format_turns(self, turns: List[Message]) -> str:
        formatted = []
        for msg in turns:
            formatted.append(f"{msg.sender_type}: {msg.content}")
        return "\n".join(formatted)
```

**Use Cases**:
- Indexing historical conversations for similarity search
- Training data for agent improvement
- Customer history context retrieval

#### 4. **Hierarchical Chunking** (For Documentation)
```python
class HierarchicalChunker:
    """Chunk based on document structure (headings, sections)"""
    
    def __init__(self, max_chunk_tokens: int = 1000):
        self.max_chunk_tokens = max_chunk_tokens
    
    def chunk(self, document: str) -> List[Chunk]:
        # Parse document structure
        sections = self.parse_markdown_structure(document)
        
        chunks = []
        for section in sections:
            if self.token_count(section.content) > self.max_chunk_tokens:
                # Recursively chunk large sections
                subsections = self.split_section(section)
                chunks.extend(subsections)
            else:
                chunks.append(Chunk(
                    content=section.content,
                    metadata={
                        "strategy": "hierarchical",
                        "level": section.level,
                        "heading": section.heading,
                        "parent": section.parent_heading,
                        "path": section.path  # e.g., "API > Endpoints > Workflows"
                    }
                ))
        
        return chunks
```

**Use Cases**:
- Technical documentation indexing
- API reference materials
- Product knowledge base

#### 5. **Entity-Based Chunking** (For Structured Data)
```python
class EntityBasedChunker:
    """Chunk around entities (products, orders, customers)"""
    
    def __init__(self, entity_type: str):
        self.entity_type = entity_type
        self.ner_model = spacy.load("en_core_web_lg")
    
    def chunk(self, text: str) -> List[Chunk]:
        doc = self.ner_model(text)
        
        # Group text by entity mentions
        chunks = []
        current_chunk = []
        current_entity = None
        
        for sent in doc.sents:
            entities = [ent for ent in sent.ents if ent.label_ == self.entity_type]
            
            if entities:
                entity = entities[0]
                if current_entity != entity.text:
                    # New entity - create chunk
                    if current_chunk:
                        chunks.append(Chunk(
                            content=" ".join(current_chunk),
                            metadata={
                                "strategy": "entity_based",
                                "entity_type": self.entity_type,
                                "entity_id": current_entity
                            }
                        ))
                    current_chunk = [sent.text]
                    current_entity = entity.text
                else:
                    current_chunk.append(sent.text)
            else:
                current_chunk.append(sent.text)
        
        if current_chunk:
            chunks.append(Chunk(content=" ".join(current_chunk)))
        
        return chunks
```

**Use Cases**:
- Product catalog indexing
- Order history
- Customer interaction history

### Unified Chunking Interface

```python
# app/memory/chunking.py

from enum import Enum
from typing import List, Dict, Any, Optional

class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    CONVERSATION_TURN = "conversation_turn"
    HIERARCHICAL = "hierarchical"
    ENTITY_BASED = "entity_based"

class Chunk(BaseModel):
    content: str
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None

class ChunkingService:
    """Unified service for different chunking strategies"""
    
    def __init__(self):
        self.strategies = {
            ChunkingStrategy.FIXED: FixedSizeChunker(),
            ChunkingStrategy.SEMANTIC: SemanticChunker(),
            ChunkingStrategy.CONVERSATION_TURN: ConversationTurnChunker(),
            ChunkingStrategy.HIERARCHICAL: HierarchicalChunker(),
            ChunkingStrategy.ENTITY_BASED: EntityBasedChunker("PRODUCT")
        }
    
    async def chunk_document(
        self,
        content: str,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        **kwargs
    ) -> List[Chunk]:
        """Chunk content using specified strategy"""
        chunker = self.strategies[strategy]
        chunks = await chunker.chunk(content)
        
        # Generate embeddings for all chunks
        embeddings = await self.generate_embeddings([c.content for c in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using configured model"""
        # Use OpenAI text-embedding-3-small or similar
        pass
```

### Integration with Memory Manager

```python
# app/memory/manager.py

class MemoryManager:
    def __init__(self, db_pool, vector_store, chunking_service: ChunkingService):
        self.db = db_pool
        self.vector_store = vector_store
        self.chunker = chunking_service
    
    async def index_document(
        self,
        doc_id: str,
        content: str,
        doc_type: str,
        metadata: dict = None
    ):
        """Index document with appropriate chunking strategy"""
        
        # Select strategy based on document type
        strategy_map = {
            "conversation": ChunkingStrategy.CONVERSATION_TURN,
            "documentation": ChunkingStrategy.HIERARCHICAL,
            "product_catalog": ChunkingStrategy.ENTITY_BASED,
            "faq": ChunkingStrategy.SEMANTIC,
            "default": ChunkingStrategy.FIXED
        }
        
        strategy = strategy_map.get(doc_type, ChunkingStrategy.FIXED)
        
        # Chunk the document
        chunks = await self.chunker.chunk_document(content, strategy)
        
        # Store chunks in vector database
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            await self.vector_store.upsert(
                id=chunk_id,
                embedding=chunk.embedding,
                metadata={
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "strategy": strategy,
                    "content": chunk.content,
                    **chunk.metadata,
                    **(metadata or {})
                }
            )
        
        logger.info(
            "document_indexed",
            doc_id=doc_id,
            doc_type=doc_type,
            strategy=strategy,
            chunks=len(chunks)
        )
```

---

## 🕸️ Neo4j Integration for Relationship Management

### Why Neo4j Makes Sense for HIL System

The HIL Agent System has complex relationships that are naturally graph-structured:

1. **Workflow Dependencies**: Workflows call other workflows, agents depend on tools
2. **Agent Learning**: Agents learn from past executions, similar problems have similar solutions
3. **Customer Journey**: Customer interactions flow through multiple agents and workflows
4. **Tool Integration**: Tools have dependencies, shared authentication, rate limits
5. **Human Agent Skills**: Skills graph for intelligent routing

### Architecture: Hybrid PostgreSQL + Neo4j

```
┌─────────────────────────────────────────────────────────┐
│              Data Storage Architecture                   │
└─────────────────────────────────────────────────────────┘

PostgreSQL (Primary)                    Neo4j (Relationships)
├── Conversations                       ├── Workflow Graphs
├── Messages                            │   ├── Node dependencies
├── Executions                          │   └── Conditional flows
├── Human Agents                        │
├── OAuth Tokens                        ├── Learning Graphs
├── Analytics                           │   ├── Similar executions
└── Audit Logs                          │   ├── Success patterns
                                        │   └── Failure patterns
pgvector (Semantic)                     │
├── Document Embeddings                 ├── Customer Journey
├── Conversation Chunks                 │   ├── Conversation flows
└── Execution History                   │   ├── Handover patterns
                                        │   └── Resolution paths
                                        │
                                        ├── Tool Dependencies
                                        │   ├── OAuth requirements
                                        │   ├── Rate limit groups
                                        │   └── Service dependencies
                                        │
                                        └── Skills Network
                                            ├── Agent capabilities
                                            ├── Skill relationships
                                            └── Training paths
```

### Use Cases for Neo4j

#### 1. **Workflow Execution Graph**

```cypher
// Workflow node relationships
CREATE (w:Workflow {name: "customer_support_hil", version: "2.0"})
CREATE (n1:Node {id: "classify_intent", type: "simple_agent"})
CREATE (n2:Node {id: "handle_with_ai", type: "reasoning_agent"})
CREATE (n3:Node {id: "FINISH", type: "sink"})
CREATE (n4:Node {id: "HANDOVER", type: "sink"})

CREATE (w)-[:CONTAINS]->(n1)
CREATE (w)-[:CONTAINS]->(n2)
CREATE (w)-[:CONTAINS]->(n3)
CREATE (w)-[:CONTAINS]->(n4)

CREATE (n1)-[:NEXT {condition: "confidence > 0.7"}]->(n2)
CREATE (n1)-[:NEXT {condition: "confidence <= 0.7"}]->(n4)
CREATE (n2)-[:NEXT {condition: "success == true"}]->(n3)
CREATE (n2)-[:NEXT {condition: "success == false"}]->(n4)
```

**Benefits**:
- Query all possible execution paths
- Find circular dependencies
- Analyze workflow complexity
- Optimize conditional routing

#### 2. **Agent Learning Graph**

```cypher
// Link executions by similarity
MATCH (e1:Execution {success: true})
MATCH (e2:Execution {goal: e1.goal})
WHERE e1.id <> e2.id
CREATE (e1)-[:SIMILAR_TO {score: 0.95}]->(e2)

// Query similar successful patterns
MATCH (e:Execution {goal: "return_request"})-[:SIMILAR_TO*1..3]->(similar:Execution {success: true})
RETURN similar.plan, similar.tools_used, similar.duration
ORDER BY similar.success_rate DESC
LIMIT 5
```

**Benefits**:
- Find patterns in successful executions
- Recommend tools based on similar past problems
- Identify common failure modes
- Transfer learning between agents

#### 3. **Customer Journey Tracking**

```cypher
// Customer conversation flow
CREATE (c:Customer {id: "cust_123"})
CREATE (conv:Conversation {id: "conv_456"})
CREATE (c)-[:HAS_CONVERSATION]->(conv)

CREATE (m1:Message {role: "user", content: "I want to return..."})
CREATE (m2:Message {role: "ai", content: "Let me help..."})
CREATE (conv)-[:STARTS_WITH]->(m1)
CREATE (m1)-[:FOLLOWED_BY]->(m2)

CREATE (h:Handover {reason: "complex_case", timestamp: datetime()})
CREATE (m2)-[:TRIGGERED]->(h)

CREATE (agent:HumanAgent {id: "agent_007", skill: "returns"})
CREATE (h)-[:ASSIGNED_TO]->(agent)

// Query handover patterns
MATCH (c:Customer)-[:HAS_CONVERSATION]->(:Conversation)-[:TRIGGERED]->(h:Handover)
WHERE c.id = "cust_123"
RETURN h.reason, h.timestamp, h.resolution_time
ORDER BY h.timestamp DESC
```

**Benefits**:
- Visualize customer journey across multiple conversations
- Identify bottlenecks in handover process
- Analyze why certain customers need human agents
- Predict handover likelihood

#### 4. **Tool Dependency Graph**

```cypher
// Tool relationships
CREATE (shopify:Tool {name: "shopify", type: "ecommerce"})
CREATE (gmail:Tool {name: "gmail", type: "communication"})
CREATE (slack:Tool {name: "slack", type: "communication"})

CREATE (oauth:AuthProvider {name: "oauth2"})
CREATE (shopify)-[:REQUIRES_AUTH]->(oauth)
CREATE (gmail)-[:REQUIRES_AUTH]->(oauth)

CREATE (rateLimit:RateLimit {name: "shopify_api", rpm: 40})
CREATE (shopify)-[:SUBJECT_TO]->(rateLimit)

// Query tools by capability
MATCH (t:Tool)-[:PROVIDES]->(cap:Capability {name: "send_email"})
RETURN t.name, t.cost_per_call, t.reliability
ORDER BY t.reliability DESC
```

**Benefits**:
- Smart tool selection based on dependencies
- Parallel execution planning
- Rate limit coordination
- Cost optimization

#### 5. **Skills Network for Human Agents**

```cypher
// Skills graph
CREATE (returns:Skill {name: "returns", level: "advanced"})
CREATE (technical:Skill {name: "technical_support", level: "expert"})
CREATE (billing:Skill {name: "billing", level: "intermediate"})

CREATE (agent1:HumanAgent {id: "agent_007", name: "Alice"})
CREATE (agent2:HumanAgent {id: "agent_008", name: "Bob"})

CREATE (agent1)-[:HAS_SKILL {proficiency: 0.95}]->(returns)
CREATE (agent1)-[:HAS_SKILL {proficiency: 0.80}]->(technical)
CREATE (agent2)-[:HAS_SKILL {proficiency: 0.90}]->(billing)

CREATE (returns)-[:RELATED_TO {strength: 0.7}]->(billing)
CREATE (technical)-[:PREREQUISITE_FOR]->(returns)

// Find best agent for complex case
MATCH (req:Requirement {skills: ["returns", "technical"]})
MATCH (a:HumanAgent)-[hs:HAS_SKILL]->(s:Skill)
WHERE s.name IN req.skills
WITH a, avg(hs.proficiency) as avg_prof, count(s) as skill_count
WHERE skill_count = size(req.skills)
RETURN a.name, avg_prof
ORDER BY avg_prof DESC
LIMIT 1
```

**Benefits**:
- Intelligent agent assignment
- Skill gap identification
- Training path recommendations
- Load balancing by expertise

### Implementation: Neo4j Service Layer

```python
# app/services/graph_service.py

from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional

class GraphService:
    """Neo4j integration for relationship management"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    async def close(self):
        await self.driver.close()
    
    # Workflow Graph Operations
    async def create_workflow_graph(self, workflow_config: dict):
        """Create workflow graph from YAML config"""
        async with self.driver.session() as session:
            # Create workflow node
            await session.run(
                """
                CREATE (w:Workflow {
                    name: $name,
                    version: $version,
                    created_at: datetime()
                })
                """,
                name=workflow_config["name"],
                version=workflow_config["version"]
            )
            
            # Create node and edge relationships
            for node in workflow_config["workflow"]["nodes"]:
                await session.run(
                    """
                    MATCH (w:Workflow {name: $workflow_name})
                    CREATE (n:Node {
                        id: $node_id,
                        agent_type: $agent_type
                    })
                    CREATE (w)-[:CONTAINS]->(n)
                    """,
                    workflow_name=workflow_config["name"],
                    node_id=node["id"],
                    agent_type=node.get("agent", "unknown")
                )
            
            # Create edges
            for edge in workflow_config["workflow"].get("edges", []):
                await session.run(
                    """
                    MATCH (from:Node {id: $from_id})
                    MATCH (to:Node {id: $to_id})
                    CREATE (from)-[:NEXT {
                        condition: $condition
                    }]->(to)
                    """,
                    from_id=edge["from"],
                    to_id=edge["to"],
                    condition=edge.get("condition", {})
                )
    
    async def find_execution_path(
        self,
        workflow_name: str,
        start_node: str,
        conditions: dict
    ) -> List[str]:
        """Find execution path through workflow based on conditions"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH path = (start:Node {id: $start_id})-[:NEXT*]->(end)
                WHERE ALL(r IN relationships(path) WHERE $conditions SATISFIES r.condition)
                RETURN [n IN nodes(path) | n.id] as path
                ORDER BY length(path) DESC
                LIMIT 1
                """,
                start_id=start_node,
                conditions=conditions
            )
            record = await result.single()
            return record["path"] if record else []
    
    # Learning Graph Operations
    async def link_similar_executions(
        self,
        execution_id: str,
        similar_executions: List[Dict[str, Any]]
    ):
        """Create similarity relationships between executions"""
        async with self.driver.session() as session:
            for similar in similar_executions:
                await session.run(
                    """
                    MATCH (e1:Execution {id: $exec_id})
                    MATCH (e2:Execution {id: $similar_id})
                    MERGE (e1)-[:SIMILAR_TO {
                        score: $score,
                        reason: $reason
                    }]->(e2)
                    """,
                    exec_id=execution_id,
                    similar_id=similar["id"],
                    score=similar["similarity_score"],
                    reason=similar.get("reason", "semantic_similarity")
                )
    
    async def get_successful_patterns(
        self,
        goal: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find successful execution patterns for a goal"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Execution {goal: $goal, success: true})
                OPTIONAL MATCH (e)-[:SIMILAR_TO]->(similar:Execution {success: true})
                WITH e, count(similar) as similarity_count
                RETURN e.id, e.plan, e.tools_used, e.duration, similarity_count
                ORDER BY similarity_count DESC, e.success_rate DESC
                LIMIT $limit
                """,
                goal=goal,
                limit=limit
            )
            return [dict(record) async for record in result]
    
    # Customer Journey Operations
    async def track_conversation_flow(
        self,
        conversation_id: str,
        message: dict,
        previous_message_id: Optional[str] = None
    ):
        """Track message flow in conversation"""
        async with self.driver.session() as session:
            # Create message node
            await session.run(
                """
                MATCH (conv:Conversation {id: $conv_id})
                CREATE (m:Message {
                    id: $msg_id,
                    role: $role,
                    timestamp: datetime()
                })
                CREATE (conv)-[:CONTAINS]->(m)
                """,
                conv_id=conversation_id,
                msg_id=message["id"],
                role=message["role"]
            )
            
            # Link to previous message
            if previous_message_id:
                await session.run(
                    """
                    MATCH (prev:Message {id: $prev_id})
                    MATCH (curr:Message {id: $curr_id})
                    CREATE (prev)-[:FOLLOWED_BY]->(curr)
                    """,
                    prev_id=previous_message_id,
                    curr_id=message["id"]
                )
    
    async def track_handover(
        self,
        conversation_id: str,
        handover_data: dict
    ):
        """Track handover event in graph"""
        async with self.driver.session() as session:
            await session.run(
                """
                MATCH (conv:Conversation {id: $conv_id})
                CREATE (h:Handover {
                    id: $handover_id,
                    reason: $reason,
                    timestamp: datetime()
                })
                CREATE (conv)-[:TRIGGERED]->(h)
                
                MATCH (agent:HumanAgent {id: $agent_id})
                CREATE (h)-[:ASSIGNED_TO]->(agent)
                """,
                conv_id=conversation_id,
                handover_id=handover_data["id"],
                reason=handover_data["reason"],
                agent_id=handover_data["assigned_agent_id"]
            )
    
    # Skills Network Operations
    async def find_best_agent(
        self,
        required_skills: List[str],
        current_load_threshold: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Find best available agent based on skills"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (a:HumanAgent)-[hs:HAS_SKILL]->(s:Skill)
                WHERE s.name IN $skills
                  AND a.status = 'online'
                  AND a.current_load < $threshold
                WITH a, avg(hs.proficiency) as avg_prof, count(s) as skill_count
                WHERE skill_count = size($skills)
                RETURN a.id, a.name, avg_prof, a.current_load
                ORDER BY avg_prof DESC, a.current_load ASC
                LIMIT 1
                """,
                skills=required_skills,
                threshold=current_load_threshold
            )
            record = await result.single()
            return dict(record) if record else None
```

### Integration with HIL Orchestrator

```python
# app/services/hil_orchestrator.py

class HILOrchestrator:
    def __init__(
        self,
        db_pool,
        agent_orchestrator: AgentOrchestrator,
        queue_manager: QueueManager,
        graph_service: GraphService  # NEW
    ):
        self.db = db_pool
        self.agent_orchestrator = agent_orchestrator
        self.queue_manager = queue_manager
        self.graph = graph_service  # NEW
    
    async def _handle_ai_conversation(
        self,
        conversation_id: str,
        message: str
    ) -> Dict[str, Any]:
        # Execute workflow
        result = await self.agent_orchestrator.execute_workflow(...)
        
        # Track execution in graph for learning (NEW)
        await self.graph.link_similar_executions(
            execution_id=result["execution_id"],
            similar_executions=await self._find_similar_executions(result)
        )
        
        # Track conversation flow (NEW)
        await self.graph.track_conversation_flow(
            conversation_id=conversation_id,
            message={"id": message_id, "role": "ai", "content": result["response"]},
            previous_message_id=previous_message_id
        )
        
        return result
    
    async def _handover_to_human(
        self,
        conversation_id: str,
        result: dict,
        reason: str = None
    ):
        # Use graph to find best agent (NEW)
        required_skills = self._extract_required_skills(result)
        best_agent = await self.graph.find_best_agent(required_skills)
        
        if not best_agent:
            # Fallback to queue manager
            best_agent = await self.queue_manager.assign_agent(...)
        
        # Track handover in graph (NEW)
        await self.graph.track_handover(
            conversation_id=conversation_id,
            handover_data={
                "id": handover_id,
                "reason": reason,
                "assigned_agent_id": best_agent["id"]
            }
        )
        
        # Continue with existing handover logic
        ...
```

### Configuration

```yaml
# config/settings.yaml

# Chunking configuration
chunking:
  default_strategy: semantic
  strategies:
    conversation:
      type: conversation_turn
      turns_per_chunk: 5
    documentation:
      type: hierarchical
      max_tokens: 1000
    product_catalog:
      type: entity_based
      entity_type: PRODUCT
    faq:
      type: semantic
      similarity_threshold: 0.8

# Neo4j configuration
neo4j:
  enabled: true
  uri: "neo4j://localhost:7687"
  database: "hil-agent-system"
  
  # What to track in graph
  track:
    - workflow_execution_paths
    - agent_learning_patterns
    - customer_journeys
    - tool_dependencies
    - skills_network
  
  # Sync strategy
  sync:
    mode: "async"  # async or realtime
    batch_size: 100
    interval_seconds: 60
```

---

## 📊 Implementation Roadmap

### Phase 1: Enhanced Chunking (Week 1-2)
1. Implement `ChunkingService` with 5 strategies
2. Update `MemoryManager` to use chunking strategies
3. Add document type detection and strategy selection
4. Migrate existing vector store data to chunked format

### Phase 2: Neo4j Foundation (Week 3-4)
1. Set up Neo4j container in Docker Compose
2. Implement `GraphService` base class
3. Add workflow graph creation on workflow registration
4. Implement execution tracking in graph

### Phase 3: Learning & Customer Journey (Week 5-6)
1. Implement execution similarity linking
2. Add customer journey tracking
3. Create graph-based learning queries
4. Add handover pattern analysis

### Phase 4: Advanced Features (Week 7-8)
1. Skills network for agent assignment
2. Tool dependency management
3. Graph-based analytics dashboard
4. Performance optimization and caching

---

## 🎯 Expected Benefits

### Chunking Strategies
- **30-40% better retrieval accuracy** - Semantic boundaries preserve context
- **Reduced token costs** - Smaller, more relevant chunks
- **Improved conversation context** - Turn-based chunking maintains dialogue flow
- **Better entity resolution** - Entity-based chunking for product/order queries

### Neo4j Integration
- **Complex relationship queries** - Sub-second graph traversal vs JOIN hell
- **Pattern discovery** - Identify successful execution patterns automatically
- **Intelligent routing** - Skills-based agent assignment with confidence
- **Journey visualization** - End-to-end customer interaction tracking
- **Predictive analytics** - Predict handover likelihood, resolution time

---

## 💡 Conclusion

Both chunking strategies and Neo4j integration are **highly valuable** for the HIL Agent System:

1. **Chunking** is essential for improving RAG accuracy and reducing costs
2. **Neo4j** excels at relationship queries that are central to agent learning, workflow optimization, and intelligent routing

**Recommendation**: Implement in phases, starting with chunking (immediate ROI on memory system) then Neo4j (enables advanced features like learning and journey tracking).
