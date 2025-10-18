"""
Agentic Text Clusterizer

Lightweight module header describing the public purpose and surface API.

Purpose:
    Provide a scalable, parallel-capable text clustering utility that
    combines fast lexical retrieval and optional semantic embeddings to
    assign texts to categories, create new categories when needed, and
    consolidate similar categories.

Key features:
    - Hybrid retrieval (lexical prefilter + optional semantic rerank)
    - Multi-signal confidence scoring (LLM scores, lexical, semantic, keywords)
    - Parallel batch processing with configurable pass/round behavior
    - Category consolidation to merge semantically similar categories
    - Pluggable configuration: use preset configurations or supply a custom one

Public API (high level):
    - clusterize_texts(texts, config=..., batch_size=..., max_passes=...)
        Main entry point. Returns a result containing categories, assignments,
        and metadata.

    - clusterize_texts_with_chunking(...)
        Convenience wrapper for handling very large/multi-topic texts by
        splitting into semantic chunks, classifying chunks, and aggregating
        results.

    - CONFIG_* presets
        A set of recommended configuration presets for common behaviors
        (balanced, conservative, aggressive, semantic-only, etc.).

Usage example (async):
    result = await clusterize_texts(texts, config=CONFIG_BALANCED_HYBRID)

"""

from __future__ import annotations
import asyncio
import hashlib
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_ai import Agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from configuration_manager import (
    ClusterConfig,
    RETRIEVAL_MODE, 
    ConfidenceWeights, 
    CONFIDENCE_WEIGHTS_BERT, 
    CONFIDENCE_WEIGHTS_NO_BERT,
    DEFAULT_CONFIG,
    CONFIG_BALANCED_HYBRID,
    CONFIG_SEMANTIC_BERT,
    CONFIG_BALANCED_TFIDF,
)

VERBOSE=True 

logger = logging.getLogger("agentic_clusterizer")

if VERBOSE:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)

# ------------------
# Constants
# ------------------

# Processing settings
DEFAULT_MAX_PASSES = 2              # Number of clustering passes
DEFAULT_BATCH_SIZE = 20             # Base batch size
DEFAULT_TOKEN_BIG_TEXT = 200        # Token threshold to consider text "big"
DEFAULT_MAX_PARALLEL_MERGES = 4     # Max parallel merges in tree merge
DEFAULT_TEXTS_PER_WORKFLOW = 50     # Texts per workflow in tree merge
DEFAULT_MAX_CONCURRENT_LLM_CALLS = 5
DEFAULT_PREFILTER_K = 3
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0

# LLM Configuration
# Further providers:
#  - 'huggingface': Community-driven models
#  - 'cohere': Another provider for LLMs
#  - 'anthropic': Focused on safety and alignment
#  - 'gemini': Known for high-quality models
# Further models:
#  - 'gpt-4o-mini': Balanced cost/performance
#  - 'gpt-4o': Higher performance, higher cost
#  - 'gpt-4o-turbo': Optimized for speed and cost

LLM_PROVIDER_NAME = 'openai'
LLM_MODEL_NAME = 'gpt-4o-mini'

# Default BERT model for embeddings
# Further options:
#   - 'all-mpnet-base-v2': Slower, better quality
#   - 'all-distilroberta-v1': Faster, lower quality
#   - 'all-MiniLM-L6-v2': Balanced option
DEFAULT_BERT_MODEL = 'all-MiniLM-L6-v2' 



# ------------------
# Pydantic models
# ------------------
class Category(BaseModel):
    id: str = Field(..., description="Unique ID")
    name: str = Field(..., description="Category name")
    description: str = Field("", description="Description")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    text_count: int = Field(0, description="Number of texts assigned")
    
    @field_validator('keywords', mode='before')
    @classmethod
    def parse_keywords(cls, v):
        """Convert comma-separated string to list if needed."""
        if isinstance(v, str):
            # Split by comma and clean whitespace
            return [kw.strip() for kw in v.split(',') if kw.strip()]
        return v if v is not None else []

class CategoryAssignment(BaseModel):
    text: str
    category_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

class MultiPassAnalysis(BaseModel):
    text: str
    candidate_categories: List[str] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    best_category_id: Optional[str] = None
    should_create_new: bool = False
    new_category: Optional[Dict[str, Any]] = None
    reasoning: str = ""

# ------------------
# Agent definitions
# ------------------
# Initialize agents conditionally to avoid API key errors
multi_pass_analyzer = None
category_consolidator = None

try:
    multi_pass_analyzer = Agent(
        f'{LLM_PROVIDER_NAME}:{LLM_MODEL_NAME}',
        output_type=MultiPassAnalysis,
        system_prompt="""You are a category advisor analyzing text for categorization.

    IMPORTANT: You must provide REALISTIC confidence scores (0.0-1.0) for each candidate category.
    - High confidence (0.8-1.0): Perfect match, clear semantic alignment
    - Medium confidence (0.5-0.7): Good match, reasonable fit
    - Low confidence (0.0-0.4): Poor match, weak connection

    Given a text and candidate categories, return:
    1. candidate_categories: List of category IDs you considered
    2. confidence_scores: Dict mapping category_id -> confidence (BE SPECIFIC, not all 0.5!)
    3. best_category_id: Your top choice OR
    4. should_create_new: True if no good match exists (confidence < 0.6 for all)
    5. new_category: {id, name, description, keywords} if creating new
    6. reasoning: Brief explanation

    BE PRECISE with confidence scores - use the full 0.0-1.0 range!"""
    )

    category_consolidator = Agent(
        f'{LLM_PROVIDER_NAME}:{LLM_MODEL_NAME}',
        output_type=dict,
        system_prompt="""You are a consolidation expert. Given categories, return:
    {should_merge:bool, category_pairs:[[id1,id2],...], merged_categories:[{id,name,description,keywords}]}.
    Only propose merges when strongly justified."""
    )
    print(f"✓ LLM agents initialized with {LLM_PROVIDER_NAME}:{LLM_MODEL_NAME}")
except Exception as e:
    print(f"⚠ LLM initialization failed: {e}")
    print("  Continuing with fallback mode (no LLM calls)")
    # Can't modify constant, but the check below will handle None agents

# ------------------
# BERT Integration
# ------------------
class BERTEncoder:
    """Wrapper for BERT-based sentence embeddings."""
    
    _instance = None
    _model = None
    
    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_BERT_MODEL):
        """Singleton pattern for model loading."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def __init__(self, model_name: str = DEFAULT_BERT_MODEL):
        """
        Initialize BERT encoder.
        
        Recommended models:
        - 'all-MiniLM-L6-v2': Fast, good quality (default)
        - 'all-mpnet-base-v2': Slower, better quality
        - 'paraphrase-MiniLM-L6-v2': Optimized for paraphrase detection
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading BERT model: {model_name}")
            self._model = SentenceTransformer(model_name)
            logger.info("✓ BERT model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    def encode(self, texts: List[str] | str, **kwargs) -> np.ndarray:
        """Encode text(s) to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kwargs
        )
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        return float(cosine_similarity(emb1, emb2)[0, 0])

# ------------------
# Utility functions
# ------------------
def encode_text_to_id(text: str) -> str:
    """Generate deterministic hash ID from text."""
    return hashlib.sha1(text.encode()).hexdigest()

def deterministic_cat_id(name: str) -> str:
    """Create deterministic category ID from name."""
    slug = "".join(ch for ch in (name or "cat").lower() if ch.isalnum())[:24] or "cat"
    digest = encode_text_to_id(name or "")[:6]
    return f"{slug}_{digest}"

def normalize_category_id(cat_id: str) -> str:
    """Normalize category IDs to ensure consistency."""
    # If it's a simple numeric ID from LLM, convert to proper format
    if cat_id and cat_id.isdigit():
        return f"cat_{cat_id.zfill(3)}"  # Convert '1' to 'cat_001'
    elif cat_id and len(cat_id) <= 3 and cat_id.replace('0', '').isdigit():
        return f"cat_{cat_id.zfill(3)}"  # Convert '001' to 'cat_001'
    return cat_id  # Keep existing format if already complex

def clamp_confidence(value: Any) -> float:
    """Safely clamp confidence value to [0.0, 1.0]."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (ValueError, TypeError):
        return 0.0

def extract_confidence(scores: Dict[str, float], key: Optional[str]) -> float:
    """Safely extract and validate confidence score."""
    if not scores or key is None:
        return 0.0
    return clamp_confidence(scores.get(key, 0.0))

def calculate_keyword_overlap(text: str, category: Category) -> float:
    """Calculate keyword overlap ratio between text and category."""
    if not category.keywords:
        return 0.0
    
    text_words = set(word.lower() for word in text.split() if len(word) > 3)
    category_words = set(word.lower() for word in category.keywords)
    
    if not text_words or not category_words:
        return 0.0
    
    overlap = len(text_words & category_words)
    
    if len(category_words) <= 3:
        return min(1.0, overlap / max(1, len(category_words)))
    else:
        return min(1.0, overlap / (len(category_words) * 0.4))

def calculate_category_maturity(category: Category, total_categories: int) -> float:
    """Calculate category maturity score based on text count."""
    if total_categories <= 1:
        return 1.0
    
    text_count = category.text_count or 0
    
    if text_count == 0:
        return 0.6
    elif text_count == 1:
        return 0.7
    elif text_count <= 3:
        return 0.8
    elif text_count <= 5:
        return 0.9
    else:
        return 0.95

def calculate_pass_bonus(current_pass: int, max_passes: int) -> float:
    """Calculate bonus for later passes."""
    if max_passes <= 1:
        return 1.0
    return 0.5 + (0.5 * (current_pass / max_passes))

def calculate_confidence(
    text: str,
    category: Category,
    llm_confidence: float,
    tfidf_similarity: float,
    current_pass: int,
    max_passes: int,
    total_categories: int,
    bert_similarity: Optional[float] = None,
    weights: Optional[ConfidenceWeights] = None,
    debug: bool = False
) -> float:
    """
    Calculate enhanced confidence score using multiple factors.
    
    New: Includes optional BERT similarity for semantic understanding.
    """
    # Choose weights based on whether BERT is available and provided weights
    if weights is None:
        weights = CONFIDENCE_WEIGHTS_BERT if bert_similarity is not None else CONFIDENCE_WEIGHTS_NO_BERT
    
    weights_dict = weights.to_dict()
    
    # Factor 1: LLM confidence
    llm_score = clamp_confidence(llm_confidence)
    
    # Factor 2: TF-IDF similarity
    tfidf_score = clamp_confidence(tfidf_similarity)
    
    # Factor 3: BERT similarity (NEW!)
    bert_score = clamp_confidence(bert_similarity) if bert_similarity is not None else 0.0
    
    # Factor 4: Keyword overlap
    keyword_score = calculate_keyword_overlap(text, category)
    
    # Factor 5: Category maturity
    maturity_score = calculate_category_maturity(category, total_categories)
    
    # Factor 6: Pass bonus
    pass_score = calculate_pass_bonus(current_pass, max_passes)
    
    # Weighted combination
    confidence = (
        weights_dict['llm_score'] * llm_score +
        weights_dict['tfidf_similarity'] * tfidf_score +
        weights_dict.get('bert_similarity', 0.0) * bert_score +
        weights_dict['keyword_overlap'] * keyword_score +
        weights_dict['category_maturity'] * maturity_score +
        weights_dict['pass_number'] * pass_score
    )
    
    if debug:
        debug_msg = (
            f"  Confidence breakdown: LLM={llm_score:.2f}({weights_dict['llm_score']:.0%}) "
            f"TFIDF={tfidf_score:.2f}({weights_dict['tfidf_similarity']:.0%}) "
        )
        if bert_similarity is not None:
            debug_msg += f"BERT={bert_score:.2f}({weights_dict['bert_similarity']:.0%}) "
        debug_msg += (
            f"Keywords={keyword_score:.2f}({weights_dict['keyword_overlap']:.0%}) "
            f"Maturity={maturity_score:.2f}({weights_dict['category_maturity']:.0%}) "
            f"Pass={pass_score:.2f}({weights_dict['pass_number']:.0%}) "
            f"→ Final={confidence:.2f}"
        )
        logger.debug(debug_msg)
    
    return clamp_confidence(confidence)

def build_category_from_raw(raw: Any, fallback_name: str, fallback_text: str) -> Category:
    """Convert raw data to Category with fallback handling."""
    if isinstance(raw, Category):
        return _ensure_category_id(raw, fallback_name, fallback_text)
    
    if isinstance(raw, dict):
        return _category_from_dict(raw, fallback_name, fallback_text)
    
    return _category_from_attributes(raw, fallback_name, fallback_text)

def _ensure_category_id(cat: Category, fallback_name: str, fallback_text: str) -> Category:
    """Ensure category has valid ID."""
    if not cat.id:
        cat.id = deterministic_cat_id(cat.name or fallback_name or fallback_text[:20])
    return cat

def _category_from_dict(raw_dict: Dict[str, Any], fallback_name: str, fallback_text: str) -> Category:
    """Create category from dictionary."""
    try:
        # Ensure required fields exist
        if 'name' not in raw_dict or not raw_dict['name']:
            raw_dict['name'] = fallback_name or fallback_text[:20] or "unnamed_category"
        
        if 'id' not in raw_dict or not raw_dict['id']:
            raw_dict['id'] = deterministic_cat_id(raw_dict['name'])
        
        cat = Category(**raw_dict)
        return _ensure_category_id(cat, fallback_name, fallback_text)
    except ValidationError as e:
        logger.warning("Category validation failed: %s", e)
        return _create_fallback_category(raw_dict, fallback_name, fallback_text)

def _category_from_attributes(raw: Any, fallback_name: str, fallback_text: str) -> Category:
    """Create category from object with attributes."""
    try:
        raw_dict = {
            k: getattr(raw, k) 
            for k in ("id", "name", "description", "keywords", "text_count") 
            if hasattr(raw, k)
        }
        return Category(**raw_dict)
    except ValidationError as e:
        logger.warning("Category from attributes failed: %s", e)
        return _create_fallback_category(raw, fallback_name, fallback_text)

def _create_fallback_category(raw: Any, fallback_name: str, fallback_text: str) -> Category:
    """Create fallback category when validation fails."""
    name = getattr(raw, "name", None) or getattr(raw, "id", None) or fallback_name or fallback_text[:20]
    cid = deterministic_cat_id(name)
    return Category(
        id=cid,
        name=name,
        description=getattr(raw, "description", "") or "Auto-created",
        keywords=list(getattr(raw, "keywords", []) or [])[:8],
        text_count=0
    )

def calculate_adaptive_batch_size(num_texts: int, base_batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """
    Calculate optimal batch size based on dataset size.
    
    Strategy:
    - Small datasets (< 100): Use smaller batches for precision
    - Medium datasets (100-1000): Use base batch size  
    - Large datasets (> 1000): Scale up to reduce total iterations
    """
    if num_texts < 100:
        return min(10, base_batch_size)
    elif num_texts < 1000:
        return base_batch_size
    else:
        # Scale up for large datasets: aim for ~50-100 total batches max
        optimal_batch_size = max(base_batch_size, num_texts // 80)
        return min(optimal_batch_size, 100)  # Cap at 100 to avoid memory issues

# ------------------
# Hybrid TF-IDF + BERT Indexer
# ------------------
class CategoryIndexer:
    """
    Hybrid TF-IDF + BERT indexer for efficient category retrieval.
    
    Modes:
    - 'tfidf': Fast lexical matching (original)
    - 'bert': Semantic matching via embeddings (slower, better)
    - 'hybrid': Two-stage cascade (recommended) - TF-IDF prefilter + BERT rerank
    """
    
    def __init__(
        self, 
        mode: str = RETRIEVAL_MODE.HYBRID,
        bert_model: str = DEFAULT_BERT_MODEL,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 5000,
        tfidf_prefilter_k: int = 20
    ):
        self.mode = mode
        self.tfidf_prefilter_k = tfidf_prefilter_k
        
        # TF-IDF components
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self._tfidf_fitted = False
        self._tfidf_X = None
        
        # BERT components
        self.bert_encoder = None
        self.bert_embeddings = None
        
        if mode in [RETRIEVAL_MODE.BERT, RETRIEVAL_MODE.HYBRID]:
            try:
                self.bert_encoder = BERTEncoder.get_instance(bert_model)
            except Exception as e:
                logger.warning(f"BERT initialization failed, falling back to TF-IDF: {e}")
                self.mode = RETRIEVAL_MODE.TFIDF
        
        # Common
        self._ids = []
        self._categories = []
        self._fitted = False

    def fit(self, categories: List[Category]) -> None:
        """Fit indexer on category descriptions."""
        if not categories:
            self._reset()
            return
        
        self._categories = categories
        self._ids = [c.id for c in categories]
        docs = [self._category_to_doc(c) for c in categories]
        
        # Fit TF-IDF (always, used for hybrid cascade)
        if self.mode in [RETRIEVAL_MODE.TFIDF, RETRIEVAL_MODE.HYBRID]:
            self._tfidf_X = self.vectorizer.fit_transform(docs)
            self._tfidf_fitted = True
        
        # Precompute BERT embeddings
        if self.mode in [RETRIEVAL_MODE.BERT, RETRIEVAL_MODE.HYBRID] and self.bert_encoder:
            logger.info(f"Computing BERT embeddings for {len(docs)} categories...")
            self.bert_embeddings = self.bert_encoder.encode(docs)
            logger.info("✓ BERT embeddings computed")
        
        self._fitted = True

    def top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return top K most similar categories with similarity scores."""
        if not self._is_ready():
            return []
        
        if self.mode == RETRIEVAL_MODE.TFIDF:
            return self._tfidf_top_k(text, k)
        elif self.mode == RETRIEVAL_MODE.BERT:
            return self._bert_top_k(text, k)
        elif self.mode == RETRIEVAL_MODE.HYBRID:
            return self._hybrid_cascade(text, k)
        else:
            logger.warning(f"Unknown mode {self.mode}, using TF-IDF")
            return self._tfidf_top_k(text, k)
    
    def get_bert_similarity(self, text: str, category_id: str) -> Optional[float]:
        """Get BERT similarity for a specific category."""
        if not self.bert_encoder or self.bert_embeddings is None:
            return None
        
        try:
            idx = self._ids.index(category_id)
            text_emb = self.bert_encoder.encode([text])
            cat_emb = self.bert_embeddings[idx:idx+1]
            return self.bert_encoder.cosine_similarity(text_emb, cat_emb)
        except (ValueError, IndexError):
            return None

    def _tfidf_top_k(self, text: str, k: int) -> List[Tuple[str, float]]:
        """TF-IDF based retrieval."""
        if not self._tfidf_fitted or self._tfidf_X is None:
            return []
        
        v = self.vectorizer.transform([text])
        sims = cosine_similarity(v, self._tfidf_X).ravel()
        top_indices = sims.argsort()[::-1][:k]
        
        return [(self._ids[i], float(sims[i])) for i in top_indices if sims[i] > 0.0]
    
    def _bert_top_k(self, text: str, k: int) -> List[Tuple[str, float]]:
        """BERT-based retrieval."""
        if self.bert_embeddings is None or not self.bert_encoder:
            return []
        
        text_emb = self.bert_encoder.encode([text])
        sims = cosine_similarity(text_emb, self.bert_embeddings).ravel()
        top_indices = sims.argsort()[::-1][:k]
        
        return [(self._ids[i], float(sims[i])) for i in top_indices]
    
    def _hybrid_cascade(self, text: str, k: int) -> List[Tuple[str, float]]:
        """
        Two-stage hybrid retrieval:
        1. TF-IDF prefilters to top N candidates (fast)
        2. BERT reranks candidates to top K (accurate)
        """
        # Stage 1: TF-IDF prefilter
        prefilter_k = min(self.tfidf_prefilter_k, len(self._ids))
        tfidf_candidates = self._tfidf_top_k(text, k=prefilter_k)
        
        if not tfidf_candidates or not self.bert_encoder or self.bert_embeddings is None:
            return tfidf_candidates[:k]
        
        # Stage 2: BERT rerank
        candidate_ids = [cid for cid, _ in tfidf_candidates]
        candidate_indices = [self._ids.index(cid) for cid in candidate_ids]
        
        # Encode query once
        text_emb = self.bert_encoder.encode([text])
        
        # Compute BERT similarity only for candidates
        candidate_embeddings = self.bert_embeddings[candidate_indices]
        bert_sims = cosine_similarity(text_emb, candidate_embeddings).ravel()
        
        # Combine and rerank
        bert_ranked = sorted(
            zip(candidate_ids, bert_sims),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [(cid, float(sim)) for cid, sim in bert_ranked]

    def _category_to_doc(self, category: Category) -> str:
        """Convert category to document string."""
        parts = [
            category.name or "",
            category.description or "",
            " ".join(category.keywords or [])
        ]
        return " ".join(parts) or " "

    def _is_ready(self) -> bool:
        """Check if indexer is ready for queries."""
        return self._fitted and self._ids

    def _reset(self) -> None:
        """Reset indexer state."""
        self._fitted = False
        self._tfidf_fitted = False
        self._ids = []
        self._categories = []
        self._tfidf_X = None
        self.bert_embeddings = None

# ------------------
# State management
# ------------------
@dataclass
class ProcessingState:
    """Helper class for managing state transitions."""
    current_index: int = 0
    current_pass: int = 1
    loop_counter: int = 0
    processing_complete: bool = False
    consolidation_complete: bool = False
    
    def should_stop(self, max_watchdog: int) -> bool:
        return self.loop_counter > max_watchdog
    
    def increment_loop(self) -> None:
        self.loop_counter += 1
    
    def advance_index(self, batch_size: int = 1) -> None:
        self.current_index += batch_size
    
    def start_next_pass(self) -> None:
        self.current_pass += 1
        self.current_index = 0
        self.processing_complete = False

def extract_processing_state(state: Dict[str, Any]) -> ProcessingState:
    """Extract ProcessingState from state dict."""
    return ProcessingState(
        current_index=int(state.get('current_index', 0)),
        current_pass=int(state.get('current_pass', 1)),
        loop_counter=int(state.get('_loop_counter', 0)),
        processing_complete=bool(state.get('processing_complete', False)),
        consolidation_complete=bool(state.get('consolidation_complete', False))
    )

def update_state_from_processing(state: Dict[str, Any], proc_state: ProcessingState) -> None:
    """Update state dict from ProcessingState."""
    state.update({
        'current_index': proc_state.current_index,
        'current_pass': proc_state.current_pass,
        '_loop_counter': proc_state.loop_counter,
        'processing_complete': proc_state.processing_complete,
        'consolidation_complete': proc_state.consolidation_complete
    })

# ------------------
# Assignment management
# ------------------
class AssignmentManager:
    """Manages text-to-category assignments idempotently."""
    
    def __init__(self):
        self.assignments_map: Dict[str, Dict[str, Any]] = {}
    
    def update_assignment(self, assignment: CategoryAssignment) -> None:
        """Update assignment, replacing if confidence is higher."""
        text_key = encode_text_to_id(assignment.text)
        new_dict = assignment.model_dump()
        
        existing = self.assignments_map.get(text_key)
        if self._should_replace(existing, new_dict):
            self.assignments_map[text_key] = new_dict
    
    def get_assignments(self) -> List[Dict[str, Any]]:
        """Get all assignments as list."""
        return list(self.assignments_map.values())
    
    def update_category_counts(self, categories: List[Dict[str, Any]]) -> None:
        """Update text_count for all categories based on assignments."""
        assignments = self.get_assignments()
        for cat in categories:
            cat['text_count'] = sum(
                1 for a in assignments if a.get('category_id') == cat['id']
            )
    
    @staticmethod
    def _should_replace(existing: Optional[Dict], new: Dict) -> bool:
        """Determine if new assignment should replace existing."""
        if existing is None:
            return True
        return new.get('confidence', 0.0) > existing.get('confidence', 0.0)

# ------------------
# Rate limiting
# ------------------
class LLMRateLimiter:
    """Rate limiter for concurrent LLM calls."""
    
    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def __aenter__(self):
        await self.semaphore.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()

_rate_limiter = LLMRateLimiter(DEFAULT_MAX_CONCURRENT_LLM_CALLS)

def _generate_semantic_category_name(text: str, existing_categories: List[Category]) -> str:
    """Generate a semantic category name based on text content."""
    # Extract key concepts without hardcoding specific domains
    words = text.lower().split()
    
    # Look for activity patterns (verbs/actions)
    activity_indicators = ['cultivo', 'desenvolvimento', 'fabricação', 'consultoria', 'suporte', 'design', 'análise']
    domain_indicators = ['tecnologia', 'agricultura', 'industrial', 'comercial', 'serviços', 'educação']
    
    activity = next((word for word in words if any(ind in word for ind in activity_indicators)), None)
    domain = next((word for word in words if any(ind in word for ind in domain_indicators)), None)
    
    # Create meaningful name
    if activity and domain:
        return f"{activity.title()} - {domain.title()}"
    elif activity:
        return f"{activity.title()}"
    elif domain:
        return f"{domain.title()}"
    else:
        # Generic naming based on first significant words
        significant_words = [w for w in words if len(w) > 3][:3]
        if significant_words:
            return " ".join(word.title() for word in significant_words)
        else:
            return f"Category_{len(existing_categories) + 1}"



def _extract_semantic_keywords(text: str) -> List[str]:
    """Extract semantic keywords without domain-specific hardcoding."""
    words = text.lower().split()
    
    # Filter by word characteristics, not specific domains
    keywords = []
    for word in words:
        # Keep words that are:
        # - Substantial length (likely meaningful)
        # - Potential nouns or verbs
        if (len(word) >= 3 and not word.isdigit()):
            keywords.append(word)
    
    return keywords[:5]  # Limit to top 5 most relevant

def _create_fallback_analysis(text: str, candidates: List[Tuple[str, float]], categories: List[Category]) -> Any:
    """Create a fallback analysis when LLM calls are disabled."""
    # Adaptive similarity threshold based on existing categories
    if not categories:
        # No categories exist - create first one
        similarity_threshold = 0.3  # Low threshold for first category
    else:
        # Calculate adaptive threshold based on category distribution
        category_sizes = [cat.text_count for cat in categories]
        avg_size = sum(category_sizes) / len(category_sizes) if category_sizes else 1
        
        # Higher threshold if categories are well-populated (encourage consolidation)
        # Lower threshold if categories are small (allow more granularity)
        if avg_size >= 3:
            similarity_threshold = 0.4  # Encourage grouping
        elif avg_size >= 2:
            similarity_threshold = 0.5  # Moderate grouping
        else:
            similarity_threshold = 0.6  # Be selective
    
    # Use best match if above adaptive threshold
    if candidates and candidates[0][1] > similarity_threshold:
        best_id = candidates[0][0]
        return type('Analysis', (), {
            'candidate_categories': [best_id],
            'confidence_scores': {best_id: candidates[0][1]},
            'best_category_id': best_id,
            'should_create_new': False,
            'new_category': None,
            'reasoning': f"Fallback: similarity {candidates[0][1]:.2f} > threshold {similarity_threshold:.2f}"
        })()
    else:
        # Create new category with semantic naming
        name = _generate_semantic_category_name(text, categories)
        new_cat = {
            'id': deterministic_cat_id(name),
            'name': name,
            'description': text[:100] + "..." if len(text) > 100 else text,
            'keywords': _extract_semantic_keywords(text)
        }
        return type('Analysis', (), {
            'candidate_categories': [],
            'confidence_scores': {},
            'best_category_id': None,
            'should_create_new': True,
            'new_category': new_cat,
            'reasoning': f"Fallback: new category (similarity {candidates[0][1]:.2f} <= {similarity_threshold:.2f})" if candidates else "Fallback: no candidates"
        })()

# ------------------
# Parallel analysis with BERT
# ------------------
async def _analyze_single_text_parallel(
    text: str,
    categories: List[Category],
    indexer: CategoryIndexer,
    state: Dict[str, Any],
    text_num: int,
    total_texts: int
) -> Tuple[CategoryAssignment, Optional[Category]]:
    """Analyze a single text with BERT-enhanced confidence."""
    current_pass = int(state.get('current_pass', 1))
    max_passes = int(state.get('max_passes', 2))
    
    # Get config from state
    config = state.get('config', DEFAULT_CONFIG)
    
    # Use adaptive top-K selection
    adaptive_selector = AdaptiveTopKSelector(
        base_k=int(state.get('prefilter_k', DEFAULT_PREFILTER_K))
    )
    
    # Estimate text complexity for adaptive K
    text_complexity = adaptive_selector.estimate_text_complexity(text)
    
    top_k = adaptive_selector.compute_k(
        current_pass=current_pass,
        max_passes=max_passes,
        num_categories=len(categories),
        text_complexity=text_complexity,
        is_creating_new=False  # Will be determined by LLM later
    )
    
    # Get candidates (using hybrid/BERT if available)
    candidates = indexer.top_k(text, k=top_k)
    candidate_ids = [cid for cid, _ in candidates]
    
    # Build prompt
    prompt = _build_analysis_prompt(text, categories, candidates, current_pass, max_passes, top_k)
    
    # Call LLM or use fallback
    try:
        if multi_pass_analyzer is None:
            # Fallback mode: use simple similarity-based assignment
            logger.info("  [%d/%d] Using fallback (no LLM): %.50s...", text_num, total_texts, text)
            analysis = _create_fallback_analysis(text, candidates, categories)
        else:
            async with _rate_limiter:
                logger.info("  [%d/%d] Analyzing: %.50s...", text_num, total_texts, text)
                logger.debug("  Calling LLM with model: %s:%s", LLM_PROVIDER_NAME, LLM_MODEL_NAME)
                result = await asyncio.wait_for(
                    multi_pass_analyzer.run(prompt),
                    timeout=DEFAULT_LLM_TIMEOUT_SECONDS
                )
                analysis = result.output
            
            if analysis.confidence_scores:
                logger.debug("  LLM confidence scores: %s", 
                           {k: f"{v:.2f}" for k, v in analysis.confidence_scores.items()})
            else:
                logger.debug("  LLM did not provide confidence scores, using fallback scoring")
            
            logger.info("  [%d/%d] ✓ Complete", text_num, total_texts)
        
        # Process with BERT-enhanced confidence (both LLM and fallback paths)
        return _process_analysis_result_enhanced(
            analysis, text, categories, candidate_ids, candidates, 
            current_pass, max_passes, indexer, config
        )
    except asyncio.TimeoutError:
        logger.error("  [%d/%d] ✗ Timeout", text_num, total_texts)
        assignment = _create_fallback_assignment(text, categories, candidate_ids)
        return (assignment, None)
    except Exception as e:
        logger.error("  [%d/%d] ✗ Error: %s", text_num, total_texts, str(e)[:50])
        assignment = _create_fallback_assignment(text, categories, candidate_ids)
        return (assignment, None)

def _build_analysis_prompt(
    text: str,
    categories: List[Category],
    candidates: List[Tuple[str, float]],
    current_pass: int,
    max_passes: int,
    top_k: int
) -> str:
    """Build analysis prompt."""
    if not categories:
        cat_desc = "No categories exist yet - you should create the first one."
    else:
        cat_desc = "\n".join([
            f"- {c.id}: '{c.name}' - {c.description or 'No description'}\n  Keywords: {', '.join(c.keywords[:8]) if c.keywords else 'none'}\n  Texts assigned: {c.text_count}"
            for c in categories
        ])
    
    candidate_info = ""
    if candidates:
        candidate_info = f"\nTop candidates by semantic similarity:\n" + "\n".join([
            f"  - {cid}: similarity={sim:.3f}"
            for cid, sim in candidates
        ])
    
    return f"""Pass {current_pass}/{max_passes}

TEXT TO CATEGORIZE:
"{text}"

EXISTING CATEGORIES:
{cat_desc}
{candidate_info}

SEMANTIC ANALYSIS GUIDELINES:
- Focus on conceptual similarity over exact word matching
- Group related activities (e.g., cultivation, processing, and production of same domain)
- Consider functional relationships (e.g., different stages of same process)
- Look for domain coherence (activities that belong to same industry/field)
- Prefer consolidation when semantic overlap exists
- Services (serviços, suporte, manutenção) should be grouped

TASK:
1. Evaluate how well this text fits each candidate category
2. Provide SPECIFIC confidence scores (0.0-1.0) - NOT all 0.5!
3. Consider semantic similarity (crops with crops, tech with tech, etc.)
4. Either select best_category_id OR create new category if all scores < 0.6"""

def _process_analysis_result_enhanced(
    analysis: MultiPassAnalysis,
    text: str,
    categories: List[Category],
    candidate_ids: List[str],
    candidates: List[Tuple[str, float]],
    current_pass: int,
    max_passes: int,
    indexer: CategoryIndexer,
    config: ClusterizerConfig
) -> Tuple[CategoryAssignment, Optional[Category]]:
    """Process LLM analysis with BERT-enhanced confidence."""
    if analysis.should_create_new:
        return _handle_new_category_enhanced(
            analysis, text, categories, candidates, current_pass, max_passes, indexer, config
        )
    else:
        assignment = _handle_existing_category_enhanced(
            analysis, text, candidate_ids, categories, candidates, 
            current_pass, max_passes, indexer, config
        )
        return (assignment, None)

def _handle_new_category_enhanced(
    analysis: MultiPassAnalysis,
    text: str,
    categories: List[Category],
    candidates: List[Tuple[str, float]],
    current_pass: int,
    max_passes: int,
    indexer: CategoryIndexer,
    config: ClusterizerConfig
) -> Tuple[CategoryAssignment, Category]:
    """Handle creation of new category with BERT confidence."""
    if analysis.new_category:
        new_cat = build_category_from_raw(
            analysis.new_category,
            fallback_name=analysis.new_category.get('name', 'newcat'),
            fallback_text=text
        )
    else:
        name = f"new_{encode_text_to_id(text)[:6]}"
        new_cat = build_category_from_raw(
            {'name': name, 'description': 'Auto-created', 'keywords': text.split()[:6]},
            fallback_name=name,
            fallback_text=text
        )
    
    # Ensure no ID collision
    existing_ids = {c.id for c in categories}
    if new_cat.id in existing_ids:
        new_cat.id = deterministic_cat_id(new_cat.name + encode_text_to_id(text)[:4])
    
    new_cat.text_count = 1
    
    # Calculate enhanced confidence
    llm_confidence = extract_confidence(analysis.confidence_scores, new_cat.id)
    if llm_confidence == 0.0:
        llm_confidence = 0.75
    
    # BERT similarity not available for new categories
    confidence = calculate_confidence(
        text=text,
        category=new_cat,
        llm_confidence=llm_confidence,
        tfidf_similarity=0.0,
        current_pass=current_pass,
        max_passes=max_passes,
        total_categories=len(categories) + 1,
        bert_similarity=None,  # New category, no prior embedding
        weights=config.confidence_weights
    )
    
    # Apply new category bonus from config
    confidence = min(1.0, confidence * config.new_category_bonus)
    
    assignment = CategoryAssignment(
        text=text,
        category_id=new_cat.id,
        confidence=confidence,
        reasoning=f"Created new category (llm={llm_confidence:.2f}, enhanced={confidence:.2f})"
    )
    
    return (assignment, new_cat)

def _handle_existing_category_enhanced(
    analysis: MultiPassAnalysis,
    text: str,
    candidate_ids: List[str],
    categories: List[Category],
    candidates: List[Tuple[str, float]],
    current_pass: int,
    max_passes: int,
    indexer: CategoryIndexer,
    config: ClusterizerConfig
) -> CategoryAssignment:
    """Handle assignment to existing category with BERT confidence."""
    chosen_id = analysis.best_category_id or (candidate_ids[0] if candidate_ids else None)
    
    if not chosen_id and categories:
        chosen_id = categories[0].id
    
    # Normalize the category ID for consistency
    if chosen_id:
        chosen_id = normalize_category_id(chosen_id)
    
    chosen_category = next((c for c in categories if c.id == chosen_id), None)
    
    if not chosen_category:
        return CategoryAssignment(
            text=text,
            category_id=chosen_id or "unknown",
            confidence=0.3,
            reasoning="Category not found"
        )
    
    # Get LLM confidence
    llm_confidence = extract_confidence(analysis.confidence_scores, chosen_id)
    if llm_confidence == 0.0:
        llm_confidence = 0.5
    
    # Get TF-IDF similarity
    tfidf_similarity = _get_similarity_for_category(chosen_id, candidates)
    
    # Get BERT similarity (NEW!)
    bert_similarity = indexer.get_bert_similarity(text, chosen_id)
    
    # Calculate enhanced confidence with BERT
    confidence = calculate_confidence(
        text=text,
        category=chosen_category,
        llm_confidence=llm_confidence,
        tfidf_similarity=tfidf_similarity,
        current_pass=current_pass,
        max_passes=max_passes,
        total_categories=len(categories),
        bert_similarity=bert_similarity,
        weights=config.confidence_weights,
        debug=True
    )
    
    # Build reasoning with BERT info
    reasoning_parts = [f"llm={llm_confidence:.2f}", f"tfidf={tfidf_similarity:.2f}"]
    if bert_similarity is not None:
        reasoning_parts.append(f"bert={bert_similarity:.2f}")
    reasoning_parts.append(f"final={confidence:.2f}")
    
    return CategoryAssignment(
        text=text,
        category_id=chosen_id,
        confidence=confidence,
        reasoning=f"Assigned ({', '.join(reasoning_parts)})"
    )

def _get_similarity_for_category(category_id: str, candidates: List[Tuple[str, float]]) -> float:
    """Extract similarity score for specific category from candidates."""
    for cid, similarity in candidates:
        if cid == category_id:
            return similarity
    return 0.0

def _create_fallback_assignment(
    text: str,
    categories: List[Category],
    candidate_ids: List[str]
) -> CategoryAssignment:
    """Create fallback assignment when LLM fails."""
    if candidate_ids:
        return CategoryAssignment(
            text=text,
            category_id=candidate_ids[0],
            confidence=0.6,
            reasoning="Fallback: top candidate"
        )
    elif categories:
        return CategoryAssignment(
            text=text,
            category_id=categories[0].id,
            confidence=0.4,
            reasoning="Fallback: first category"
        )
    else:
        return CategoryAssignment(
            text=text,
            category_id="unknown",
            confidence=0.3,
            reasoning="Fallback: no categories"
        )

# ------------------
# Graph nodes
# ------------------
async def load_next_batch(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load next batch of texts for processing."""
    logger.debug("NODE load_next_batch called")
    new_state = dict(state)
    
    proc_state = extract_processing_state(state)
    
    # No watchdog needed - rely on natural termination (finite passes, batches)
    
    texts = state.get('texts', []) or []
    batch_size = int(state.get('batch_size', DEFAULT_BATCH_SIZE))
    max_passes = int(state.get('max_passes', 2))
    
    if not texts:
        return _finalize_processing(new_state, proc_state)
    
    if proc_state.current_index >= len(texts):
        if proc_state.current_pass < max_passes:
            proc_state.start_next_pass()
            proc_state.increment_loop()  # Increment when starting new pass
            batch = texts[:batch_size]
            new_state['current_batch'] = batch
            new_state['decision'] = 'analyze'
            logger.info("Step %d: Starting pass %d", proc_state.loop_counter, proc_state.current_pass)
        else:
            return _finalize_processing(new_state, proc_state)
    else:
        proc_state.increment_loop()  # Increment when processing batch
        end_idx = min(proc_state.current_index + batch_size, len(texts))
        batch = texts[proc_state.current_index:end_idx]
        new_state['current_batch'] = batch
        new_state['decision'] = 'analyze'
        
        # Better logging
        logger.info("Step %d: | Pass %d | Text %d/%d | Categories: %d", 
                   proc_state.loop_counter, proc_state.current_pass, 
                   proc_state.current_index + 1, len(texts), 
                   len(state.get('categories', [])))
    
    update_state_from_processing(new_state, proc_state)
    return new_state

def _finalize_processing(state: Dict[str, Any], proc_state: ProcessingState) -> Dict[str, Any]:
    """Helper to finalize processing state."""
    proc_state.processing_complete = True
    update_state_from_processing(state, proc_state)
    state['current_batch'] = []
    state['decision'] = 'consolidate'
    return state

async def analyze_batch_parallel(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze batch with BERT-powered indexer."""
    logger.info("NODE analyze_batch_parallel called with %d texts", len(state.get('current_batch', [])))
    new_state = dict(state)
    
    batch = state.get('current_batch', [])
    if not batch:
        logger.warning("analyze_batch_parallel called with empty batch")
        new_state['decision'] = 'analyzed'
        return new_state
    
    categories_raw = state.get('categories', [])
    categories = [Category(**c) if not isinstance(c, Category) else c for c in categories_raw]
    
    # Get config from state
    config = state.get('config', DEFAULT_CONFIG)
    if not isinstance(config, ClusterConfig):
        # Backward compatibility: if config is not in state, build from retrieval_mode
        retrieval_mode = state.get('retrieval_mode', RETRIEVAL_MODE.HYBRID)
        if retrieval_mode == RETRIEVAL_MODE.HYBRID:
            config = CONFIG_BALANCED_HYBRID
        elif retrieval_mode == RETRIEVAL_MODE.BERT:
            config = CONFIG_SEMANTIC_BERT
        else:
            config = CONFIG_BALANCED_TFIDF
    
    # Setup indexer with configured mode
    indexer = CategoryIndexer(mode=config.retrieval_mode)
    indexer.fit(categories)
    
    initial_category_ids = {c.id for c in categories}
    
    # Process batch in parallel
    logger.info("Starting parallel processing of %d texts...", len(batch))
    tasks = [
        _analyze_single_text_parallel(text, categories, indexer, state, i+1, len(batch))
        for i, text in enumerate(batch)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Parallel processing complete")
    
    # Collect results
    assignment_mgr = AssignmentManager()
    assignment_mgr.assignments_map = dict(state.get('assignments_map') or {})
    new_categories_created = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Task %d failed: %s", i+1, result)
            text = batch[i]
            fallback = _create_fallback_assignment(text, categories, [])
            assignment_mgr.update_assignment(fallback)
        else:
            assignment, maybe_new_cat = result
            assignment_mgr.update_assignment(assignment)
            
            if maybe_new_cat and maybe_new_cat.id not in initial_category_ids:
                # Check for duplicate names (case-insensitive)
                existing_names = {c.name.lower() for c in categories + new_categories_created}
                if maybe_new_cat.name.lower() in existing_names:
                    # Find the existing category with same name and reuse it
                    existing_cat = next(
                        (c for c in categories + new_categories_created 
                         if c.name.lower() == maybe_new_cat.name.lower()),
                        None
                    )
                    if existing_cat:
                        logger.info(f"Reusing existing category '{existing_cat.name}' instead of creating duplicate")
                        # Create new assignment with existing category ID
                        new_assignment = CategoryAssignment(
                            text=assignment.text,
                            category_id=existing_cat.id,
                            confidence=assignment.confidence,
                            reasoning=assignment.reasoning
                        )
                        assignment_mgr.update_assignment(new_assignment)
                        continue
                
                if maybe_new_cat.id not in [c.id for c in new_categories_created]:
                    new_categories_created.append(maybe_new_cat)
                    initial_category_ids.add(maybe_new_cat.id)
                    logger.info("Created new category: %s", maybe_new_cat.name)
    
    # Combine categories
    all_categories = categories + new_categories_created
    categories_dicts = [c.model_dump() for c in all_categories]
    assignment_mgr.update_category_counts(categories_dicts)
    
    proc_state = extract_processing_state(state)
    proc_state.advance_index(len(batch))
    
    new_state.update({
        'decision': 'analyzed',
        'categories': categories_dicts,
        'assignments': assignment_mgr.get_assignments(),
        'assignments_map': assignment_mgr.assignments_map,
    })
    update_state_from_processing(new_state, proc_state)
    
    logger.info("Batch analysis complete. Categories: %d, Assignments: %d", 
                len(categories_dicts), len(assignment_mgr.get_assignments()))
    
    return new_state

# ------------------
# Chunked Categorization for Large Texts
# ------------------
@dataclass
class TextChunk:
    """Represents a chunk of a larger text."""
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    parent_text_id: str
    chunk_index: int
    total_chunks: int

@dataclass
class ChunkAssignment:
    """Assignment of a chunk to a category."""
    chunk: TextChunk
    category_id: str
    confidence: float
    reasoning: str

@dataclass
class AggregatedAssignment:
    """Final assignment after aggregating chunk results."""
    text: str
    primary_category_id: str
    confidence: float
    secondary_categories: List[Tuple[str, float]]  # For multi-topic texts
    is_multi_topic: bool
    reasoning: str

class SemanticChunker:
    """
    Smart text chunking that preserves semantic boundaries.
    
    Strategies:
    - Sentence-boundary aware (don't split mid-sentence)
    - Paragraph-aware (prefer splitting at paragraph breaks)
    - Overlap for context preservation
    """
    
    DEFAULT_CHUNK_SIZE = 500  # tokens (roughly 300-400 words)
    DEFAULT_OVERLAP = 100     # tokens overlap for context
    MIN_CHUNK_SIZE = 100      # Don't create tiny chunks
    MAX_CHUNKS_PER_TEXT = 20  # Safety limit
    
    def __init__(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def should_chunk(self, text: str, threshold: int = 800) -> bool:
        """Determine if text should be chunked based on token count."""
        tokens = text.split()
        return len(tokens) >= threshold
    
    def chunk_text(self, text: str, text_id: str) -> List[TextChunk]:
        """
        Split text into semantic chunks.
        
        Returns:
            List of TextChunk objects
        """
        tokens = text.split()  # Simple tokenization
        
        # If text is small, return as single chunk
        if len(tokens) <= self.chunk_size:
            return [
                TextChunk(
                    chunk_id=f"{text_id}_chunk_0",
                    text=text,
                    start_pos=0,
                    end_pos=len(text),
                    parent_text_id=text_id,
                    chunk_index=0,
                    total_chunks=1
                )
            ]
        
        # Split into chunks with overlap
        chunks = []
        sentences = self._split_sentences(text)
        
        current_chunk_tokens = []
        current_chunk_sentences = []
        start_pos = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = sentence.split()
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk_tokens) + len(sentence_tokens) > self.chunk_size and current_chunk_sentences:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    TextChunk(
                        chunk_id=f"{text_id}_chunk_{chunk_index}",
                        text=chunk_text,
                        start_pos=start_pos,
                        end_pos=start_pos + len(chunk_text),
                        parent_text_id=text_id,
                        chunk_index=chunk_index,
                        total_chunks=-1  # Will update later
                    )
                )
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, 
                    current_chunk_tokens
                )
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = []
                for s in overlap_sentences:
                    current_chunk_tokens.extend(s.split())
                start_pos = start_pos + len(chunk_text) - sum(len(s) for s in overlap_sentences)
                chunk_index += 1
            
            current_chunk_sentences.append(sentence)
            current_chunk_tokens.extend(sentence_tokens)
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                TextChunk(
                    chunk_id=f"{text_id}_chunk_{chunk_index}",
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    parent_text_id=text_id,
                    chunk_index=chunk_index,
                    total_chunks=-1
                )
            )
        
        # Update total_chunks for all
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        # Safety: limit number of chunks
        if len(chunks) > self.MAX_CHUNKS_PER_TEXT:
            logger.warning(f"Text {text_id} produced {len(chunks)} chunks, limiting to {self.MAX_CHUNKS_PER_TEXT}")
            chunks = chunks[:self.MAX_CHUNKS_PER_TEXT]
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple heuristic)."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(
        self, 
        sentences: List[str], 
        tokens: List[str]
    ) -> List[str]:
        """Get sentences for overlap window."""
        overlap_tokens = []
        overlap_sentences = []
        
        # Take sentences from end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = sentence.split()
            if len(overlap_tokens) + len(sentence_tokens) <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens = sentence_tokens + overlap_tokens
            else:
                break
        
        return overlap_sentences

class ChunkAggregator:
    """
    Aggregate chunk categorization results into final assignment.
    
    Strategies:
    - Majority voting for single-topic texts
    - Multi-label for multi-topic texts
    - Confidence-weighted aggregation
    """
    
    def __init__(self, multi_topic_threshold: float = 0.3):
        """
        Args:
            multi_topic_threshold: If secondary category has >30% of votes, 
                                   consider it multi-topic
        """
        self.multi_topic_threshold = multi_topic_threshold
    
    def aggregate(
        self, 
        text: str,
        chunk_assignments: List[ChunkAssignment]
    ) -> AggregatedAssignment:
        """
        Aggregate chunk assignments into final categorization.
        
        Returns:
            AggregatedAssignment with primary category and potential secondary categories
        """
        if not chunk_assignments:
            return AggregatedAssignment(
                text=text,
                primary_category_id="unknown",
                confidence=0.0,
                secondary_categories=[],
                is_multi_topic=False,
                reasoning="No chunk assignments"
            )
        
        # Collect category votes with confidence weighting
        category_votes = {}
        category_confidences = {}
        
        for assignment in chunk_assignments:
            cat_id = assignment.category_id
            conf = assignment.confidence
            
            if cat_id not in category_votes:
                category_votes[cat_id] = 0
                category_confidences[cat_id] = []
            
            # Weight vote by confidence
            category_votes[cat_id] += conf
            category_confidences[cat_id].append(conf)
        
        # Sort categories by vote weight
        sorted_categories = sorted(
            category_votes.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        if not sorted_categories:
            return AggregatedAssignment(
                text=text,
                primary_category_id="unknown",
                confidence=0.0,
                secondary_categories=[],
                is_multi_topic=False,
                reasoning="No valid categories"
            )
        
        # Primary category (highest votes)
        primary_cat_id, primary_votes = sorted_categories[0]
        primary_confidence = sum(category_confidences[primary_cat_id]) / len(category_confidences[primary_cat_id])
        
        # Check for multi-topic text
        total_votes = sum(v for _, v in sorted_categories)
        secondary_categories = []
        is_multi_topic = False
        
        for cat_id, votes in sorted_categories[1:]:
            vote_ratio = votes / total_votes
            if vote_ratio >= self.multi_topic_threshold:
                is_multi_topic = True
                avg_conf = sum(category_confidences[cat_id]) / len(category_confidences[cat_id])
                secondary_categories.append((cat_id, avg_conf))
        
        # Build reasoning
        from collections import Counter
        chunk_distribution = Counter(a.category_id for a in chunk_assignments)
        reasoning_parts = [
            f"Aggregated from {len(chunk_assignments)} chunks",
            f"Primary: {chunk_distribution[primary_cat_id]}/{len(chunk_assignments)} chunks"
        ]
        
        if is_multi_topic:
            reasoning_parts.append(
                f"Multi-topic: {len(secondary_categories)} secondary categories"
            )
        
        reasoning = " | ".join(reasoning_parts)
        
        return AggregatedAssignment(
            text=text,
            primary_category_id=primary_cat_id,
            confidence=primary_confidence,
            secondary_categories=secondary_categories,
            is_multi_topic=is_multi_topic,
            reasoning=reasoning
        )

# ------------------
# Consolidation with Multi-Signal Analysis
# ------------------
class MultiSignalConsolidator:
    """
    Multi-signal category consolidation with adaptive thresholds and AI-driven decisions.
    
    Improvements over basic TF-IDF:
    - BERT semantic similarity (catches "Finance Markets" vs "Finance Economics")
    - Keyword overlap analysis  
    - Name similarity detection
    - Adaptive thresholds based on data distribution
    - AI agent for nuanced merge decisions
    - Statistical outlier detection
    """
    
    def __init__(self, bert_encoder: Optional[BERTEncoder] = None):
        self.bert_encoder = bert_encoder
        self._similarity_cache = {}  # Cache for expensive computations
    
    def find_merge_candidates(
        self, 
        categories: List[Category],
        aggressive: bool = True
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """
        Find category pairs for merging using adaptive thresholds and AI decisions.
        
        Args:
            categories: List of categories to analyze
            aggressive: If True, use more lenient criteria (for early passes)
        
        Returns:
            List of (cat1_id, cat2_id, overall_score, signal_breakdown)
        """
        if len(categories) <= 1:
            return []
        
        # Step 1: Compute all pairwise similarities
        all_scores = self._compute_all_similarities(categories)
        
        # Step 2: Determine adaptive thresholds from data distribution
        thresholds = self._compute_adaptive_thresholds(all_scores, aggressive)
        
        # Step 3: Apply multi-criteria filtering
        candidates = self._apply_multi_criteria_filtering(all_scores, thresholds, aggressive)
        
        # Step 4: AI agent validation for borderline cases
        validated_candidates = self._ai_validate_candidates(candidates, categories)
        
        # Log analysis for debugging
        self._log_consolidation_analysis(all_scores, thresholds, candidates, validated_candidates)
        
        return validated_candidates
    
    def _compute_all_similarities(self, categories: List[Category]) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """Compute all pairwise similarities efficiently."""
        all_scores = []
        
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                
                # Use cache for expensive operations
                cache_key = (cat1.id, cat2.id)
                if cache_key in self._similarity_cache:
                    signals = self._similarity_cache[cache_key]
                else:
                    signals = self._analyze_similarity(cat1, cat2)
                    self._similarity_cache[cache_key] = signals
                
                # Compute overall score
                overall = self._compute_merge_score(signals, aggressive=True)
                
                all_scores.append((cat1.name, cat2.name, overall, signals))
        
        return all_scores
    
    def _compute_adaptive_thresholds(self, all_scores: List[Tuple], aggressive: bool) -> Dict[str, float]:
        """
        Compute adaptive thresholds based on data distribution.
        
        Uses statistical methods:
        - Percentile-based thresholds
        - Outlier detection (IQR method)
        - Signal strength distribution analysis
        """
        import numpy as np
        
        if not all_scores:
            return self._get_fallback_thresholds(aggressive)
        
        # Extract signal arrays
        bert_scores = [s[3].get('bert_similarity', 0) for s in all_scores if s[3].get('bert_similarity', 0) > 0]
        tfidf_scores = [s[3].get('tfidf_similarity', 0) for s in all_scores]
        overall_scores = [s[2] for s in all_scores]
        name_scores = [s[3].get('name_similarity', 0) for s in all_scores]
        
        thresholds = {}
        
        if bert_scores:
            # BERT: Use 75th percentile for conservative, 60th for aggressive
            bert_array = np.array(bert_scores)
            percentile = 60 if aggressive else 75
            thresholds['bert_auto_approve'] = float(np.percentile(bert_array, percentile))
            thresholds['bert_auto_approve'] = max(0.55, min(0.80, thresholds['bert_auto_approve']))  # Bounds
        else:
            thresholds['bert_auto_approve'] = 0.65
        
        # Overall score: Use IQR method to find "unusually high" similarities
        if overall_scores:
            overall_array = np.array(overall_scores)
            q75 = np.percentile(overall_array, 75)
            q50 = np.percentile(overall_array, 50)
            iqr = q75 - q50
            
            # Adaptive threshold: median + 0.5*IQR (aggressive) or median + 1*IQR (conservative)
            multiplier = 0.3 if aggressive else 0.7
            thresholds['combined_threshold'] = float(q50 + multiplier * iqr)
            thresholds['combined_threshold'] = max(0.25, min(0.70, thresholds['combined_threshold']))  # Bounds
        else:
            thresholds['combined_threshold'] = 0.40 if aggressive else 0.50
        
        # Name similarity: High name similarity should almost always merge
        if name_scores:
            name_array = np.array([s for s in name_scores if s > 0])
            if len(name_array) > 0:
                thresholds['name_auto_approve'] = float(np.percentile(name_array, 80))
                thresholds['name_auto_approve'] = max(0.60, thresholds['name_auto_approve'])
            else:
                thresholds['name_auto_approve'] = 0.75
        else:
            thresholds['name_auto_approve'] = 0.75
        
        return thresholds
    
    def _get_fallback_thresholds(self, aggressive: bool) -> Dict[str, float]:
        """Fallback thresholds when no data available."""
        return {
            'bert_auto_approve': 0.60 if aggressive else 0.70,
            'combined_threshold': 0.35 if aggressive else 0.45,
            'name_auto_approve': 0.75
        }
    
    def _apply_multi_criteria_filtering(
        self, 
        all_scores: List[Tuple], 
        thresholds: Dict[str, float], 
        aggressive: bool
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """Apply multi-criteria decision analysis for merge candidates."""
        candidates = []
        
        for name1, name2, overall, signals in all_scores:
            should_merge = False
            merge_reason = None
            
            # Criterion 1: High BERT similarity (semantic duplicates)
            bert_sim = signals.get('bert_similarity', 0.0)
            if bert_sim >= thresholds['bert_auto_approve']:
                should_merge = True
                merge_reason = f"high_bert_similarity_{bert_sim:.3f}"
            
            # Criterion 2: High name similarity (naming duplicates)
            name_sim = signals.get('name_similarity', 0.0)
            if name_sim >= thresholds['name_auto_approve']:
                should_merge = True
                merge_reason = f"high_name_similarity_{name_sim:.3f}"
            
            # Criterion 3: Multi-signal consensus
            if overall >= thresholds['combined_threshold']:
                # Check for signal consensus (at least 2 strong signals)
                strong_signals = sum([
                    signals.get('bert_similarity', 0) > 0.50,
                    signals.get('tfidf_similarity', 0) > 0.40,
                    signals.get('keyword_overlap', 0) > 0.30,
                    signals.get('name_similarity', 0) > 0.50
                ])
                
                if strong_signals >= 2:
                    should_merge = True
                    merge_reason = f"consensus_{strong_signals}_signals"
            
            # Criterion 4: Perfect keyword match (domain-specific duplicates)
            keyword_overlap = signals.get('keyword_overlap', 0.0)
            if keyword_overlap >= 0.80:  # 80%+ keyword overlap
                should_merge = True
                merge_reason = f"high_keyword_overlap_{keyword_overlap:.3f}"
            
            if should_merge:
                # Convert names back to IDs (this is a simplification - in real implementation, track IDs)
                candidates.append((name1, name2, overall, signals, merge_reason))
        
        return candidates
    
    def _ai_validate_candidates(
        self, 
        candidates: List[Tuple], 
        categories: List[Category]
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """
        Use AI agent to validate borderline merge decisions.
        
        For now, this is a placeholder - could integrate with PydanticAI agent
        to make nuanced decisions about edge cases.
        """
        # TODO: Implement AI agent validation
        # For borderline cases (score 0.3-0.6), ask AI agent:
        # "Should these categories be merged? Category 1: ... Category 2: ..."
        
        validated = []
        for candidate in candidates:
            # For now, accept all candidates that passed multi-criteria
            # In future: AI agent can reject/approve borderline cases
            name1, name2, score, signals, reason = candidate
            
            # Convert back to category IDs
            cat1_id = next((c.id for c in categories if c.name == name1), name1)
            cat2_id = next((c.id for c in categories if c.name == name2), name2)
            
            validated.append((cat1_id, cat2_id, score, signals))
        
        return validated
    
    def _log_consolidation_analysis(
        self, 
        all_scores: List[Tuple], 
        thresholds: Dict[str, float],
        candidates: List[Tuple],
        validated: List[Tuple]
    ) -> None:
        """Enhanced logging for consolidation analysis."""
        logger.info(f"=== Analyzed {len(all_scores)} category pairs ===")
        logger.info(f"Adaptive thresholds: BERT≥{thresholds.get('bert_auto_approve', 0):.3f}, Combined≥{thresholds.get('combined_threshold', 0):.3f}, Name≥{thresholds.get('name_auto_approve', 0):.3f}")
        
        # Show top pairs (regardless of threshold)
        all_scores.sort(key=lambda x: x[2], reverse=True)
        for name1, name2, score, sigs in all_scores[:3]:
            logger.info(f"  {name1} <-> {name2}: {score:.3f}")
            logger.info(f"    Signals: tfidf={sigs.get('tfidf_similarity', 0):.3f}, bert={sigs.get('bert_similarity', 0):.3f}, keywords={sigs.get('keyword_overlap', 0):.3f}, name={sigs.get('name_similarity', 0):.3f}")
        
        if candidates:
            logger.info(f"Found {len(validated)} merge candidates (after AI validation)")
        else:
            logger.info("No merge candidates found")
    
    def _analyze_similarity(self, cat1: Category, cat2: Category) -> Dict[str, float]:
        """Multi-signal similarity analysis."""
        signals = {}
        
        # Signal 1: TF-IDF similarity (lexical)
        signals['tfidf_similarity'] = self._tfidf_similarity(cat1, cat2)
        
        # Signal 2: BERT similarity (semantic) - MOST IMPORTANT for catching duplicates
        if self.bert_encoder:
            signals['bert_similarity'] = self._bert_similarity(cat1, cat2)
        
        # Signal 3: Keyword overlap
        signals['keyword_overlap'] = self._keyword_overlap(cat1, cat2)
        
        # Signal 4: Name similarity (catches "Finance Markets" vs "Finance Economics")
        signals['name_similarity'] = self._name_similarity(cat1.name, cat2.name)
        
        return signals
    
    def _compute_merge_score(self, signals: Dict[str, float], aggressive: bool) -> float:
        """
        Compute weighted merge score from multiple signals.
        
        Aggressive mode: Trust BERT heavily, require fewer confirming signals
        Conservative mode: Require multiple strong signals
        """
        if aggressive:
            # Early-stage: trust semantic similarity
            weights = {
                'tfidf_similarity': 0.15,
                'bert_similarity': 0.50,  # BERT is king
                'keyword_overlap': 0.20,
                'name_similarity': 0.15
            }
        else:
            # Late-stage: require consensus
            weights = {
                'tfidf_similarity': 0.25,
                'bert_similarity': 0.35,
                'keyword_overlap': 0.25,
                'name_similarity': 0.15
            }
        
        score = 0.0
        for signal, value in signals.items():
            weight = weights.get(signal, 0.0)
            score += weight * value
        
        return score
    
    def _tfidf_similarity(self, cat1: Category, cat2: Category) -> float:
        """Compute TF-IDF cosine similarity."""
        doc1 = self._category_to_doc(cat1)
        doc2 = self._category_to_doc(cat2)
        
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            X = vectorizer.fit_transform([doc1, doc2])
            sim = cosine_similarity(X[0:1], X[1:2])[0, 0]
            return float(sim)
        except Exception:
            return 0.0
    
    def _bert_similarity(self, cat1: Category, cat2: Category) -> float:
        """Compute BERT semantic similarity."""
        if not self.bert_encoder:
            return 0.0
        
        doc1 = self._category_to_doc(cat1)
        doc2 = self._category_to_doc(cat2)
        
        try:
            emb1 = self.bert_encoder.encode([doc1])
            emb2 = self.bert_encoder.encode([doc2])
            sim = self.bert_encoder.cosine_similarity(emb1, emb2)
            return float(sim)
        except Exception:
            return 0.0
    
    def _keyword_overlap(self, cat1: Category, cat2: Category) -> float:
        """Compute keyword Jaccard similarity."""
        kw1 = set(k.lower() for k in (cat1.keywords or []))
        kw2 = set(k.lower() for k in (cat2.keywords or []))
        
        if not kw1 or not kw2:
            return 0.0
        
        intersection = len(kw1 & kw2)
        union = len(kw1 | kw2)
        
        return intersection / union if union > 0 else 0.0
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Compute name similarity using word overlap ratio."""
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))
        
        return overlap / min_len if min_len > 0 else 0.0
    
    def _category_to_doc(self, category: Category) -> str:
        """Convert category to document string."""
        parts = [
            category.name or "",
            category.description or "",
            " ".join(category.keywords or [])
        ]
        return " ".join(parts) or " "

# ------------------
# Adaptive Top-K Selection
# ------------------
class AdaptiveTopKSelector:
    """
    Context-aware top-K selection for candidate retrieval.
    
    Adapts K based on:
    - Pass number (later passes = larger K for better recall)
    - Category count (more categories = larger K)
    - Text complexity (complex texts = larger K)
    - New category creation (check more thoroughly)
    """
    
    def __init__(self, base_k: int = 3, max_k: int = 10, min_k: int = 2):
        self.base_k = base_k
        self.max_k = max_k
        self.min_k = min_k
    
    def compute_k(
        self,
        current_pass: int,
        max_passes: int,
        num_categories: int,
        text_complexity: float = 0.5,  # 0=simple, 1=complex
        is_creating_new: bool = False
    ) -> int:
        """Compute adaptive K based on context."""
        k = self.base_k
        
        # Factor 1: Pass number (later passes = larger K)
        if max_passes > 1:
            pass_ratio = current_pass / max_passes
            k = int(self.base_k * (1 + 0.5 * pass_ratio))
        
        # Factor 2: Category count (more categories = larger K)
        if num_categories > 10:
            k += 1
        if num_categories > 20:
            k += 1
        
        # Factor 3: Text complexity (complex texts = larger K)
        if text_complexity > 0.7:
            k += 1
        
        # Factor 4: Creating new category (check more thoroughly)
        if is_creating_new:
            k = int(k * 1.5)
        
        # Clamp to bounds
        k = max(self.min_k, min(k, self.max_k))
        k = min(k, num_categories) if num_categories > 0 else k
        
        return k
    
    def estimate_text_complexity(self, text: str) -> float:
        """
        Estimate text complexity (0=simple, 1=complex).
        
        Simple heuristic based on:
        - Token count
        - Vocabulary richness
        - Average word length
        """
        tokens = text.split()
        if not tokens:
            return 0.0
        
        # Factor 1: Length (longer = more complex)
        length_score = min(1.0, len(tokens) / 200)
        
        # Factor 2: Vocabulary richness (unique tokens / total tokens)
        vocab_score = len(set(tokens)) / len(tokens)
        
        # Factor 3: Average word length
        avg_word_len = sum(len(w) for w in tokens) / len(tokens)
        word_len_score = min(1.0, (avg_word_len - 3) / 7)  # Normalize around 3-10 chars
        
        # Combine
        complexity = (length_score + vocab_score + word_len_score) / 3
        return min(1.0, max(0.0, complexity))

async def consolidate_categories(state: Dict[str, Any]) -> Dict[str, Any]:
    """Consolidate similar categories using enhanced multi-signal analysis."""
    logger.info("NODE consolidate called")
    new_state = dict(state)
    
    categories_raw = state.get('categories', [])
    categories = [Category(**c) for c in categories_raw]
    
    if len(categories) <= 1:
        return _skip_consolidation(new_state, categories)
    
    # Get config and determine if we should be aggressive
    config = state.get('config', DEFAULT_CONFIG)
    current_pass = state.get('current_pass', 1)
    max_passes = state.get('max_passes', 2)
    aggressive = current_pass < max_passes  # Be aggressive in early passes
    
    # Initialize MultiSignalConsolidator with BERT if available
    bert_encoder = None
    if config.retrieval_mode in [RETRIEVAL_MODE.BERT, RETRIEVAL_MODE.HYBRID]:
        try:
            bert_encoder = BERTEncoder.get_instance()
        except Exception as e:
            logger.warning(f"BERT not available for consolidation: {e}")
    
    consolidator = MultiSignalConsolidator(bert_encoder=bert_encoder)
    
    # Find merge candidates using multi-signal analysis
    candidate_pairs = consolidator.find_merge_candidates(categories, aggressive=aggressive)
    
    if not candidate_pairs:
        logger.info("No merge candidates found")
        return _skip_consolidation(new_state, categories)
    
    # Log merge candidates for transparency
    logger.info(f"Found {len(candidate_pairs)} merge candidates (aggressive={aggressive})")
    for cat1_id, cat2_id, score, signals in candidate_pairs[:3]:  # Show top 3
        cat1 = next(c for c in categories if c.id == cat1_id)
        cat2 = next(c for c in categories if c.id == cat2_id)
        logger.info(f"  {cat1.name} <-> {cat2.name}: score={score:.3f}")
        if logger.isEnabledFor(logging.DEBUG):
            for signal_name, value in signals.items():
                logger.debug(f"    {signal_name}={value:.3f}")
    
    # Build merge result for LLM validation (or auto-approve high-confidence)
    merge_result = _build_merge_result_from_candidates(candidate_pairs, categories, aggressive)
    
    if not merge_result or not merge_result.get('should_merge', False):
        logger.info("No merges approved")
        return _skip_consolidation(new_state, categories)
    
    new_categories, updated_assignments = _apply_merges(
        categories,
        state.get('assignments', []),
        merge_result
    )
    
    logger.info(f"Consolidated {len(categories)} → {len(new_categories)} categories")
    
    new_state.update({
        'consolidation_complete': True,
        'categories': new_categories,
        'assignments': updated_assignments
    })
    return new_state

def _build_merge_result_from_candidates(
    candidate_pairs: List[Tuple[str, str, float, Dict]],
    categories: List[Category],
    aggressive: bool,
    max_merges: int = 3
) -> Dict[str, Any]:
    """
    Build merge result from candidates, auto-approving high-confidence merges.
    
    Auto-approve if:
    - BERT similarity >= 0.70 OR
    - Overall score >= 0.65 (aggressive) or 0.70 (conservative)
    """
    auto_approve_threshold = 0.65 if aggressive else 0.70
    bert_auto_approve = 0.70  # Match threshold from find_merge_candidates
    
    approved_pairs = []
    merged_ids = set()
    
    for cat1_id, cat2_id, score, signals in candidate_pairs[:max_merges]:
        # Skip if already part of a merge
        if cat1_id in merged_ids or cat2_id in merged_ids:
            continue
        
        # Auto-approve high-confidence merges
        bert_sim = signals.get('bert_similarity', 0.0)
        if bert_sim >= bert_auto_approve or score >= auto_approve_threshold:
            approved_pairs.append([cat1_id, cat2_id])
            merged_ids.update([cat1_id, cat2_id])
            
            cat1 = next(c for c in categories if c.id == cat1_id)
            cat2 = next(c for c in categories if c.id == cat2_id)
            logger.info(f"Auto-approved merge: {cat1.name} + {cat2.name} (score={score:.3f})")
    
    if not approved_pairs:
        return {'should_merge': False}
    
    # Build merged category definitions
    merged_categories = []
    for cat1_id, cat2_id in approved_pairs:
        cat1 = next(c for c in categories if c.id == cat1_id)
        cat2 = next(c for c in categories if c.id == cat2_id)
        
        # Prefer category with more texts
        if cat1.text_count >= cat2.text_count:
            primary, secondary = cat1, cat2
        else:
            primary, secondary = cat2, cat1
        
        # Merge keywords (unique)
        merged_keywords = list(set((primary.keywords or []) + (secondary.keywords or [])))[:12]
        
        merged_categories.append({
            'id': primary.id,
            'name': primary.name,
            'description': primary.description or secondary.description or "Merged category",
            'keywords': merged_keywords
        })
    
    return {
        'should_merge': True,
        'category_pairs': approved_pairs,
        'merged_categories': merged_categories
    }

def _skip_consolidation(state: Dict[str, Any], categories: List[Category]) -> Dict[str, Any]:
    """Skip consolidation."""
    state.update({
        'consolidation_complete': True,
        'categories': [c.model_dump() for c in categories],
        'assignments': state.get('assignments', [])
    })
    return state

def _apply_merges(
    categories: List[Category],
    assignments: List[Dict[str, Any]],
    merge_result: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply category merges."""
    category_pairs = merge_result.get('category_pairs', [])
    merged_defs = merge_result.get('merged_categories', [])
    
    mapping = {}
    merged_by_id = {
        mc.get('id'): build_category_from_raw(mc, mc.get('name', 'merged'), "")
        for mc in merged_defs
    }
    
    for pair in category_pairs:
        for old_id in pair:
            for mid in merged_by_id.keys():
                if mid not in mapping.values():
                    mapping[old_id] = mid
                    break
    
    new_categories = []
    processed_old = set()
    
    for mid, mcat in merged_by_id.items():
        total_count = sum(
            c.text_count for c in categories
            if c.id in [old for old, new in mapping.items() if new == mid]
        )
        processed_old.update(
            old for old, new in mapping.items() if new == mid
        )
        mcat.text_count = total_count
        new_categories.append(mcat.model_dump())
    
    for c in categories:
        if c.id not in processed_old:
            new_categories.append(c.model_dump())
    
    updated_assignments = []
    for a in assignments:
        old_id = a.get('category_id')
        new_id = mapping.get(old_id, old_id)
        updated = dict(a)
        updated['category_id'] = new_id
        if new_id != old_id:
            updated['reasoning'] = f"Reassigned after merge: {updated.get('reasoning', '')}"
        updated_assignments.append(updated)
    
    return new_categories, updated_assignments

# ------------------
# Router and graph
# ------------------

def route_decision(state: Dict[str, Any]) -> str:
    """Route to next node with proper termination."""
    if state.get('consolidation_complete', False):
        return END
    if state.get('processing_complete', False):
        return 'consolidate'
    
    decision = state.get('decision', '')
    if decision == 'analyzed':
        return 'load_next'
    if decision == 'analyze':
        return 'analyze'
    if decision == 'consolidate':
        return 'consolidate'
    
    # Default to load_next for initial state
    return 'load_next'

def build_clusterizer_graph():
    """Build the processing graph with fixed recursion logic."""
    workflow = StateGraph(dict)
    workflow.add_node('load_next', load_next_batch)
    workflow.add_node('analyze', analyze_batch_parallel)
    workflow.add_node('consolidate', consolidate_categories)
    workflow.set_entry_point('load_next')
    
    # The key fix: ensure proper termination conditions
    workflow.add_conditional_edges(
        'load_next',
        route_decision,
        {
            'analyze': 'analyze',
            'consolidate': 'consolidate',
            END: END
        }
    )
    
    workflow.add_edge('analyze', 'load_next')
    workflow.add_edge('consolidate', END)
    
    return workflow.compile(checkpointer=MemorySaver())

async def execute_clusterizer_graph(
    texts: List[str],
    max_passes: int = DEFAULT_MAX_PASSES,
    prefilter_k: int = DEFAULT_PREFILTER_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent: int = DEFAULT_MAX_PARALLEL_MERGES,
    config: Optional[ClusterizerConfig] = DEFAULT_CONFIG,
    bert_model: Optional[str] = DEFAULT_BERT_MODEL
) -> Dict[str, Any]:
    graph = build_clusterizer_graph()
    
    initial_state = {
        'texts': texts,
        'current_index': 0,
        'current_batch': [],
        'categories': [],
        'assignments': [],
        'assignments_map': {},
        'decision': 'start',
        'processing_complete': False,
        'consolidation_complete': False,
        'current_pass': 1,
        'max_passes': max_passes,
        'prefilter_k': prefilter_k,
        'batch_size': batch_size,
        'config': config,
        'bert_model': bert_model,
        '_loop_counter': 0
    }
    
    graph_config = {
        'configurable': {'thread_id': 'clusterizer_1'},
        'recursion_limit': 1000
    }
    
    logger.info("Starting BERT-enhanced clusterization")
    logger.info("  Texts: %d | Batch: %d | Mode: %s | Strategy: %s", 
                len(texts), batch_size, config.retrieval_mode, config.scoring_strategy)
    
    step_count = 0
    last_state = None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BERT-ENHANCED CLUSTERIZATION")
    logger.info(f"{'='*60}")
    logger.info(f"Texts: {len(texts)}")
    logger.info(f"Config: {config.get_description()}")
    logger.info(f"Batch size: {batch_size} (parallel)")
    logger.info(f"Max concurrent: {max_concurrent}")
    logger.info(f"Max passes: {max_passes}")
    logger.info(f"{'='*60}\n")
    
    async for event in graph.astream(initial_state, graph_config):
        step_count += 1
        
        node_name = list(event.keys())[0] if event else "unknown"
        logger.info(f"Step {step_count:2d}: {node_name:20s}")
        
        try:
            payload = next(iter(event.values()))
            if isinstance(payload, dict):
                if 'current_index' in payload:
                    idx = payload.get('current_index', 0)
                    total = len(payload.get('texts', []))
                    pass_num = payload.get('current_pass', 1)
                    cat_count = len(payload.get('categories', []))
                    logger.info(f" | Pass {pass_num} | Text {idx}/{total} | Categories: {cat_count}")
                
                if any(k in payload for k in ('categories', 'consolidation_complete', 'assignments')):
                    last_state = payload
        except Exception as e:
            logger.info(f" [Error: {e}]")
    
    logger.info(f"\n{'='*60}\n")
    
    if last_state is None:
        final_state = await graph.ainvoke(initial_state, graph_config)
    else:
        final_state = last_state
    
    logger.info("Clusterization complete after %d steps", step_count)

    # Statistics
    assignments_list = final_state.get('assignments', [])
    if assignments_list:
        confidences = [a.get('confidence', 0.0) for a in assignments_list]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        high_conf = sum(1 for c in confidences if c >= 0.75)
        med_conf = sum(1 for c in confidences if 0.5 <= c < 0.75)
        low_conf = sum(1 for c in confidences if c < 0.5)
    else:
        avg_confidence = min_confidence = max_confidence = 0.0
        high_conf = med_conf = low_conf = 0
    
    return {
        'categories': [
            Category(**c) if not isinstance(c, Category) else c
            for c in final_state.get('categories', [])
        ],
        'assignments': [
            CategoryAssignment(**a) if not isinstance(a, CategoryAssignment) else a
            for a in assignments_list
        ],
        'metadata': {
            'total_passes': final_state.get('current_pass', 1),
            'total_texts': len(texts),
            'total_categories': len(final_state.get('categories', [])),
            'total_steps': step_count,
            'parallel_processing': True,
            'batch_size': batch_size,
            'max_concurrent': max_concurrent,
            'config': {
                'retrieval_mode': config.retrieval_mode,
                'scoring_strategy': config.scoring_strategy,
                'new_category_threshold': config.new_category_threshold,
                'new_category_bonus': config.new_category_bonus,
            },
            'retrieval_mode': config.retrieval_mode,
            'bert_enabled': config.retrieval_mode in [RETRIEVAL_MODE.BERT, RETRIEVAL_MODE.HYBRID],
            'confidence_stats': {
                'average': round(avg_confidence, 3),
                'min': round(min_confidence, 3),
                'max': round(max_confidence, 3),
                'high_confidence_count': high_conf,
                'medium_confidence_count': med_conf,
                'low_confidence_count': low_conf
            },
            'used_tree_merge': False  # Standard execution
        }
    }

# ------------------
# Public API
# ------------------
async def clusterize_texts_large(
    texts: List[str],
    max_passes: int = 2,
    prefilter_k: int = DEFAULT_PREFILTER_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent: int = 5,
    config: Optional[ClusterizerConfig] = None,
    # New parameters for large datasets
    enable_tree_merge: bool = True,
    texts_per_workflow: int = DEFAULT_TEXTS_PER_WORKFLOW,
    token_threshold: int = DEFAULT_TOKEN_BIG_TEXT,
    max_parallel_merges: int = DEFAULT_MAX_PARALLEL_MERGES
) -> Dict[str, Any]:
    """
    Enhanced clusterization that handles arbitrarily large datasets.
    
    Workflow:
    1. If dataset is small (≤ texts_per_workflow), use standard clusterize_texts()
    2. If dataset is large:
       a. Use TextAssignmentManager for dry assignment (with big text chunking)
       b. Use TreeMergeProcessor for hierarchical execution and merging
       c. Consolidate big texts at the end
    
    Args:
        texts: List of text strings to categorize
        max_passes: Number of passes per workflow execution
        prefilter_k: Number of candidate categories to consider
        batch_size: Texts per batch (for parallel processing within workflow)
        max_concurrent: Maximum concurrent LLM calls
        config: ClusterizerConfig instance (recommended, defaults to CONFIG_BALANCED_HYBRID)
        
        # Large dataset parameters
        enable_tree_merge: Whether to enable tree merge for large datasets
        texts_per_workflow: Maximum texts per single workflow execution
        token_threshold: Token threshold for big text detection (texts > M tokens get chunked)
        max_parallel_merges: Maximum parallel merge operations
    
    Returns:
        Dict with 'categories', 'assignments', and 'metadata'
        
        Additional metadata for large datasets:
        - 'used_tree_merge': bool
        - 'total_workflows': int
        - 'big_texts_consolidated': int
        - 'tree_merge_levels': int
    
    Examples:
        # Small dataset (automatic)
        >>> result = await clusterize_texts_large(
        ...     texts[:100],
        ...     config=CONFIG_BALANCED_HYBRID
        ... )
        
        # Large dataset with tree merge
        >>> result = await clusterize_texts_large(
        ...     texts[:10000],
        ...     texts_per_workflow=500,
        ...     token_threshold=200,
        ...     config=CONFIG_BALANCED_HYBRID
        ... )
        
        # Force single execution (no tree merge)
        >>> result = await clusterize_texts_large(
        ...     texts,
        ...     enable_tree_merge=False,
        ...     config=CONFIG_BALANCED_HYBRID
        ... )
    """
    
    # Handle config
    if config is None:
        config = CONFIG_BALANCED_HYBRID
    # Decision point: Use tree merge or standard execution?
    if enable_tree_merge and len(texts) > texts_per_workflow:
        logger.info(f"🌳 Large dataset detected ({len(texts)} texts). Using tree merge strategy.")
        return await _clusterize_with_tree_merge(
            texts=texts,
            max_passes=max_passes,
            prefilter_k=prefilter_k,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            config=config,
            texts_per_workflow=texts_per_workflow,
            token_threshold=token_threshold,
            max_parallel_merges=max_parallel_merges
        )
    else:
        logger.info(f"📄 Standard execution for {len(texts)} texts.")
        result = await clusterize_texts(
            texts=texts,
            max_passes=max_passes,
            prefilter_k=prefilter_k,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            config=config
        )
        result['metadata']['used_tree_merge'] = False
        return result


async def _clusterize_with_tree_merge(
    texts: List[str],
    max_passes: int,
    prefilter_k: int,
    batch_size: int,
    max_concurrent: int,
    config: ClusterizerConfig,
    texts_per_workflow: int,
    token_threshold: int,
    max_parallel_merges: int
) -> Dict[str, Any]:
    """
    Internal function: Execute clustering with tree merge strategy.
    """
    
    from text_assignment_manager import TextAssignmentManager, Text, Tokenizer
    from tree_merge_processor import TreeMergeProcessor
    
    # Step 1: Calculate optimal parameters
    logger.info(f"Step 1: Calculating optimal workflow distribution...")
    
    # Calculate batch/workflow parameters to fit within texts_per_workflow
    # We want: batches_per_workflow * batch_size ≈ texts_per_workflow
    batches_per_workflow = max(1, texts_per_workflow // batch_size)
    actual_texts_per_workflow = min(texts_per_workflow, batches_per_workflow * batch_size)
    
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Batches per workflow: {batches_per_workflow}")
    logger.info(f"  Texts per workflow: {actual_texts_per_workflow}")
    
    # Step 2: Create Text objects and perform dry assignment
    logger.info(f"\nStep 2: Performing dry assignment...")
    
    text_objects = [
        Text(id=f"text_{i}", content=text)
        for i, text in enumerate(texts)
    ]
    
    assignment_manager = TextAssignmentManager(
        batch_size=batch_size,
        max_batches_per_workflow=batches_per_workflow,
        token_threshold=token_threshold
    )
    
    dry_assignment = assignment_manager.dry_assign(text_objects)
    
    logger.info(f"\n{dry_assignment.get_summary()}")
    
    # Step 3: Process with TreeMergeProcessor
    logger.info(f"\nStep 3: Processing with tree merge...")
    
    processor = TreeMergeProcessor(max_parallel_merges=max_parallel_merges)
    
    result = await processor.process_with_priority_queue(
        dry_assignment_result=dry_assignment,
        clusterizer_config=config,
        max_passes=max_passes,
        prefilter_k=prefilter_k,
        batch_size=batch_size,
        max_concurrent=max_concurrent
    )
    
    # Step 4: Enhance metadata
    result['metadata']['used_tree_merge'] = True
    result['metadata']['total_workflows'] = dry_assignment.total_workflows_needed
    result['metadata']['texts_per_workflow'] = actual_texts_per_workflow
    result['metadata']['token_threshold'] = token_threshold
    
    return result


# Backward compatibility: Keep old clusterize_texts signature
# But add auto-detection for large datasets
async def clusterize_texts(
    texts: List[str],
    max_passes: int = 2,
    prefilter_k: int = DEFAULT_PREFILTER_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT_LLM_CALLS,
    retrieval_mode: Optional[str] = None,
    bert_model: str = DEFAULT_BERT_MODEL,
    config: Optional[ClusterizerConfig] = None,
    max_texts_per_run: int = DEFAULT_TEXTS_PER_WORKFLOW,
    token_threshold: int = DEFAULT_TOKEN_BIG_TEXT,
    max_parallel_merges: int = DEFAULT_MAX_PARALLEL_MERGES
) -> Dict[str, Any]:
    """
    Clusterize texts with automatic tree merge for large datasets.
    
    This is the enhanced version that automatically detects large datasets
    and uses tree merge when appropriate.
    
    Args:
        texts: List of text strings to categorize
        max_passes: Number of passes over the data
        prefilter_k: Number of candidate categories to consider
        batch_size: Texts per batch (processed in parallel)
        max_concurrent: Maximum concurrent LLM calls
        retrieval_mode: DEPRECATED - Use `config` instead
        bert_model: Sentence-transformer model name
        config: ClusterizerConfig instance (recommended)
        max_texts_per_run: Threshold for enabling tree merge (default: 500)
    
    Returns:
        Dict with 'categories', 'assignments', and 'metadata'
    
    Examples:
        # Standard usage (auto-detects if tree merge needed)
        >>> result = await clusterize_texts(texts, config=CONFIG_BALANCED_HYBRID)
        
        # Force higher threshold before tree merge
        >>> result = await clusterize_texts(
        ...     texts,
        ...     max_texts_per_run=1000,
        ...     config=CONFIG_BALANCED_HYBRID
        ... )
    """
    
    # Handle config resolution
    if config is None:
        if retrieval_mode is not None:
            logger.warning(
                "Using retrieval_mode parameter is deprecated. "
                "Please use config=CONFIG_* presets instead."
            )
            if retrieval_mode == RETRIEVAL_MODE.HYBRID:
                config = CONFIG_BALANCED_HYBRID
            elif retrieval_mode == RETRIEVAL_MODE.BERT:
                config = CONFIG_SEMANTIC_BERT
            elif retrieval_mode == RETRIEVAL_MODE.TFIDF:
                config = CONFIG_BALANCED_TFIDF
            else:
                logger.warning(f"Unknown retrieval_mode '{retrieval_mode}', using default")
                config = DEFAULT_CONFIG
        else:
            config = DEFAULT_CONFIG
    
    # Auto-detect: Use tree merge for large datasets
    if len(texts) > max_texts_per_run:
        logger.info(f"🌳 Auto-detected large dataset ({len(texts)} > {max_texts_per_run}). Using tree merge.")
        return await clusterize_texts_large(
            texts=texts,
            max_passes=max_passes,
            prefilter_k=prefilter_k,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            config=config,
            texts_per_workflow=max_texts_per_run,
            token_threshold=token_threshold,
            max_parallel_merges=max_parallel_merges
        )
    
    # Standard execution for small datasets
    logger.info(f"Using configuration: {config.get_description()}")
    global _rate_limiter
    _rate_limiter = LLMRateLimiter(max_concurrent)
    
    # Apply adaptive batch sizing
    if batch_size == DEFAULT_BATCH_SIZE:
        adaptive_batch_size = calculate_adaptive_batch_size(len(texts))
        if adaptive_batch_size != batch_size:
            logger.info(f"Auto-adjusting batch size: {batch_size} → {adaptive_batch_size}")
            batch_size = adaptive_batch_size
    
    return await execute_clusterizer_graph(
        texts=texts,
        max_passes=max_passes,
        prefilter_k=prefilter_k,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        config=config,
        bert_model=bert_model
    )
    
    

# ------------------
# Persistence Functions
# ------------------

def save_results(result: Dict[str, Any], filename: str = None) -> str:
    """
    Save clusterization results to JSON file with timestamp.
    
    Args:
        result: Result from clusterize_texts()
        filename: Optional custom filename (defaults to timestamp-based)
        
    Returns:
        Path to saved file
    """
    import json
    from datetime import datetime
    import os
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clusterization_results_{timestamp}.json"
    
    # Convert Category objects to dictionaries for JSON serialization
    serializable_result = {
        'categories': [
            {
                'id': cat.id,
                'name': cat.name,
                'description': cat.description,
                'keywords': cat.keywords,
                'text_count': cat.text_count
            } for cat in result['categories']
        ],
        'assignments': [
            {
                'text': assign.text,
                'category_id': assign.category_id,
                'confidence': assign.confidence,
                'reasoning': assign.reasoning
            } for assign in result['assignments']
        ],
        'metadata': result['metadata']
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {filename}")
    return filename

def load_results(filename: str) -> Dict[str, Any]:
    """
    Load clusterization results from JSON file.
    
    Args:
        filename: Path to saved JSON file
        
    Returns:
        Loaded result dictionary with Category and CategoryAssignment objects
    """
    import json
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert back to Category and CategoryAssignment objects
    categories = [
        Category(
            id=cat['id'],
            name=cat['name'], 
            description=cat['description'],
            keywords=cat['keywords'],
            text_count=cat['text_count']
        ) for cat in data['categories']
    ]
    
    assignments = [
        CategoryAssignment(
            text=assign['text'],
            category_id=assign['category_id'],
            confidence=assign['confidence'],
            reasoning=assign['reasoning']
        ) for assign in data['assignments']
    ]
    
    return {
        'categories': categories,
        'assignments': assignments,
        'metadata': data['metadata']
    }

def save_categories_csv(result: Dict[str, Any], filename: str = None) -> str:
    """
    Save categories to CSV for easy viewing/analysis.
    
    Args:
        result: Result from clusterize_texts()
        filename: Optional custom filename
        
    Returns:
        Path to saved CSV file
    """
    import csv
    from datetime import datetime
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"categories_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category ID', 'Name', 'Description', 'Text Count', 'Keywords'])
        
        for cat in result['categories']:
            writer.writerow([
                cat.id,
                cat.name,
                cat.description,
                cat.text_count,
                ', '.join(cat.keywords) if cat.keywords else ''
            ])
    
    print(f"✅ Categories saved to CSV: {filename}")
    return filename

def save_assignments_csv(result: Dict[str, Any], filename: str = None) -> str:
    """
    Save assignments to CSV for easy viewing/analysis.
    
    Args:
        result: Result from clusterize_texts()
        filename: Optional custom filename
        
    Returns:
        Path to saved CSV file
    """
    import csv
    from datetime import datetime
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"assignments_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Text', 'Category ID', 'Category Name', 'Confidence', 'Reasoning'])
        
        # Create category lookup for names
        cat_lookup = {cat.id: cat.name for cat in result['categories']}
        
        for assign in result['assignments']:
            writer.writerow([
                assign.text,
                assign.category_id,
                cat_lookup.get(assign.category_id, 'Unknown'),
                assign.confidence,
                assign.reasoning
            ])
    
    print(f"✅ Assignments saved to CSV: {filename}")
    return filename

async def clusterize_and_save(
    texts: List[str],
    save_json: bool = True,
    save_csv: bool = True,
    output_prefix: str = None,
    **clusterize_kwargs
) -> Dict[str, Any]:
    """
    Convenience function that clusterizes and automatically saves results.
    
    Args:
        texts: List of texts to clusterize
        save_json: Whether to save complete results as JSON
        save_csv: Whether to save categories and assignments as CSV
        output_prefix: Optional prefix for output files
        **clusterize_kwargs: Arguments passed to clusterize_texts()
        
    Returns:
        Clusterization result (same as clusterize_texts)
    """
    from datetime import datetime
    
    print(f"🚀 Starting clusterization of {len(texts)} texts...")
    result = await clusterize_texts(texts, **clusterize_kwargs)
    
    print(f"✅ Clusterization complete: {len(result['categories'])} categories, {len(result['assignments'])} assignments")
    
    # Generate timestamp for consistent naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = []
    
    if save_json:
        json_filename = f"{output_prefix}_{timestamp}.json" if output_prefix else None
        saved_files.append(save_results(result, json_filename))
    
    if save_csv:
        cat_filename = f"{output_prefix}_categories_{timestamp}.csv" if output_prefix else None
        assign_filename = f"{output_prefix}_assignments_{timestamp}.csv" if output_prefix else None
        saved_files.append(save_categories_csv(result, cat_filename))
        saved_files.append(save_assignments_csv(result, assign_filename))
    
    print(f"📁 All files saved: {', '.join(saved_files)}")
    
    return result

# ------------------
# Meta-Categorization (Experimental)
# ------------------

async def meta_categorize_categories(
    categories: List[Category],
    config: Optional[ClusterizerConfig] = None
) -> Dict[str, Any]:
    """
    Experimental: Categorize the categories themselves using our existing agentic workflow.
    
    Simply feeds category information as texts to the same intelligent system
    that already handles LLM agents, BERT embeddings, and adaptive consolidation.
    
    Args:
        categories: List of categories to meta-categorize
        config: Configuration (defaults to SEMANTIC_BERT for concept analysis)
        
    Returns:
        Dict with 'meta_categories', 'category_assignments', and 'taxonomy'
    """
    if len(categories) <= 2:
        # Too few categories for meaningful meta-categorization
        return {
            'meta_categories': [],
            'category_assignments': [],
            'taxonomy': {},
            'meta_enabled': False
        }
    
    # Use semantic BERT config for concept-level analysis
    if config is None:
        config = CONFIG_SEMANTIC_BERT
    
    # Convert categories to text representations for our agentic workflow
    category_texts = []
    for cat in categories:
        # Rich representation combining all category information
        text_parts = [cat.name]
        if cat.description:
            text_parts.append(cat.description)
        if cat.keywords:
            text_parts.extend(cat.keywords[:5])  # Top keywords only
        
        category_text = " ".join(text_parts)
        category_texts.append(category_text)
    
    logger.info(f"Meta-categorizing {len(categories)} categories using agentic workflow...")
    
    # Apply our existing intelligent clusterization to category representations
    # This automatically gets: LLM agents, BERT embeddings, adaptive consolidation, confidence scoring
    meta_result = await clusterize_texts(
        category_texts,
        config=config,
        batch_size=min(10, len(categories)),
        max_passes=2  # Conservative for meta-level
    )
    
    # Build taxonomy mapping from the results
    taxonomy = {}
    category_assignments = []
    
    for i, assignment in enumerate(meta_result['assignments']):
        original_category = categories[i]
        meta_category_id = assignment.category_id
        
        # Find the meta-category details
        meta_category = next(
            (mc for mc in meta_result['categories'] if mc.id == meta_category_id), 
            None
        )
        
        if meta_category:
            if meta_category_id not in taxonomy:
                taxonomy[meta_category_id] = {
                    'meta_name': meta_category.name,
                    'meta_description': meta_category.description,
                    'categories': []
                }
            
            taxonomy[meta_category_id]['categories'].append({
                'id': original_category.id,
                'name': original_category.name,
                'text_count': original_category.text_count,
                'confidence': assignment.confidence
            })
            
            category_assignments.append({
                'category_id': original_category.id,
                'category_name': original_category.name,
                'meta_category_id': meta_category_id,
                'meta_category_name': meta_category.name,
                'confidence': assignment.confidence,
                'reasoning': assignment.reasoning
            })
    
    logger.info(f"Meta-categorization complete: {len(categories)} → {len(meta_result['categories'])} meta-categories")
    
    return {
        'meta_categories': meta_result['categories'],
        'category_assignments': category_assignments,
        'taxonomy': taxonomy,
        'meta_enabled': True,
        'compression_ratio': len(categories) / len(meta_result['categories']) if meta_result['categories'] else 1
    }

def print_taxonomy(taxonomy: Dict[str, Any]) -> None:
    """Print hierarchical taxonomy in a readable format."""
    print(f"\n{'='*60}")
    print(f"HIERARCHICAL TAXONOMY")
    print(f"{'='*60}")
    
    for meta_id, meta_info in taxonomy.items():
        meta_name = meta_info['meta_name']
        categories = meta_info['categories']
        total_texts = sum(cat['text_count'] for cat in categories)
        
        print(f"\n📁 {meta_name} ({len(categories)} categories, {total_texts} texts)")
        if meta_info.get('meta_description'):
            print(f"   {meta_info['meta_description']}")
        
        # Sort categories by text count (descending)
        sorted_cats = sorted(categories, key=lambda x: x['text_count'], reverse=True)
        for cat in sorted_cats:
            conf = cat.get('confidence', 0)
            print(f"   └── {cat['name']}: {cat['text_count']} texts (conf: {conf:.2f})")
    
    print(f"\n{'='*60}\n")

async def clusterize_with_hierarchy(
    texts: List[str],
    enable_meta_categorization: bool = True,
    meta_threshold: int = 5,  # Minimum categories needed for meta-categorization
    **clusterize_kwargs
) -> Dict[str, Any]:
    """
    Enhanced clusterization with optional hierarchical meta-categorization.
    
    Args:
        texts: Input texts to categorize
        enable_meta_categorization: Whether to create meta-categories
        meta_threshold: Minimum number of categories to enable meta-categorization
        **clusterize_kwargs: Arguments passed to clusterize_texts()
        
    Returns:
        Enhanced result with taxonomy information if enabled
    """
    # Step 1: Standard categorization
    result = await clusterize_texts(texts, **clusterize_kwargs)
    
    # Step 2: Meta-categorization if enabled and feasible
    if enable_meta_categorization and len(result['categories']) >= meta_threshold:
        logger.info("Performing experimental meta-categorization...")
        
        meta_result = await meta_categorize_categories(
            result['categories'],
            config=clusterize_kwargs.get('config')
        )
        
        # Enhance the result with taxonomy
        result['taxonomy'] = meta_result
        result['meta_enabled'] = True
        
        # Print taxonomy for immediate feedback
        if meta_result.get('taxonomy'):
            print_taxonomy(meta_result['taxonomy'])
    else:
        result['taxonomy'] = {'meta_enabled': False}
        result['meta_enabled'] = False
        
        if enable_meta_categorization:
            logger.info(f"Meta-categorization skipped: {len(result['categories'])} < {meta_threshold} categories")
    
    return result

# ------------------
# Example usage
# ------------------
async def example_small_dataset():
    """Example: Small dataset (uses standard execution)."""
    
    texts = [f"Sample text {i}" for i in range(10)]
    
    result = await clusterize_texts(
        texts,
        config=CONFIG_BALANCED_HYBRID
    )
    
    print(f"Categories: {len(result['categories'])}")
    print(f"Tree merge used: {result['metadata']['used_tree_merge']}")


async def example_large_dataset():
    """Example: Large dataset (uses tree merge automatically)."""
    
    texts = [f"Sample text {i}" for i in range(10)]
    
    result = await clusterize_texts(
        texts,
        max_texts_per_run=5,  # Will create 2 workflows
        batch_size=5,  # Set small batch size to enable multiple workflows
        config=CONFIG_BALANCED_HYBRID
    )
    
    print(f"Categories: {len(result['categories'])}")
    print(f"Tree merge used: {result['metadata']['used_tree_merge']}")
    if result['metadata']['used_tree_merge']:
        print(f"Total workflows: {result['metadata']['total_workflows']}")
        print(f"Big texts consolidated: {result['metadata'].get('big_texts_consolidated', 0)}")


async def example_explicit_tree_merge():
    """Example: Explicitly use tree merge with custom parameters."""
    
    texts = [f"Sample text {i}" for i in range(10)]
    
    result = await clusterize_texts_large(
        texts,
        texts_per_workflow=5,    # 2 workflows with batch_size=5
        batch_size=5,            # Set small batch size for multiple workflows
        token_threshold=150,     # Lower threshold for big texts
        max_parallel_merges=2,   # Conservative parallelism
        config=CONFIG_SEMANTIC_BERT
    )
    
    print(f"Categories: {len(result['categories'])}")
    print(f"Workflows: {result['metadata']['total_workflows']}")


if __name__ == "__main__":
    import asyncio
    
    print("Example 1: Small dataset")
    asyncio.run(example_small_dataset())
    
    print("\nExample 2: Large dataset (auto tree merge)")
    asyncio.run(example_large_dataset())
    
    print("\nExample 3: Explicit tree merge")
    asyncio.run(example_explicit_tree_merge())