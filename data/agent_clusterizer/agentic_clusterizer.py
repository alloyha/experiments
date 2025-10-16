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

Notes:
    This header intentionally avoids referencing internal refactor details
    (specific class names or historical thresholds) so the top-level
    documentation remains stable as implementation evolves.
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

logger = logging.getLogger("agentic_clusterizer")
logging.basicConfig(level=logging.INFO)

# Constants
MAX_WATCHDOG = 10000
DEFAULT_BATCH_SIZE = 6
MAX_CONCURRENT_LLM_CALLS = 5
SIMILARITY_THRESHOLD = 0.6
DEFAULT_PREFILTER_K = 3
LLM_TIMEOUT_SECONDS = 30.0

# LLM Configuration
LLM_PROVIDER_NAME = 'openai'
LLM_MODEL_NAME = 'gpt-4o-mini'


# ------------------
# Configuration Dataclasses
# ------------------
@dataclass(frozen=True)
class RetrievalMode:
    """Retrieval mode constants for category indexing."""
    TFIDF: str = "tfidf"
    BERT: str = "bert"
    HYBRID: str = "hybrid"  # Recommended

@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights for confidence score calculation."""
    llm_score: float
    tfidf_similarity: float
    bert_similarity: float
    keyword_overlap: float
    category_maturity: float
    pass_number: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for backward compatibility."""
        return {
            'llm_score': self.llm_score,
            'tfidf_similarity': self.tfidf_similarity,
            'bert_similarity': self.bert_similarity,
            'keyword_overlap': self.keyword_overlap,
            'category_maturity': self.category_maturity,
            'pass_number': self.pass_number
        }
    
    def validate(self) -> None:
        """Validate that weights sum to approximately 1.0."""
        total = sum(self.to_dict().values())
        if not (0.95 <= total <= 1.05):
            logger.warning(f"Confidence weights sum to {total:.3f}, expected ~1.0")

@dataclass(frozen=True)
class ScoringStrategy:
    """Scoring strategy constants."""
    BALANCED: str = "balanced"      # Standard scoring
    CONSERVATIVE: str = "conservative"  # Favor established categories
    AGGRESSIVE: str = "aggressive"  # Favor new category creation
    SEMANTIC: str = "semantic"      # Heavy emphasis on BERT/semantic

@dataclass(frozen=True)
class ClusterizerConfig:
    """
    Complete configuration for the clusterizer.
    
    Couples retrieval mode with appropriate confidence weights and scoring strategy.
    """
    retrieval_mode: str
    confidence_weights: ConfidenceWeights
    scoring_strategy: str
    new_category_threshold: float = 0.6  # Min confidence to avoid creating new category
    new_category_bonus: float = 1.1      # Multiplier for new category confidence
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate retrieval mode
        valid_modes = [RETRIEVAL_MODE.TFIDF, RETRIEVAL_MODE.BERT, RETRIEVAL_MODE.HYBRID]
        if self.retrieval_mode not in valid_modes:
            raise ValueError(f"Invalid retrieval_mode: {self.retrieval_mode}")
        
        # Validate scoring strategy
        valid_strategies = [
            SCORING_STRATEGY.BALANCED, 
            SCORING_STRATEGY.CONSERVATIVE,
            SCORING_STRATEGY.AGGRESSIVE,
            SCORING_STRATEGY.SEMANTIC
        ]
        if self.scoring_strategy not in valid_strategies:
            raise ValueError(f"Invalid scoring_strategy: {self.scoring_strategy}")
        
        # Validate thresholds
        if not (0.0 <= self.new_category_threshold <= 1.0):
            raise ValueError(f"new_category_threshold must be in [0.0, 1.0]")
        if not (0.5 <= self.new_category_bonus <= 2.0):
            raise ValueError(f"new_category_bonus must be in [0.5, 2.0]")
        
        # Validate weights
        self.confidence_weights.validate()
    
    def get_description(self) -> str:
        """Get human-readable description of config."""
        return (
            f"{self.scoring_strategy.title()} strategy "
            f"using {self.retrieval_mode.upper()} retrieval "
            f"(threshold={self.new_category_threshold:.2f})"
        )

# Singleton instances
RETRIEVAL_MODE = RetrievalMode()
SCORING_STRATEGY = ScoringStrategy()

# Predefined weight configurations
_WEIGHTS_BALANCED_BERT = ConfidenceWeights(
    llm_score=0.35,          # LLM's judgment
    tfidf_similarity=0.10,   # Lexical similarity
    bert_similarity=0.25,    # Semantic similarity
    keyword_overlap=0.20,    # Direct matches
    category_maturity=0.05,  # Category age/size
    pass_number=0.05         # Refinement bonus
)

_WEIGHTS_BALANCED_NO_BERT = ConfidenceWeights(
    llm_score=0.40,
    tfidf_similarity=0.20,
    bert_similarity=0.0,     # Not used
    keyword_overlap=0.25,
    category_maturity=0.05,
    pass_number=0.10
)

_WEIGHTS_CONSERVATIVE_BERT = ConfidenceWeights(
    llm_score=0.30,          # Lower LLM weight
    tfidf_similarity=0.15,   # Higher lexical weight
    bert_similarity=0.20,
    keyword_overlap=0.20,
    category_maturity=0.10,  # Higher maturity weight (favor established)
    pass_number=0.05
)

_WEIGHTS_CONSERVATIVE_NO_BERT = ConfidenceWeights(
    llm_score=0.35,
    tfidf_similarity=0.25,
    bert_similarity=0.0,
    keyword_overlap=0.20,
    category_maturity=0.15,  # Higher maturity weight
    pass_number=0.05
)

_WEIGHTS_AGGRESSIVE_BERT = ConfidenceWeights(
    llm_score=0.40,          # Higher LLM weight (trust new suggestions)
    tfidf_similarity=0.05,   # Lower traditional weights
    bert_similarity=0.25,
    keyword_overlap=0.15,
    category_maturity=0.02,  # Lower maturity weight (don't favor established)
    pass_number=0.13         # Higher pass bonus (refinement matters)
)

_WEIGHTS_AGGRESSIVE_NO_BERT = ConfidenceWeights(
    llm_score=0.45,
    tfidf_similarity=0.15,
    bert_similarity=0.0,
    keyword_overlap=0.18,
    category_maturity=0.02,
    pass_number=0.20
)

_WEIGHTS_SEMANTIC_BERT = ConfidenceWeights(
    llm_score=0.25,          # Lower LLM weight
    tfidf_similarity=0.05,   # Lower lexical weight
    bert_similarity=0.45,    # Heavy BERT emphasis
    keyword_overlap=0.15,
    category_maturity=0.05,
    pass_number=0.05
)

# Preset configurations
CONFIG_BALANCED_HYBRID = ClusterizerConfig(
    retrieval_mode=RETRIEVAL_MODE.HYBRID,
    confidence_weights=_WEIGHTS_BALANCED_BERT,
    scoring_strategy=SCORING_STRATEGY.BALANCED,
    new_category_threshold=0.6,
    new_category_bonus=1.1
)

CONFIG_BALANCED_TFIDF = ClusterizerConfig(
    retrieval_mode=RETRIEVAL_MODE.TFIDF,
    confidence_weights=_WEIGHTS_BALANCED_NO_BERT,
    scoring_strategy=SCORING_STRATEGY.BALANCED,
    new_category_threshold=0.6,
    new_category_bonus=1.1
)

CONFIG_CONSERVATIVE_HYBRID = ClusterizerConfig(
    retrieval_mode=RETRIEVAL_MODE.HYBRID,
    confidence_weights=_WEIGHTS_CONSERVATIVE_BERT,
    scoring_strategy=SCORING_STRATEGY.CONSERVATIVE,
    new_category_threshold=0.70,  # Higher threshold = harder to create new
    new_category_bonus=1.05       # Lower bonus
)

CONFIG_CONSERVATIVE_TFIDF = ClusterizerConfig(
    retrieval_mode=RETRIEVAL_MODE.TFIDF,
    confidence_weights=_WEIGHTS_CONSERVATIVE_NO_BERT,
    scoring_strategy=SCORING_STRATEGY.CONSERVATIVE,
    new_category_threshold=0.70,
    new_category_bonus=1.05
)

CONFIG_AGGRESSIVE_HYBRID = ClusterizerConfig(
    retrieval_mode=RETRIEVAL_MODE.HYBRID,
    confidence_weights=_WEIGHTS_AGGRESSIVE_BERT,
    scoring_strategy=SCORING_STRATEGY.AGGRESSIVE,
    new_category_threshold=0.50,  # Lower threshold = easier to create new
    new_category_bonus=1.20       # Higher bonus
)

CONFIG_AGGRESSIVE_TFIDF = ClusterizerConfig(
    retrieval_mode=RETRIEVAL_MODE.TFIDF,
    confidence_weights=_WEIGHTS_AGGRESSIVE_NO_BERT,
    scoring_strategy=SCORING_STRATEGY.AGGRESSIVE,
    new_category_threshold=0.50,
    new_category_bonus=1.20
)

CONFIG_SEMANTIC_BERT = ClusterizerConfig(
    retrieval_mode=RETRIEVAL_MODE.BERT,
    confidence_weights=_WEIGHTS_SEMANTIC_BERT,
    scoring_strategy=SCORING_STRATEGY.SEMANTIC,
    new_category_threshold=0.65,
    new_category_bonus=1.1
)

# Default configuration (recommended)
DEFAULT_CONFIG = CONFIG_BALANCED_HYBRID

# Backward compatibility constants
CONFIDENCE_WEIGHTS_BERT = _WEIGHTS_BALANCED_BERT
CONFIDENCE_WEIGHTS_NO_BERT = _WEIGHTS_BALANCED_NO_BERT

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
- Medium confidence (0.5-0.7): Good match, reasonable fit
- Low confidence (0.0-0.4): Poor match, weak connection

Given a text and candidate categories, return:
    new_category: Optional[Dict[str, Any]] = None
    reasoning: str = ""

# ------------------
# Agent definitions
# ------------------
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

# ------------------
# BERT Integration
# ------------------
class BERTEncoder:
    """Wrapper for BERT-based sentence embeddings."""
    
    _instance = None
    _model = None
    
    @classmethod
    def get_instance(cls, model_name: str = 'all-MiniLM-L6-v2'):
        """Singleton pattern for model loading."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
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

def calculate_enhanced_confidence(
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
        bert_model: str = 'all-MiniLM-L6-v2',
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

_rate_limiter = LLMRateLimiter(MAX_CONCURRENT_LLM_CALLS)

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
    
    # Call LLM
    try:
        async with _rate_limiter:
            logger.info("  [%d/%d] Analyzing: %.50s...", text_num, total_texts, text)
            result = await asyncio.wait_for(
                multi_pass_analyzer.run(prompt),
                timeout=LLM_TIMEOUT_SECONDS
            )
            analysis = result.output
            
            if analysis.confidence_scores:
                logger.debug("  LLM confidence scores: %s", 
                           {k: f"{v:.2f}" for k, v in analysis.confidence_scores.items()})
            else:
                logger.warning("  LLM did not provide confidence scores!")
            
            logger.info("  [%d/%d] ✓ Complete", text_num, total_texts)
            
            # Process with BERT-enhanced confidence
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

TASK:
1. Evaluate how well this text fits each candidate category
2. Provide SPECIFIC confidence scores (0.0-1.0) - NOT all 0.5!
3. Either select best_category_id OR create new category if all scores < 0.6"""

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
    confidence = calculate_enhanced_confidence(
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
    confidence = calculate_enhanced_confidence(
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
    proc_state.increment_loop()
    
    if proc_state.should_stop(MAX_WATCHDOG):
        logger.error("Watchdog exceeded; forcing completion")
        proc_state.processing_complete = True
        proc_state.consolidation_complete = True
        update_state_from_processing(new_state, proc_state)
        new_state['decision'] = 'consolidate'
        return new_state
    
    texts = state.get('texts', []) or []
    batch_size = int(state.get('batch_size', DEFAULT_BATCH_SIZE))
    max_passes = int(state.get('max_passes', 2))
    
    if not texts:
        return _finalize_processing(new_state, proc_state)
    
    if proc_state.current_index >= len(texts):
        if proc_state.current_pass < max_passes:
            proc_state.start_next_pass()
            batch = texts[:batch_size]
            new_state['current_batch'] = batch
            new_state['decision'] = 'analyze'
        else:
            return _finalize_processing(new_state, proc_state)
    else:
        end_idx = min(proc_state.current_index + batch_size, len(texts))
        batch = texts[proc_state.current_index:end_idx]
        new_state['current_batch'] = batch
        new_state['decision'] = 'analyze'
    
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
    if not isinstance(config, ClusterizerConfig):
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
                        # Update assignment to use existing category
                        assignment['category_id'] = existing_cat.id
                        assignment_mgr.update_assignment(assignment)
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
            return [TextChunk(
                chunk_id=f"{text_id}_chunk_0",
                text=text,
                start_pos=0,
                end_pos=len(text),
                parent_text_id=text_id,
                chunk_index=0,
                total_chunks=1
            )]
        
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
                chunks.append(TextChunk(
                    chunk_id=f"{text_id}_chunk_{chunk_index}",
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=start_pos + len(chunk_text),
                    parent_text_id=text_id,
                    chunk_index=chunk_index,
                    total_chunks=-1  # Will update later
                ))
                
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
            chunks.append(TextChunk(
                chunk_id=f"{text_id}_chunk_{chunk_index}",
                text=chunk_text,
                start_pos=start_pos,
                end_pos=start_pos + len(chunk_text),
                parent_text_id=text_id,
                chunk_index=chunk_index,
                total_chunks=-1
            ))
        
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
# Enhanced Consolidation with Multi-Signal Analysis
# ------------------
class SmartConsolidator:
    """
    Multi-signal category consolidation with semantic understanding.
    
    Improvements over basic TF-IDF:
    - BERT semantic similarity (catches "Finance Markets" vs "Finance Economics")
    - Keyword overlap analysis
    - Name similarity detection
    - Weighted signal combination
    - Aggressive vs conservative modes
    """
    
    # Adjusted thresholds for better merging
    MERGE_THRESHOLDS = {
        'tfidf_similarity': 0.45,      # Lower from 0.6 - catches similar concepts
        'bert_similarity': 0.70,       # BERT is reliable for semantic merging
        'keyword_overlap': 0.40,       # 40% overlap = strong signal
        'name_similarity': 0.60,       # Similar names = likely duplicates
        'combined_threshold': 0.55     # Lower from 0.7 - more aggressive
    }
    
    def __init__(self, bert_encoder: Optional[BERTEncoder] = None):
        self.bert_encoder = bert_encoder
    
    def find_merge_candidates(
        self, 
        categories: List[Category],
        aggressive: bool = True
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """
        Find category pairs for merging with multi-signal analysis.
        
        Args:
            categories: List of categories to analyze
            aggressive: If True, use more lenient thresholds (for early passes)
        
        Returns:
            List of (cat1_id, cat2_id, overall_score, signal_breakdown)
        """
        if len(categories) <= 1:
            return []
        
        candidates = []
        
        # Track all scores for debugging
        all_scores = []
        
        # Auto-approval thresholds
        bert_auto_approve = 0.70  # Lower from 0.75 - catches most semantic duplicates
        combined_auto_approve = 0.65 if aggressive else 0.70
        
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                
                # Multi-signal analysis
                signals = self._analyze_similarity(cat1, cat2)
                
                # Combined score (weighted)
                overall = self._compute_merge_score(signals, aggressive)
                
                # Store for debugging
                all_scores.append((cat1.name, cat2.name, overall, signals))
                
                # AUTO-APPROVAL: High BERT similarity overrides combined score
                bert_sim = signals.get('bert_similarity', 0.0)
                if bert_sim >= bert_auto_approve:
                    candidates.append((cat1.id, cat2.id, overall, signals))
                    continue  # Skip threshold check
                
                # Standard threshold check
                threshold = self.MERGE_THRESHOLDS['combined_threshold']
                if aggressive:
                    threshold *= 0.85  # 15% more lenient
                
                if overall >= threshold or overall >= combined_auto_approve:
                    candidates.append((cat1.id, cat2.id, overall, signals))
        
        # Log top scores for debugging (even if below threshold)
        logger.info(f"=== Analyzed {len(all_scores)} category pairs ===")
        all_scores.sort(key=lambda x: x[2], reverse=True)
        for name1, name2, score, sigs in all_scores[:3]:  # Show top 3
            logger.info(f"  {name1} <-> {name2}: {score:.3f}")
            logger.info(f"    Signals: tfidf={sigs.get('tfidf_similarity', 0):.3f}, bert={sigs.get('bert_similarity', 0):.3f}, keywords={sigs.get('keyword_overlap', 0):.3f}, name={sigs.get('name_similarity', 0):.3f}")

        
        # Sort by confidence (merge most similar first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates
    
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
    
    # Initialize SmartConsolidator with BERT if available
    bert_encoder = None
    if config.retrieval_mode in [RETRIEVAL_MODE.BERT, RETRIEVAL_MODE.HYBRID]:
        try:
            bert_encoder = BERTEncoder.get_instance()
        except Exception as e:
            logger.warning(f"BERT not available for consolidation: {e}")
    
    consolidator = SmartConsolidator(bert_encoder=bert_encoder)
    
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

# Legacy functions removed - replaced by SmartConsolidator
# _find_merge_candidates() -> SmartConsolidator.find_merge_candidates()
# _call_consolidation_llm() -> _build_merge_result_from_candidates() with auto-approval

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
    """Route to next node."""
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
    
    return 'load_next'

def build_clusterizer_graph():
    """Build the processing graph."""
    workflow = StateGraph(dict)
    workflow.add_node('load_next', load_next_batch)
    workflow.add_node('analyze', analyze_batch_parallel)
    workflow.add_node('consolidate', consolidate_categories)
    workflow.set_entry_point('load_next')
    
    workflow.add_conditional_edges(
        'load_next',
        route_decision,
        {
            'analyze': 'analyze',
            'consolidate': 'consolidate',
            'load_next': 'load_next',
            END: END
        }
    )
    
    workflow.add_edge('analyze', 'load_next')
    workflow.add_edge('consolidate', END)
    
    return workflow.compile(checkpointer=MemorySaver())

# ------------------
# Public API
# ------------------
async def clusterize_texts(
    texts: List[str],
    max_passes: int = 2,
    prefilter_k: int = DEFAULT_PREFILTER_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent: int = MAX_CONCURRENT_LLM_CALLS,
    retrieval_mode: Optional[str] = None,
    bert_model: str = 'all-MiniLM-L6-v2',
    config: Optional[ClusterizerConfig] = None
) -> Dict[str, Any]:
    """
    Clusterize texts with BERT-enhanced semantic understanding.
    
    Args:
        texts: List of text strings to categorize
        max_passes: Number of passes over the data
        prefilter_k: Number of candidate categories to consider
        batch_size: Texts per batch (processed in parallel)
        max_concurrent: Maximum concurrent LLM calls
        retrieval_mode: DEPRECATED - Use `config` instead. 'tfidf', 'bert', or 'hybrid'
        bert_model: Sentence-transformer model name
        config: ClusterizerConfig instance (recommended). If not provided, uses DEFAULT_CONFIG
                or builds from retrieval_mode for backward compatibility.
    
    Returns:
        Dict with 'categories', 'assignments', and 'metadata'
    
    Examples:
        # Recommended: Use preset configs
        >>> res = await clusterize_texts(texts, config=CONFIG_CONSERVATIVE_HYBRID)
        
        # Or create custom config
        >>> custom = ClusterizerConfig(
        ...     retrieval_mode=RETRIEVAL_MODE.HYBRID,
        ...     confidence_weights=_WEIGHTS_BALANCED_BERT,
        ...     scoring_strategy=SCORING_STRATEGY.BALANCED
        ... )
        >>> res = await clusterize_texts(texts, config=custom)
        
        # Backward compatible (deprecated)
        >>> res = await clusterize_texts(texts, retrieval_mode=RETRIEVAL_MODE.HYBRID)
    """
    # Handle config resolution
    if config is None:
        if retrieval_mode is not None:
            # Backward compatibility: build config from retrieval_mode
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
            # Use default
            config = DEFAULT_CONFIG
    
    logger.info(f"Using configuration: {config.get_description()}")
    global _rate_limiter
    _rate_limiter = LLMRateLimiter(max_concurrent)
    
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
        'config': config,  # Store full config
        'bert_model': bert_model,
        '_loop_counter': 0
    }
    
    graph_config = {
        'configurable': {'thread_id': 'clusterizer_1'},
        'recursion_limit': 100
    }
    
    logger.info("Starting BERT-enhanced clusterization")
    logger.info("  Texts: %d | Batch: %d | Mode: %s | Strategy: %s", 
                len(texts), batch_size, config.retrieval_mode, config.scoring_strategy)
    
    step_count = 0
    last_state = None
    
    print(f"\n{'='*60}")
    print(f"BERT-ENHANCED CLUSTERIZATION")
    print(f"{'='*60}")
    print(f"Texts: {len(texts)}")
    print(f"Config: {config.get_description()}")
    print(f"Batch size: {batch_size} (parallel)")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Max passes: {max_passes}")
    print(f"{'='*60}\n")
    
    async for event in graph.astream(initial_state, graph_config):
        step_count += 1
        
        node_name = list(event.keys())[0] if event else "unknown"
        print(f"Step {step_count:2d}: {node_name:20s}", end="")
        
        if step_count > 100:
            logger.warning("Step limit exceeded")
            print(" [STOPPED]")
            break
        
        try:
            payload = next(iter(event.values()))
            if isinstance(payload, dict):
                if 'current_index' in payload:
                    idx = payload.get('current_index', 0)
                    total = len(payload.get('texts', []))
                    pass_num = payload.get('current_pass', 1)
                    cat_count = len(payload.get('categories', []))
                    print(f" | Pass {pass_num} | Text {idx}/{total} | Categories: {cat_count}")
                else:
                    print()
                
                if any(k in payload for k in ('categories', 'consolidation_complete', 'assignments')):
                    last_state = payload
        except Exception as e:
            print(f" [Error: {e}]")
    
    print(f"\n{'='*60}\n")
    
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
            'retrieval_mode': config.retrieval_mode,  # Backward compatibility
            'bert_enabled': config.retrieval_mode in [RETRIEVAL_MODE.BERT, RETRIEVAL_MODE.HYBRID],
            'confidence_stats': {
                'average': round(avg_confidence, 3),
                'min': round(min_confidence, 3),
                'max': round(max_confidence, 3),
                'high_confidence_count': high_conf,
                'medium_confidence_count': med_conf,
                'low_confidence_count': low_conf
            }
        }
    }

async def clusterize_texts_with_chunking(
    texts: List[str],
    large_text_threshold: int = 800,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    multi_topic_threshold: float = 0.3,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced clusterization that automatically handles large texts via chunking.
    
    This is a wrapper around clusterize_texts() that:
    1. Identifies large texts (>= large_text_threshold tokens)
    2. Chunks them using SemanticChunker
    3. Processes all units (small texts + chunks) through standard pipeline
    4. Aggregates chunk results back to original texts
    5. Detects multi-topic documents
    
    Args:
        texts: Input texts (can be any size)
        large_text_threshold: Texts with >= this many tokens get chunked (default: 800)
        chunk_size: Target size for each chunk in tokens (default: 500)
        chunk_overlap: Overlap between chunks in tokens (default: 100)
        multi_topic_threshold: Secondary category threshold for multi-topic detection (default: 0.3)
        **kwargs: All other arguments passed to clusterize_texts()
    
    Returns:
        Dict with:
        - 'categories': List of Category objects
        - 'assignments': List of CategoryAssignment objects (one per original text)
        - 'metadata': Enhanced metadata with chunking stats
        - 'multi_topic_texts': List of texts identified as multi-topic (if any)
    
    Examples:
        # Handle mix of small and large texts
        >>> texts = ["Short text", "Very long document..." * 100]
        >>> result = await clusterize_texts_with_chunking(
        ...     texts,
        ...     large_text_threshold=800,
        ...     config=CONFIG_BALANCED_HYBRID
        ... )
        >>> 
        >>> # Check for multi-topic documents
        >>> multi_topic = result.get('multi_topic_texts', [])
        >>> for mt in multi_topic:
        ...     print(f"{mt['text'][:50]}... has {len(mt['categories'])} topics")
    """
    logger.info(f"Starting chunked clusterization for {len(texts)} texts")
    
    # Step 1: Identify and chunk large texts
    chunker = SemanticChunker(chunk_size=chunk_size, overlap=chunk_overlap)
    aggregator = ChunkAggregator(multi_topic_threshold=multi_topic_threshold)
    
    processed_units = []
    text_to_chunks = {}  # Maps text_index -> List[chunk_indices in processed_units]
    chunk_to_text = {}   # Maps processed_unit_index -> original_text_index
    
    chunked_count = 0
    total_chunks = 0
    
    for text_idx, text in enumerate(texts):
        if chunker.should_chunk(text, large_text_threshold):
            # Large text: chunk it
            chunks = chunker.chunk_text(text, f"text_{text_idx}")
            chunked_count += 1
            total_chunks += len(chunks)
            
            chunk_indices = []
            for chunk in chunks:
                unit_idx = len(processed_units)
                processed_units.append(chunk.text)
                chunk_to_text[unit_idx] = text_idx
                chunk_indices.append(unit_idx)
            
            text_to_chunks[text_idx] = chunk_indices
            logger.info(f"  Text {text_idx}: Chunked into {len(chunks)} parts")
        else:
            # Small text: keep as-is
            unit_idx = len(processed_units)
            processed_units.append(text)
            chunk_to_text[unit_idx] = text_idx
            text_to_chunks[text_idx] = [unit_idx]
    
    logger.info(f"Preprocessing complete: {len(texts)} texts → {len(processed_units)} units")
    logger.info(f"  Chunked: {chunked_count} texts into {total_chunks} chunks")
    
    # Step 2: Process all units through standard clusterization
    logger.info("Processing units through clusterizer...")
    result = await clusterize_texts(processed_units, **kwargs)
    
    # Step 3: Aggregate results back to original texts
    logger.info("Aggregating results...")
    
    categories = result['categories']
    unit_assignments = result['assignments']
    
    # Build assignment map by unit index
    unit_assignment_map = {}
    for assignment in unit_assignments:
        # Find which unit this assignment corresponds to
        for unit_idx, unit_text in enumerate(processed_units):
            if assignment.text == unit_text:
                unit_assignment_map[unit_idx] = assignment
                break
    
    # Aggregate by original text
    final_assignments = []
    multi_topic_texts = []
    
    for text_idx, text in enumerate(texts):
        chunk_indices = text_to_chunks[text_idx]
        
        if len(chunk_indices) == 1:
            # Single unit (wasn't chunked)
            unit_idx = chunk_indices[0]
            assignment = unit_assignment_map.get(unit_idx)
            if assignment:
                final_assignments.append(assignment)
        else:
            # Multiple chunks: aggregate
            chunk_assignments = []
            for chunk_idx_in_units, unit_idx in enumerate(chunk_indices):
                assignment = unit_assignment_map.get(unit_idx)
                if assignment:
                    # Create ChunkAssignment
                    chunk_obj = TextChunk(
                        chunk_id=f"text_{text_idx}_chunk_{chunk_idx_in_units}",
                        text=assignment.text,
                        start_pos=0,
                        end_pos=len(assignment.text),
                        parent_text_id=f"text_{text_idx}",
                        chunk_index=chunk_idx_in_units,
                        total_chunks=len(chunk_indices)
                    )
                    chunk_assignment = ChunkAssignment(
                        chunk=chunk_obj,
                        category_id=assignment.category_id,
                        confidence=assignment.confidence,
                        reasoning=assignment.reasoning
                    )
                    chunk_assignments.append(chunk_assignment)
            
            # Aggregate chunks
            if chunk_assignments:
                agg_result = aggregator.aggregate(text, chunk_assignments)
                
                # Convert to CategoryAssignment
                final_assignment = CategoryAssignment(
                    text=text,
                    category_id=agg_result.primary_category_id,
                    confidence=agg_result.confidence,
                    reasoning=agg_result.reasoning
                )
                final_assignments.append(final_assignment)
                
                # Track multi-topic texts
                if agg_result.is_multi_topic:
                    multi_topic_texts.append({
                        'text_index': text_idx,
                        'text': text[:200] + "..." if len(text) > 200 else text,
                        'primary_category': agg_result.primary_category_id,
                        'categories': [agg_result.primary_category_id] + 
                                     [cat_id for cat_id, _ in agg_result.secondary_categories],
                        'confidence': agg_result.confidence
                    })
                    logger.info(f"  Multi-topic detected: Text {text_idx} spans {len(agg_result.secondary_categories) + 1} categories")
    
    logger.info(f"Aggregation complete: {len(final_assignments)} final assignments")
    if multi_topic_texts:
        logger.info(f"  Found {len(multi_topic_texts)} multi-topic documents")
    
    # Build enhanced result
    enhanced_metadata = result['metadata'].copy()
    enhanced_metadata.update({
        'chunking_enabled': True,
        'large_text_threshold': large_text_threshold,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'texts_chunked': chunked_count,
        'total_chunks_created': total_chunks,
        'processing_units': len(processed_units),
        'multi_topic_count': len(multi_topic_texts)
    })
    
    return {
        'categories': categories,
        'assignments': final_assignments,
        'metadata': enhanced_metadata,
        'multi_topic_texts': multi_topic_texts
    }

# ------------------
# Example usage
# ------------------
if __name__ == '__main__':
    sample_texts = [
        "The stock market reached new highs today with technology shares leading gains.",
        "Scientists discovered a new species of deep-sea fish near the Mariana Trench.",
        "The new iPhone features improved battery life and camera capabilities.",
        "Federal Reserve announces interest rate decision affecting global markets.",
        "Researchers found evidence of water on Jupiter's moon Europa.",
        "Samsung launches foldable smartphone with innovative display technology.",
        "Astronomers detect unusual radio signals from distant galaxy.",
        "Cryptocurrency prices surge following institutional adoption news.",
        "Marine biologists study coral reef ecosystems in the Pacific Ocean.",
        "Apple announces new MacBook with revolutionary M-series chip.",
        "Central banks coordinate response to inflation concerns.",
        "NASA plans mission to explore Saturn's moon Titan.",
    ]
    
    async def main():
        import time
        
        logging.getLogger("agentic_clusterizer").setLevel(logging.INFO)
        
        logger.info("Starting BERT-enhanced agentic clusterizer...")
        start_time = time.time()
        
        # NEW: Use preset configs (recommended)
        # Options: 
        #   - CONFIG_BALANCED_HYBRID
        #   - CONFIG_CONSERVATIVE_HYBRID
        #   - CONFIG_AGGRESSIVE_HYBRID
        #   - CONFIG_SEMANTIC_BERT
        #   - CONFIG_BALANCED_TFIDF, etc.
        res = await clusterize_texts(
            sample_texts,
            max_passes=2,
            prefilter_k=3,
            batch_size=6,
            max_concurrent=5,
            config=CONFIG_SEMANTIC_BERT  # 🎯 Use config for better control
        )
        
        elapsed = time.time() - start_time
        
        print("=" * 60)
        print("CATEGORIES")
        print("=" * 60)
        for c in res['categories']:
            print(f"- {c.name} ({c.id})")
            print(f"  Count: {c.text_count}")
            print(f"  Keywords: {', '.join(c.keywords[:5])}")
            print(f"  Description: {c.description[:80]}...")
            print()
        
        print("=" * 60)
        print("ASSIGNMENTS (BERT-Enhanced Confidence)")
        print("=" * 60)
        
        sorted_assignments = sorted(res['assignments'], key=lambda a: a.confidence, reverse=True)
        
        for a in sorted_assignments:
            if a.confidence >= 0.75:
                indicator = "✓✓"
            elif a.confidence >= 0.5:
                indicator = "✓ "
            else:
                indicator = "? "
            
            print(f"{indicator} {a.text[:58]}...")
            print(f"   → {a.category_id}")
            print(f"   Confidence: {a.confidence:.2f} | {a.reasoning}")
            print()
        
        print("=" * 60)
        print("METADATA & PERFORMANCE")
        print("=" * 60)
        metadata = res['metadata']
        
        print(f"total_passes: {metadata['total_passes']}")
        print(f"total_texts: {metadata['total_texts']}")
        print(f"total_categories: {metadata['total_categories']}")
        print(f"retrieval_mode: {metadata['retrieval_mode']}")
        print(f"bert_enabled: {metadata['bert_enabled']}")
        print(f"elapsed_time: {elapsed:.2f}s")
        print(f"texts_per_second: {len(sample_texts)/elapsed:.2f}")
        
        conf_stats = metadata.get('confidence_stats', {})
        if conf_stats:
            print(f"\nConfidence Statistics:")
            print(f"  average: {conf_stats.get('average', 0):.3f}")
            print(f"  min: {conf_stats.get('min', 0):.3f}")
            print(f"  max: {conf_stats.get('max', 0):.3f}")
            print(f"  high (≥0.75): {conf_stats.get('high_confidence_count', 0)}")
            print(f"  medium (0.5-0.75): {conf_stats.get('medium_confidence_count', 0)}")
            print(f"  low (<0.5): {conf_stats.get('low_confidence_count', 0)}")
    
    asyncio.run(main())