"""
Intelligent Configuration Recommender for Text Clustering

Analyzes dataset characteristics and recommends optimal parameters:
- Dataset size (number of texts)
- Text complexity (length, vocabulary, structure)
- Big text detection and handling
- Memory and performance constraints
- Cost estimation (LLM API calls)

Key Innovation: Multi-dimensional analysis instead of size-only heuristics
"""

import statistics
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

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

@dataclass
class TextStatistics:
    """Statistical analysis of text dataset."""
    total_texts: int
    total_tokens: int
    
    # Token distribution
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    median_tokens: float
    std_tokens: float
    
    # Big text detection
    big_text_count: int
    big_text_percentage: float
    big_text_threshold: int
    
    # Complexity indicators
    vocabulary_richness: float  # Unique tokens / total tokens
    avg_word_length: float
    complexity_score: float  # 0.0-1.0
    
    # Memory estimation
    estimated_memory_mb: float


@dataclass
class RecommendedConfig:
    """Complete recommended configuration."""
    # Core parameters
    max_texts_per_run: int
    batch_size: int
    max_passes: int
    token_threshold: int  # For big text detection
    max_parallel_merges: int
    max_concurrent: int  # LLM calls
    
    # Strategy
    config_preset: str  # e.g., 'CONFIG_BALANCED_HYBRID'
    use_tree_merge: bool
    
    # Predictions
    expected_workflows: int
    expected_batches_per_workflow: int
    expected_total_batches: int
    expected_merge_levels: int
    
    # Cost & time estimates
    estimated_llm_calls: int
    estimated_duration_minutes: Tuple[float, float]  # (min, max)
    estimated_cost_usd: Tuple[float, float]  # (min, max)
    
    # Reasoning
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def print_summary(self):
        """Print human-readable summary."""
        print(f"\n{'='*70}")
        print(f"🎯 RECOMMENDED CONFIGURATION")
        print(f"{'='*70}\n")
        
        print(f"📊 Core Parameters:")
        print(f"  max_texts_per_run:    {self.max_texts_per_run:>6}")
        print(f"  batch_size:           {self.batch_size:>6}")
        print(f"  max_passes:           {self.max_passes:>6}")
        print(f"  token_threshold:      {self.token_threshold:>6}")
        print(f"  max_parallel_merges:  {self.max_parallel_merges:>6}")
        print(f"  max_concurrent:       {self.max_concurrent:>6}")
        
        print(f"\n⚙️  Strategy:")
        print(f"  Config preset:        {self.config_preset}")
        print(f"  Tree merge:           {'Yes' if self.use_tree_merge else 'No'}")
        
        print(f"\n📈 Expected Execution:")
        print(f"  Workflows:            {self.expected_workflows}")
        print(f"  Batches per workflow: {self.expected_batches_per_workflow}")
        print(f"  Total batches:        {self.expected_total_batches}")
        if self.use_tree_merge:
            print(f"  Merge levels:         {self.expected_merge_levels}")
        
        print(f"\n💰 Estimates:")
        print(f"  LLM calls:            ~{self.estimated_llm_calls:,}")
        print(f"  Duration:             {self.estimated_duration_minutes[0]:.1f}-{self.estimated_duration_minutes[1]:.1f} minutes")
        print(f"  Cost (gpt-4o-mini):   ${self.estimated_cost_usd[0]:.2f}-${self.estimated_cost_usd[1]:.2f}")
        
        if self.reasoning:
            print(f"\n💡 Reasoning:")
            for reason in self.reasoning:
                print(f"  • {reason}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        print(f"\n{'='*70}\n")


class DatasetCategory(Enum):
    """Dataset size categories."""
    TINY = "tiny"           # < 50
    SMALL = "small"         # 50-500
    MEDIUM = "medium"       # 500-2000
    LARGE = "large"         # 2000-10000
    XLARGE = "xlarge"       # 10000-50000
    MASSIVE = "massive"     # > 50000


class ComplexityLevel(Enum):
    """Text complexity levels."""
    SIMPLE = "simple"       # Short, simple texts
    MODERATE = "moderate"   # Average complexity
    COMPLEX = "complex"     # Long, complex texts
    MIXED = "mixed"         # High variance

class Priority(Enum):
    """Processing priorities."""
    SPEED = "speed"
    QUALITY = "quality"
    COST = "cost"
    BALANCED = "balanced"

# ============================================================================
# TEXT ANALYZER
# ============================================================================

class TextAnalyzer:
    """Analyzes text dataset characteristics."""
    
    # Token estimation: ~0.75 tokens per word on average
    TOKENS_PER_WORD = 0.75
    
    # Complexity thresholds
    COMPLEXITY_THRESHOLDS = {
        'vocabulary_richness': 0.5,  # Below this = repetitive
        'avg_word_length': 5.0,      # Above this = complex
        'token_variance': 100.0      # High variance = mixed
    }
    
    def __init__(self, token_threshold: int = 200):
        """
        Args:
            token_threshold: Threshold for big text detection
        """
        self.token_threshold = token_threshold
    
    def analyze(self, texts: List[str]) -> TextStatistics:
        """Perform comprehensive analysis of text dataset."""
        if not texts:
            raise ValueError("Cannot analyze empty text list")
        
        logger.info(f"Analyzing {len(texts)} texts...")
        
        # Token counts
        token_counts = [self._estimate_tokens(text) for text in texts]
        total_tokens = sum(token_counts)
        
        # Big text detection
        big_text_count = sum(1 for count in token_counts if count > self.token_threshold)
        big_text_percentage = (big_text_count / len(texts)) * 100
        
        # Complexity analysis
        vocabulary_richness = self._calculate_vocabulary_richness(texts)
        avg_word_length = self._calculate_avg_word_length(texts)
        complexity_score = self._calculate_complexity_score(
            token_counts, vocabulary_richness, avg_word_length
        )
        
        # Memory estimation (rough heuristic)
        estimated_memory_mb = self._estimate_memory(texts, token_counts)
        
        stats = TextStatistics(
            total_texts=len(texts),
            total_tokens=total_tokens,
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
            mean_tokens=statistics.mean(token_counts),
            median_tokens=statistics.median(token_counts),
            std_tokens=statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0,
            big_text_count=big_text_count,
            big_text_percentage=big_text_percentage,
            big_text_threshold=self.token_threshold,
            vocabulary_richness=vocabulary_richness,
            avg_word_length=avg_word_length,
            complexity_score=complexity_score,
            estimated_memory_mb=estimated_memory_mb
        )
        
        self._log_statistics(stats)
        
        return stats
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        words = text.split()
        return int(len(words) * self.TOKENS_PER_WORD)
    
    def _calculate_vocabulary_richness(self, texts: List[str]) -> float:
        """Calculate vocabulary richness (unique words / total words)."""
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        if not all_words:
            return 0.0
        
        unique_words = set(all_words)
        return len(unique_words) / len(all_words)
    
    def _calculate_avg_word_length(self, texts: List[str]) -> float:
        """Calculate average word length across all texts."""
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        
        if not all_words:
            return 0.0
        
        return sum(len(word) for word in all_words) / len(all_words)
    
    def _calculate_complexity_score(
        self,
        token_counts: List[int],
        vocab_richness: float,
        avg_word_length: float
    ) -> float:
        """
        Calculate overall complexity score (0.0-1.0).
        
        Factors:
        - Token count variance (mixed lengths = more complex)
        - Vocabulary richness (diverse = more complex)
        - Average word length (longer = more complex)
        """
        # Factor 1: Token variance (normalized)
        if len(token_counts) > 1:
            std_tokens = statistics.stdev(token_counts)
            mean_tokens = statistics.mean(token_counts)
            variance_score = min(1.0, std_tokens / (mean_tokens + 1))
        else:
            variance_score = 0.0
        
        # Factor 2: Vocabulary richness
        vocab_score = vocab_richness
        
        # Factor 3: Word length (normalized)
        word_length_score = min(1.0, avg_word_length / 10.0)
        
        # Weighted combination
        complexity = (
            0.4 * variance_score +
            0.3 * vocab_score +
            0.3 * word_length_score
        )
        
        return complexity
    
    def _estimate_memory(self, texts: List[str], token_counts: List[int]) -> float:
        """Estimate peak memory usage in MB."""
        # Rough heuristic:
        # - Raw text: ~1 byte per character
        # - Embeddings (if BERT): ~4 bytes per token per dimension (384 dims typical)
        # - Category objects: ~1KB per category (assuming ~100 categories)
        # - Assignment objects: ~500 bytes per assignment
        
        raw_text_bytes = sum(len(text) for text in texts)
        embeddings_bytes = sum(token_counts) * 4 * 384  # BERT embeddings
        categories_bytes = 100 * 1024  # Assume 100 categories
        assignments_bytes = len(texts) * 500
        
        total_bytes = raw_text_bytes + embeddings_bytes + categories_bytes + assignments_bytes
        
        # Add 50% overhead for Python objects, intermediate data
        total_mb = (total_bytes * 1.5) / (1024 * 1024)
        
        return total_mb
    
    def _log_statistics(self, stats: TextStatistics):
        """Log statistics summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Total texts: {stats.total_texts:,}")
        logger.info(f"Total tokens: {stats.total_tokens:,}")
        logger.info(f"Token range: {stats.min_tokens}-{stats.max_tokens}")
        logger.info(f"Token mean/median: {stats.mean_tokens:.1f} / {stats.median_tokens:.1f}")
        logger.info(f"Token std dev: {stats.std_tokens:.1f}")
        logger.info(f"Big texts (>{stats.big_text_threshold} tokens): {stats.big_text_count} ({stats.big_text_percentage:.1f}%)")
        logger.info(f"Vocabulary richness: {stats.vocabulary_richness:.3f}")
        logger.info(f"Avg word length: {stats.avg_word_length:.2f}")
        logger.info(f"Complexity score: {stats.complexity_score:.3f}")
        logger.info(f"Est. memory: {stats.estimated_memory_mb:.1f} MB")
        logger.info(f"{'='*60}\n")


# ============================================================================
# CONFIGURATION RECOMMENDER
# ============================================================================

class ConfigurationRecommender:
    """
    Intelligent configuration recommender.
    
    Uses multi-dimensional analysis to recommend optimal parameters.
    """
    
    # LLM cost estimates (per 1K tokens) - gpt-4o-mini
    LLM_COST_INPUT = 0.000150   # $0.15 per 1M tokens
    LLM_COST_OUTPUT = 0.000600  # $0.60 per 1M tokens
    
    # Performance estimates
    SECONDS_PER_TEXT = 0.5  # Average time per text (optimistic)
    SECONDS_PER_TEXT_SLOW = 2.0  # Pessimistic estimate
    
    def __init__(self):
        self.analyzer = TextAnalyzer()
    
    def recommend(
        self,
        texts: List[str],
        priority: Priority = Priority.BALANCED,
        memory_limit_mb: Optional[int] = None
    ) -> RecommendedConfig:
        """
        Recommend optimal configuration based on dataset analysis.
        
        Args:
            texts: List of texts to analyze
            priority: Optimization priority
            memory_limit_mb: Optional memory limit (auto-adjusts params)
        
        Returns:
            RecommendedConfig with complete recommendations
        """
        # Step 1: Analyze dataset
        stats = self.analyzer.analyze(texts)
        
        # Step 2: Categorize dataset
        dataset_category = self._categorize_dataset(stats)
        complexity_level = self._categorize_complexity(stats)
        
        # Step 3: Generate base recommendations
        config = self._generate_base_config(
            stats, dataset_category, complexity_level, priority
        )
        
        # Step 4: Adjust for constraints
        if memory_limit_mb:
            config = self._adjust_for_memory_limit(config, stats, memory_limit_mb)
        
        # Step 5: Add predictions and estimates
        config = self._add_predictions(config, stats)
        config = self._add_cost_estimates(config, stats)
        
        # Step 6: Generate reasoning
        config = self._add_reasoning(config, stats, dataset_category, complexity_level, priority)
        
        return config
    
    def _categorize_dataset(self, stats: TextStatistics) -> DatasetCategory:
        """Categorize dataset by size."""
        n = stats.total_texts
        
        if n < 50:
            return DatasetCategory.TINY
        elif n < 500:
            return DatasetCategory.SMALL
        elif n < 2000:
            return DatasetCategory.MEDIUM
        elif n < 10000:
            return DatasetCategory.LARGE
        elif n < 50000:
            return DatasetCategory.XLARGE
        else:
            return DatasetCategory.MASSIVE
    
    def _categorize_complexity(self, stats: TextStatistics) -> ComplexityLevel:
        """Categorize text complexity."""
        # High variance = mixed complexity
        if stats.std_tokens > 100:
            return ComplexityLevel.MIXED
        
        # Simple: short, repetitive texts
        if stats.mean_tokens < 50 and stats.vocabulary_richness < 0.3:
            return ComplexityLevel.SIMPLE
        
        # Complex: long texts or high vocabulary richness
        if stats.mean_tokens > 200 or stats.vocabulary_richness > 0.6:
            return ComplexityLevel.COMPLEX
        
        return ComplexityLevel.MODERATE
    
    def _generate_base_config(
        self,
        stats: TextStatistics,
        dataset_category: DatasetCategory,
        complexity_level: ComplexityLevel,
        priority: Priority
    ) -> RecommendedConfig:
        """Generate base configuration recommendations."""
        
        # Base parameters by dataset size
        size_params = {
            DatasetCategory.TINY: {
                'max_texts_per_run': 1000,
                'batch_size': 10,
                'max_passes': 2,
                'max_parallel_merges': 2,
                'max_concurrent': 3
            },
            DatasetCategory.SMALL: {
                'max_texts_per_run': 1000,
                'batch_size': 20,
                'max_passes': 2,
                'max_parallel_merges': 4,
                'max_concurrent': 5
            },
            DatasetCategory.MEDIUM: {
                'max_texts_per_run': 400,
                'batch_size': 50,
                'max_passes': 2,
                'max_parallel_merges': 4,
                'max_concurrent': 5
            },
            DatasetCategory.LARGE: {
                'max_texts_per_run': 300,
                'batch_size': 75,
                'max_passes': 1,
                'max_parallel_merges': 6,
                'max_concurrent': 8
            },
            DatasetCategory.XLARGE: {
                'max_texts_per_run': 200,
                'batch_size': 100,
                'max_passes': 1,
                'max_parallel_merges': 8,
                'max_concurrent': 10
            },
            DatasetCategory.MASSIVE: {
                'max_texts_per_run': 150,
                'batch_size': 100,
                'max_passes': 1,
                'max_parallel_merges': 10,
                'max_concurrent': 10
            }
        }
        
        params = size_params[dataset_category].copy()
        
        # Adjust for complexity
        if complexity_level == ComplexityLevel.COMPLEX:
            params['batch_size'] = max(10, params['batch_size'] // 2)  # Smaller batches
            params['max_passes'] += 1  # More refinement
        elif complexity_level == ComplexityLevel.SIMPLE:
            params['batch_size'] = min(100, params['batch_size'] * 2)  # Larger batches
        
        # Adjust for priority
        if priority == Priority.SPEED:
            params['max_passes'] = 1  # Faster
            params['max_concurrent'] = min(15, params['max_concurrent'] + 5)  # More parallelism
        elif priority == Priority.QUALITY:
            params['max_passes'] = min(3, params['max_passes'] + 1)  # More refinement
            params['batch_size'] = max(10, params['batch_size'] // 2)  # Smaller batches
        elif priority == Priority.COST:
            params['max_passes'] = 1  # Fewer LLM calls
            params['max_texts_per_run'] = min(1000, params['max_texts_per_run'] * 2)  # Fewer workflows
        
        # Token threshold for big texts
        token_threshold = self._calculate_token_threshold(stats)
        
        # Config preset selection
        if complexity_level == ComplexityLevel.COMPLEX:
            config_preset = 'CONFIG_SEMANTIC_BERT'
        elif priority == Priority.QUALITY:
            config_preset = 'CONFIG_CONSERVATIVE_HYBRID'
        elif priority == Priority.SPEED or \
            dataset_category in [DatasetCategory.XLARGE, DatasetCategory.MASSIVE]:
            config_preset = 'CONFIG_AGGRESSIVE_HYBRID'
        else:
            config_preset = 'CONFIG_BALANCED_HYBRID'
        
        # Tree merge decision
        use_tree_merge = stats.total_texts > params['max_texts_per_run']
        
        return RecommendedConfig(
            max_texts_per_run=params['max_texts_per_run'],
            batch_size=params['batch_size'],
            max_passes=params['max_passes'],
            token_threshold=token_threshold,
            max_parallel_merges=params['max_parallel_merges'],
            max_concurrent=params['max_concurrent'],
            config_preset=config_preset,
            use_tree_merge=use_tree_merge,
            expected_workflows=0,  # Will be calculated
            expected_batches_per_workflow=0,
            expected_total_batches=0,
            expected_merge_levels=0,
            estimated_llm_calls=0,
            estimated_duration_minutes=(0, 0),
            estimated_cost_usd=(0, 0)
        )
    
    def _calculate_token_threshold(self, stats: TextStatistics) -> int:
        """Calculate optimal token threshold for big text detection."""
        # Use percentile-based approach
        # Set threshold at 90th percentile or 200, whichever is higher
        threshold = max(200, int(stats.mean_tokens + 2 * stats.std_tokens))
        
        # Cap at reasonable maximum
        threshold = min(threshold, 500)
        
        return threshold
    
    def _adjust_for_memory_limit(
        self,
        config: RecommendedConfig,
        stats: TextStatistics,
        memory_limit_mb: int
    ) -> RecommendedConfig:
        """Adjust configuration to fit within memory limit."""
        if stats.estimated_memory_mb <= memory_limit_mb:
            return config  # Already within limit
        
        # Strategy: Reduce batch size and workflow size
        reduction_factor = memory_limit_mb / stats.estimated_memory_mb
        
        config.batch_size = max(10, int(config.batch_size * reduction_factor))
        config.max_texts_per_run = max(50, int(config.max_texts_per_run * reduction_factor))
        
        config.warnings.append(
            f"Configuration adjusted to fit within {memory_limit_mb}MB memory limit. "
            f"This may increase processing time."
        )
        
        return config
    
    def _add_predictions(
        self,
        config: RecommendedConfig,
        stats: TextStatistics
    ) -> RecommendedConfig:
        """Add execution predictions."""
        if config.use_tree_merge:
            config.expected_workflows = (
                stats.total_texts + config.max_texts_per_run - 1
            ) // config.max_texts_per_run
            
            config.expected_batches_per_workflow = (
                config.max_texts_per_run + config.batch_size - 1
            ) // config.batch_size
            
            config.expected_total_batches = (
                config.expected_workflows * config.expected_batches_per_workflow
            )
            
            # Merge levels in binary tree
            import math
            config.expected_merge_levels = math.ceil(math.log2(config.expected_workflows))
        else:
            config.expected_workflows = 1
            config.expected_batches_per_workflow = (
                stats.total_texts + config.batch_size - 1
            ) // config.batch_size
            config.expected_total_batches = config.expected_batches_per_workflow
            config.expected_merge_levels = 0
        
        return config
    
    def _add_cost_estimates(
        self,
        config: RecommendedConfig,
        stats: TextStatistics
    ) -> RecommendedConfig:
        """Add cost and time estimates."""
        # LLM calls estimate
        # Each text gets analyzed once per pass
        base_analysis_calls = stats.total_texts * config.max_passes
        
        # Merges: Each merge analyzes category representatives
        if config.use_tree_merge:
            # Rough estimate: ~100 categories per workflow, ~500 tokens per category
            merge_calls = config.expected_workflows * 100
        else:
            merge_calls = 0
        
        # Consolidation: One pass at the end
        consolidation_calls = len([c for c in range(10)])  # Rough estimate
        
        config.estimated_llm_calls = base_analysis_calls + merge_calls + consolidation_calls
        
        # Cost estimate (gpt-4o-mini)
        # Assume average 150 input tokens, 50 output tokens per call
        input_tokens = config.estimated_llm_calls * 150
        output_tokens = config.estimated_llm_calls * 50
        
        cost_min = (
            (input_tokens / 1000) * self.LLM_COST_INPUT +
            (output_tokens / 1000) * self.LLM_COST_OUTPUT
        )
        cost_max = cost_min * 2.0  # 2x buffer for variability
        
        config.estimated_cost_usd = (cost_min, cost_max)
        
        # Time estimate
        time_min = stats.total_texts * self.SECONDS_PER_TEXT / 60
        time_max = stats.total_texts * self.SECONDS_PER_TEXT_SLOW / 60
        
        config.estimated_duration_minutes = (time_min, time_max)
        
        return config
    
    def _add_reasoning(
        self,
        config: RecommendedConfig,
        stats: TextStatistics,
        dataset_category: DatasetCategory,
        complexity_level: ComplexityLevel,
        priority: Priority
    ) -> RecommendedConfig:
        """Add human-readable reasoning."""
        config.reasoning = []
        
        # Dataset size reasoning
        config.reasoning.append(
            f"Dataset size: {stats.total_texts} texts → {dataset_category.value.upper()} category"
        )
        
        # Complexity reasoning
        config.reasoning.append(
            f"Text complexity: {complexity_level.value.upper()} "
            f"(avg {stats.mean_tokens:.0f} tokens, richness {stats.vocabulary_richness:.2f})"
        )
        
        # Big texts
        if stats.big_text_percentage > 10:
            config.reasoning.append(
                f"Big texts detected: {stats.big_text_count} texts "
                f"({stats.big_text_percentage:.1f}%) will be chunked at {config.token_threshold} tokens"
            )
        
        # Tree merge reasoning
        if config.use_tree_merge:
            config.reasoning.append(
                f"Tree merge enabled: {config.expected_workflows} workflows "
                f"→ {config.expected_merge_levels} merge levels"
            )
        else:
            config.reasoning.append("Single workflow mode: dataset small enough for direct processing")
        
        # Priority reasoning
        if priority != 'balanced':
            config.reasoning.append(f"Optimized for: {priority.upper()}")
        
        # Warnings
        if stats.estimated_memory_mb > 1000:
            config.warnings.append(
                f"High memory usage expected: ~{stats.estimated_memory_mb:.0f}MB. "
                "Consider processing on a machine with sufficient RAM."
            )
        
        if config.estimated_duration_minutes[1] > 60:
            config.warnings.append(
                f"Long processing time expected: up to {config.estimated_duration_minutes[1]:.0f} minutes. "
                "Consider running in background or using faster hardware."
            )
        
        if config.estimated_cost_usd[1] > 5.0:
            config.warnings.append(
                f"Significant LLM costs expected: ${config.estimated_cost_usd[0]:.2f}-${config.estimated_cost_usd[1]:.2f}. "
                "Review your API budget."
            )
        
        return config


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def recommend_config(
    texts: List[str],
    priority: Priority = Priority.BALANCED,
    memory_limit_mb: Optional[int] = None,
    print_summary: bool = True
) -> RecommendedConfig:
    """
    Convenience function to get configuration recommendations.
    
    Args:
        texts: List of texts to analyze
        priority: 'speed', 'quality', 'cost', or 'balanced'
        memory_limit_mb: Optional memory limit
        print_summary: Whether to print summary
    
    Returns:
        RecommendedConfig object
    
    Example:
        >>> texts = load_texts('cnae.txt')
        >>> config = recommend_config(texts, priority='quality')
        >>> # Use config in clustering
        >>> result = await clusterize_texts(
        ...     texts,
        ...     max_texts_per_run=config.max_texts_per_run,
        ...     batch_size=config.batch_size,
        ...     ...
        ... )
    """
    recommender = ConfigurationRecommender()
    config = recommender.recommend(texts, priority, memory_limit_mb)
    
    if print_summary:
        config.print_summary()
    
    return config


# ============================================================================
# TESTING
# ============================================================================

def test_recommender():
    """Test the recommender with various dataset sizes."""
    print("\n" + "="*70)
    print("TESTING CONFIGURATION RECOMMENDER")
    print("="*70 + "\n")
    
    test_cases = [
        ("Tiny dataset", ["short text"] * 30),
        ("Small dataset", ["medium length text " * 10] * 300),
        ("Large dataset with big texts", ["very long text " * 50] * 5000),
        ("Mixed complexity", ["short"] * 500 + ["long text " * 100] * 500)
    ]
    
    for name, texts in test_cases:
        print(f"\n{'='*70}")
        print(f"Test Case: {name} ({len(texts)} texts)")
        print(f"{'='*70}")
        
        config = recommend_config(texts, priority='balanced', print_summary=False)
        config.print_summary()
        
        input("Press Enter to continue...")


if __name__ == "__main__":
    # Example usage
    print("Configuration Recommender - Example Usage\n")
    
    # Simulate loading texts
    example_texts = ["This is a sample text about finance and markets."] * 1500
    
    print("Analyzing dataset and recommending configuration...\n")
    config = recommend_config(example_texts, priority='balanced')
    
    print("\nTo use this configuration:")
    print(f"""
result = await clusterize_texts(
    texts,
    config={config.config_preset},
    max_texts_per_run={config.max_texts_per_run},
    batch_size={config.batch_size},
    max_passes={config.max_passes},
    max_concurrent={config.max_concurrent}
)
""")
    
    # Run full test suite
    test_recommender()
        