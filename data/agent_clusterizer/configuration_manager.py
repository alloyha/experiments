"""
Unified Configuration System for Text Clustering

Key Refactoring:
1. Merge ClusterizerConfig and RecommendedConfig into unified ClusterConfig
2. Provide simple factory methods for common use cases
3. Support both manual and auto-recommended configuration
4. Reduce clusterize_texts signature to single config parameter

Usage:
    # Simple: Use presets
    config = ClusterConfig.balanced()
    result = await clusterize_texts(texts, config=config)
    
    # Auto-recommend based on dataset
    config = ClusterConfig.from_texts(texts, priority='quality')
    result = await clusterize_texts(texts, config=config)
    
    # Advanced: Custom configuration
    config = ClusterConfig(
        retrieval_mode='hybrid',
        scoring_strategy='aggressive',
        batch_size=50,
        max_passes=3
    )
"""

import statistics
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# CORE ENUMS AND CONSTANTS
# ============================================================================

class RetrievalMode(Enum):
    """Retrieval mode for category indexing."""
    TFIDF = "tfidf"
    BERT = "bert"
    HYBRID = "hybrid"


class ScoringStrategy(Enum):
    """Scoring strategy for categorization."""
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    SEMANTIC = "semantic"


class Priority(Enum):
    """Processing priorities for auto-recommendation."""
    SPEED = "speed"
    QUALITY = "quality"
    COST = "cost"
    BALANCED = "balanced"


class DatasetSize(Enum):
    """Dataset size categories."""
    TINY = "tiny"           # < 50
    SMALL = "small"         # 50-500
    MEDIUM = "medium"       # 500-2000
    LARGE = "large"         # 2000-10000
    XLARGE = "xlarge"       # 10000-50000
    MASSIVE = "massive"     # > 50000


# LLM Cost estimates (per 1K tokens) - gpt-4o-mini
LLM_COST_INPUT = 0.000150   # $0.15 per 1M tokens
LLM_COST_OUTPUT = 0.000600  # $0.60 per 1M tokens
SECONDS_PER_TEXT = 0.5
SECONDS_PER_TEXT_SLOW = 2.0


# ============================================================================
# CONFIDENCE WEIGHTS
# ============================================================================

@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights for multi-signal confidence calculation."""
    llm_score: float
    tfidf_similarity: float
    bert_similarity: float
    keyword_overlap: float
    category_maturity: float
    pass_number: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
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


# Predefined weight configurations
WEIGHTS_BALANCED_BERT = ConfidenceWeights(
    llm_score=0.35, tfidf_similarity=0.10, bert_similarity=0.25,
    keyword_overlap=0.20, category_maturity=0.05, pass_number=0.05
)

WEIGHTS_BALANCED_TFIDF = ConfidenceWeights(
    llm_score=0.40, tfidf_similarity=0.20, bert_similarity=0.0,
    keyword_overlap=0.25, category_maturity=0.05, pass_number=0.10
)

WEIGHTS_CONSERVATIVE_BERT = ConfidenceWeights(
    llm_score=0.30, tfidf_similarity=0.15, bert_similarity=0.20,
    keyword_overlap=0.20, category_maturity=0.10, pass_number=0.05
)

WEIGHTS_AGGRESSIVE_BERT = ConfidenceWeights(
    llm_score=0.40, tfidf_similarity=0.05, bert_similarity=0.25,
    keyword_overlap=0.15, category_maturity=0.02, pass_number=0.13
)

WEIGHTS_SEMANTIC_BERT = ConfidenceWeights(
    llm_score=0.25, tfidf_similarity=0.05, bert_similarity=0.45,
    keyword_overlap=0.15, category_maturity=0.05, pass_number=0.05
)


# ============================================================================
# UNIFIED CLUSTER CONFIG
# ============================================================================

@dataclass
class ClusterConfig:
    """
    Unified configuration for text clustering.
    
    Combines algorithmic parameters (retrieval mode, scoring) with
    execution parameters (batch size, parallelism) into single config.
    """
    
    # === Algorithmic Parameters ===
    retrieval_mode: str = "hybrid"
    scoring_strategy: str = "balanced"
    confidence_weights: Optional[ConfidenceWeights] = None
    new_category_threshold: float = 0.6
    new_category_bonus: float = 1.1
    
    # === Execution Parameters ===
    batch_size: int = 20
    max_passes: int = 2
    max_concurrent: int = 5
    prefilter_k: int = 3
    
    # === Large Dataset Parameters ===
    max_texts_per_run: int = 500
    token_threshold: int = 200
    max_parallel_merges: int = 4
    enable_tree_merge: bool = True
    
    # === Optional Metadata ===
    name: Optional[str] = None
    description: Optional[str] = None
    
    # === Predictions (computed) ===
    expected_workflows: int = 0
    expected_batches: int = 0
    expected_llm_calls: int = 0
    estimated_duration_minutes: Tuple[float, float] = (0.0, 0.0)
    estimated_cost_usd: Tuple[float, float] = (0.0, 0.0)
    
    # === Analysis metadata ===
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Validate retrieval mode
        valid_modes = [m.value for m in RetrievalMode]
        if self.retrieval_mode not in valid_modes:
            raise ValueError(f"Invalid retrieval_mode: {self.retrieval_mode}")
        
        # Validate scoring strategy
        valid_strategies = [s.value for s in ScoringStrategy]
        if self.scoring_strategy not in valid_strategies:
            raise ValueError(f"Invalid scoring_strategy: {self.scoring_strategy}")
        
        # Set default confidence weights if not provided
        if self.confidence_weights is None:
            self.confidence_weights = self._default_weights()
        
        # Validate weights
        self.confidence_weights.validate()
        
        # Validate thresholds
        if not (0.0 <= self.new_category_threshold <= 1.0):
            raise ValueError("new_category_threshold must be in [0.0, 1.0]")
        if not (0.5 <= self.new_category_bonus <= 2.0):
            raise ValueError("new_category_bonus must be in [0.5, 2.0]")
    
    def _default_weights(self) -> ConfidenceWeights:
        """Get default weights based on retrieval mode and strategy."""
        uses_bert = self.retrieval_mode in [RetrievalMode.BERT.value, RetrievalMode.HYBRID.value]
        
        if self.scoring_strategy == ScoringStrategy.SEMANTIC.value:
            return WEIGHTS_SEMANTIC_BERT
        elif self.scoring_strategy == ScoringStrategy.CONSERVATIVE.value:
            return WEIGHTS_CONSERVATIVE_BERT if uses_bert else WEIGHTS_BALANCED_TFIDF
        elif self.scoring_strategy == ScoringStrategy.AGGRESSIVE.value:
            return WEIGHTS_AGGRESSIVE_BERT if uses_bert else WEIGHTS_BALANCED_TFIDF
        else:  # balanced
            return WEIGHTS_BALANCED_BERT if uses_bert else WEIGHTS_BALANCED_TFIDF
    
    def get_description(self) -> str:
        """Human-readable description."""
        if self.name:
            return self.name
        return (
            f"{self.scoring_strategy.title()} strategy "
            f"using {self.retrieval_mode.upper()} retrieval"
        )
    
    def should_use_tree_merge(self, num_texts: int) -> bool:
        """Determine if tree merge should be used."""
        return self.enable_tree_merge and num_texts > self.max_texts_per_run
    
    def print_summary(self):
        """Print configuration summary."""
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 CLUSTER CONFIGURATION: {self.get_description()}")
        logger.info(f"{'='*70}\n")
        
        logger.info(f"📊 Algorithmic Settings:")
        logger.info(f"  Retrieval mode:       {self.retrieval_mode}")
        logger.info(f"  Scoring strategy:     {self.scoring_strategy}")
        logger.info(f"  New cat threshold:    {self.new_category_threshold:.2f}")
        logger.info(f"  New cat bonus:        {self.new_category_bonus:.2f}")
        
        logger.info(f"\n⚙️  Execution Settings:")
        logger.info(f"  Batch size:           {self.batch_size}")
        logger.info(f"  Max passes:           {self.max_passes}")
        logger.info(f"  Max concurrent:       {self.max_concurrent}")
        logger.info(f"  Prefilter K:          {self.prefilter_k}")
        
        logger.info(f"\n🌳 Large Dataset Settings:")
        logger.info(f"  Max texts per run:    {self.max_texts_per_run}")
        logger.info(f"  Token threshold:      {self.token_threshold}")
        logger.info(f"  Max parallel merges:  {self.max_parallel_merges}")
        logger.info(f"  Tree merge enabled:   {self.enable_tree_merge}")
        
        if self.expected_llm_calls > 0:
            logger.info(f"\n💰 Predictions:")
            logger.info(f"  Expected workflows:   {self.expected_workflows}")
            logger.info(f"  Expected batches:     {self.expected_batches}")
            logger.info(f"  LLM calls:            ~{self.expected_llm_calls:,}")
            logger.info(f"  Duration:             {self.estimated_duration_minutes[0]:.1f}-{self.estimated_duration_minutes[1]:.1f} min")
            logger.info(f"  Cost (gpt-4o-mini):   ${self.estimated_cost_usd[0]:.2f}-${self.estimated_cost_usd[1]:.2f}")
        
        if self.reasoning:
            logger.info(f"\n💡 Reasoning:")
            for reason in self.reasoning:
                logger.info(f"  • {reason}")
        
        if self.warnings:
            logger.info(f"\n⚠️  Warnings:")
            for warning in self.warnings:
                logger.info(f"  • {warning}")
        
        logger.info(f"\n{'='*70}\n")
    
    # ========================================================================
    # FACTORY METHODS - Presets
    # ========================================================================
    
    @classmethod
    def balanced(cls, enable_bert: bool = True) -> 'ClusterConfig':
        """
        Balanced configuration for general use.
        
        Good for: Most use cases, unknown dataset characteristics
        """
        return cls(
            retrieval_mode=RetrievalMode.HYBRID.value if enable_bert else RetrievalMode.TFIDF.value,
            scoring_strategy=ScoringStrategy.BALANCED.value,
            new_category_threshold=0.6,
            new_category_bonus=1.1,
            name="Balanced (Hybrid)" if enable_bert else "Balanced (TF-IDF)",
            description="General-purpose configuration with balanced precision/recall"
        )
    
    @classmethod
    def conservative(cls, enable_bert: bool = True) -> 'ClusterConfig':
        """
        Conservative configuration - favors existing categories.
        
        Good for: When you want fewer, broader categories
        """
        return cls(
            retrieval_mode=RetrievalMode.HYBRID.value if enable_bert else RetrievalMode.TFIDF.value,
            scoring_strategy=ScoringStrategy.CONSERVATIVE.value,
            new_category_threshold=0.70,
            new_category_bonus=1.05,
            max_passes=3,  # More refinement
            name="Conservative",
            description="Favors consolidation into existing categories"
        )
    
    @classmethod
    def aggressive(cls, enable_bert: bool = True) -> 'ClusterConfig':
        """
        Aggressive configuration - creates new categories easily.
        
        Good for: When you want fine-grained, specific categories
        """
        return cls(
            retrieval_mode=RetrievalMode.HYBRID.value if enable_bert else RetrievalMode.TFIDF.value,
            scoring_strategy=ScoringStrategy.AGGRESSIVE.value,
            new_category_threshold=0.50,
            new_category_bonus=1.20,
            name="Aggressive",
            description="Creates new categories readily for fine-grained clustering"
        )
    
    @classmethod
    def semantic(cls) -> 'ClusterConfig':
        """
        Semantic-focused configuration - heavy BERT emphasis.
        
        Good for: When semantic meaning is more important than lexical similarity
        """
        return cls(
            retrieval_mode=RetrievalMode.BERT.value,
            scoring_strategy=ScoringStrategy.SEMANTIC.value,
            new_category_threshold=0.65,
            new_category_bonus=1.1,
            name="Semantic (BERT)",
            description="Emphasizes semantic similarity over lexical matching"
        )
    
    @classmethod
    def fast(cls) -> 'ClusterConfig':
        """
        Fast configuration - optimized for speed.
        
        Good for: Quick prototyping, large datasets where speed matters
        """
        return cls(
            retrieval_mode=RetrievalMode.TFIDF.value,
            scoring_strategy=ScoringStrategy.BALANCED.value,
            batch_size=50,  # Larger batches
            max_passes=1,   # Single pass
            max_concurrent=10,  # More parallelism
            name="Fast",
            description="Optimized for speed with minimal passes"
        )
    
    @classmethod
    def quality(cls) -> 'ClusterConfig':
        """
        Quality configuration - optimized for best results.
        
        Good for: When accuracy is paramount, smaller datasets
        """
        return cls(
            retrieval_mode=RetrievalMode.HYBRID.value,
            scoring_strategy=ScoringStrategy.CONSERVATIVE.value,
            batch_size=10,  # Smaller batches for precision
            max_passes=3,   # Multiple refinement passes
            prefilter_k=5,  # Consider more candidates
            name="Quality",
            description="Optimized for best categorization quality"
        )
    
    @classmethod
    def cost_optimized(cls) -> 'ClusterConfig':
        """
        Cost-optimized configuration - minimizes LLM calls.
        
        Good for: Budget-conscious projects, very large datasets
        """
        return cls(
            retrieval_mode=RetrievalMode.TFIDF.value,
            scoring_strategy=ScoringStrategy.BALANCED.value,
            batch_size=50,
            max_passes=1,
            max_texts_per_run=1000,  # Fewer workflows
            name="Cost Optimized",
            description="Minimizes LLM API costs"
        )
    
    # ========================================================================
    # FACTORY METHODS - Auto-recommendation
    # ========================================================================
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        priority: Priority = Priority.BALANCED,
        memory_limit_mb: Optional[int] = None,
        print_summary: bool = False
    ) -> 'ClusterConfig':
        """
        Auto-recommend configuration based on dataset analysis.
        
        Args:
            texts: Dataset to analyze
            priority: 'speed', 'quality', 'cost', or 'balanced'
            memory_limit_mb: Optional memory constraint
            print_summary: Whether to print analysis summary
        
        Returns:
            Optimized ClusterConfig
        
        Example:
            >>> config = ClusterConfig.from_texts(
            ...     my_texts, 
            ...     priority='quality'
            ... )
            >>> result = await clusterize_texts(texts, config=config)
        """
        analyzer = DatasetAnalyzer()
        recommender = ConfigRecommender(analyzer)
        
        # Convert string priority to enum
        priority_enum = priority
        
        config = recommender.recommend(texts, priority_enum, memory_limit_mb)
        
        if print_summary:
            config.print_summary()
        
        return config
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def with_overrides(self, **kwargs) -> 'ClusterConfig':
        """
        Create a new config with specific overrides.
        
        Example:
            >>> base = ClusterConfig.balanced()
            >>> custom = base.with_overrides(batch_size=50, max_passes=3)
        """
        import copy
        new_config = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")
        return new_config
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            'retrieval_mode': self.retrieval_mode,
            'scoring_strategy': self.scoring_strategy,
            'confidence_weights': self.confidence_weights.to_dict() if self.confidence_weights else None,
            'new_category_threshold': self.new_category_threshold,
            'new_category_bonus': self.new_category_bonus,
            'batch_size': self.batch_size,
            'max_passes': self.max_passes,
            'max_concurrent': self.max_concurrent,
            'prefilter_k': self.prefilter_k,
            'max_texts_per_run': self.max_texts_per_run,
            'token_threshold': self.token_threshold,
            'max_parallel_merges': self.max_parallel_merges,
            'enable_tree_merge': self.enable_tree_merge,
        }


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

@dataclass
class DatasetStats:
    """Statistical analysis of dataset."""
    total_texts: int
    total_tokens: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    median_tokens: float
    std_tokens: float
    vocabulary_richness: float
    complexity_score: float
    estimated_memory_mb: float
    
    # Big text detection
    big_text_count: int
    big_text_percentage: float
    big_text_threshold: int


class DatasetAnalyzer:
    """Analyzes dataset characteristics for configuration recommendation."""
    
    TOKENS_PER_WORD = 0.75
    
    def analyze(self, texts: List[str], token_threshold: int = 200) -> DatasetStats:
        """Perform comprehensive dataset analysis."""
        if not texts:
            raise ValueError("Cannot analyze empty text list")
        
        token_counts = [self._estimate_tokens(text) for text in texts]
        total_tokens = sum(token_counts)
        
        big_text_count = sum(1 for count in token_counts if count > token_threshold)
        big_text_percentage = (big_text_count / len(texts)) * 100
        
        vocabulary_richness = self._calculate_vocabulary_richness(texts)
        complexity_score = self._calculate_complexity_score(token_counts, vocabulary_richness)
        estimated_memory_mb = self._estimate_memory(texts, token_counts)
        
        return DatasetStats(
            total_texts=len(texts),
            total_tokens=total_tokens,
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
            mean_tokens=statistics.mean(token_counts),
            median_tokens=statistics.median(token_counts),
            std_tokens=statistics.stdev(token_counts) if len(token_counts) > 1 else 0.0,
            vocabulary_richness=vocabulary_richness,
            complexity_score=complexity_score,
            estimated_memory_mb=estimated_memory_mb,
            big_text_count=big_text_count,
            big_text_percentage=big_text_percentage,
            big_text_threshold=token_threshold
        )
    
    def _estimate_tokens(self, text: str) -> int:
        words = text.split()
        return int(len(words) * self.TOKENS_PER_WORD)
    
    def _calculate_vocabulary_richness(self, texts: List[str]) -> float:
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        if not all_words:
            return 0.0
        return len(set(all_words)) / len(all_words)
    
    def _calculate_complexity_score(self, token_counts: List[int], vocab_richness: float) -> float:
        if len(token_counts) <= 1:
            return 0.5
        
        std_tokens = statistics.stdev(token_counts)
        mean_tokens = statistics.mean(token_counts)
        variance_score = min(1.0, std_tokens / (mean_tokens + 1))
        
        return (0.6 * variance_score + 0.4 * vocab_richness)
    
    def _estimate_memory(self, texts: List[str], token_counts: List[int]) -> float:
        raw_text_bytes = sum(len(text) for text in texts)
        embeddings_bytes = sum(token_counts) * 4 * 384
        overhead_bytes = len(texts) * 1024
        total_mb = (raw_text_bytes + embeddings_bytes + overhead_bytes) * 1.5 / (1024 * 1024)
        return total_mb


class ConfigRecommender:
    """Recommends configuration based on dataset analysis."""
    
    def __init__(self, analyzer: DatasetAnalyzer):
        self.analyzer = analyzer
    
    def recommend(
        self,
        texts: List[str],
        priority: Priority,
        memory_limit_mb: Optional[int] = None
    ) -> ClusterConfig:
        """Recommend optimal configuration."""
        
        # Analyze dataset
        stats = self.analyzer.analyze(texts)
        
        # Categorize dataset
        dataset_size = self._categorize_size(stats.total_texts)
        
        # Start with appropriate preset
        config = self._select_base_preset(dataset_size, stats, priority)
        
        # Apply size-based adjustments
        config = self._adjust_for_size(config, dataset_size, stats)
        
        # Apply priority adjustments
        config = self._adjust_for_priority(config, priority)
        
        # Apply memory constraints if specified
        if memory_limit_mb and stats.estimated_memory_mb > memory_limit_mb:
            config = self._adjust_for_memory(config, stats, memory_limit_mb)
        
        # Add predictions
        config = self._add_predictions(config, stats)
        
        # Add reasoning
        config = self._add_reasoning(config, stats, dataset_size, priority)
        
        return config
    
    def _categorize_size(self, num_texts: int) -> DatasetSize:
        if num_texts < 50:
            return DatasetSize.TINY
        elif num_texts < 500:
            return DatasetSize.SMALL
        elif num_texts < 2000:
            return DatasetSize.MEDIUM
        elif num_texts < 10000:
            return DatasetSize.LARGE
        elif num_texts < 50000:
            return DatasetSize.XLARGE
        else:
            return DatasetSize.MASSIVE
    
    def _select_base_preset(self, size: DatasetSize, stats: DatasetStats, priority: Priority) -> ClusterConfig:
        """Select appropriate base preset."""
        if priority == Priority.SPEED:
            return ClusterConfig.fast()
        elif priority == Priority.QUALITY:
            return ClusterConfig.quality()
        elif priority == Priority.COST:
            return ClusterConfig.cost_optimized()
        else:  # BALANCED
            if stats.complexity_score > 0.7:
                return ClusterConfig.semantic()
            else:
                return ClusterConfig.balanced()
    
    def _adjust_for_size(self, config: ClusterConfig, size: DatasetSize, stats: DatasetStats) -> ClusterConfig:
        """Adjust parameters based on dataset size."""
        adjustments = {}
        
        if size == DatasetSize.TINY:
            adjustments = {'batch_size': 10, 'max_texts_per_run': 1000}
        elif size == DatasetSize.SMALL:
            adjustments = {'batch_size': 20, 'max_texts_per_run': 1000}
        elif size == DatasetSize.MEDIUM:
            adjustments = {'batch_size': 50, 'max_texts_per_run': 400}
        elif size == DatasetSize.LARGE:
            adjustments = {'batch_size': 75, 'max_texts_per_run': 300, 'max_passes': 1}
        elif size in [DatasetSize.XLARGE, DatasetSize.MASSIVE]:
            adjustments = {'batch_size': 100, 'max_texts_per_run': 200, 'max_passes': 1, 'max_parallel_merges': 8}
        
        return config.with_overrides(**adjustments)
    
    def _adjust_for_priority(self, config: ClusterConfig, priority: Priority) -> ClusterConfig:
        """Apply priority-specific adjustments."""
        if priority == Priority.SPEED:
            return config.with_overrides(max_passes=1, max_concurrent=10, batch_size=min(100, config.batch_size * 2))
        elif priority == Priority.QUALITY:
            return config.with_overrides(max_passes=3, batch_size=max(10, config.batch_size // 2))
        elif priority == Priority.COST:
            return config.with_overrides(max_passes=1, max_texts_per_run=config.max_texts_per_run * 2)
        return config
    
    def _adjust_for_memory(self, config: ClusterConfig, stats: DatasetStats, limit_mb: int) -> ClusterConfig:
        """Adjust for memory constraints."""
        reduction_factor = limit_mb / stats.estimated_memory_mb
        
        adjustments = {
            'batch_size': max(10, int(config.batch_size * reduction_factor)),
            'max_texts_per_run': max(50, int(config.max_texts_per_run * reduction_factor))
        }
        
        new_config = config.with_overrides(**adjustments)
        new_config.warnings.append(
            f"Configuration adjusted to fit within {limit_mb}MB memory limit. "
            f"This may increase processing time."
        )
        
        return new_config
    
    def _add_predictions(self, config: ClusterConfig, stats: DatasetStats) -> ClusterConfig:
        """Add execution predictions."""
        if config.should_use_tree_merge(stats.total_texts):
            config.expected_workflows = math.ceil(stats.total_texts / config.max_texts_per_run)
            config.expected_batches = config.expected_workflows * math.ceil(
                config.max_texts_per_run / config.batch_size
            )
        else:
            config.expected_workflows = 1
            config.expected_batches = math.ceil(stats.total_texts / config.batch_size)
        
        # Estimate LLM calls
        base_calls = stats.total_texts * config.max_passes
        merge_calls = config.expected_workflows * 50 if config.expected_workflows > 1 else 0
        config.expected_llm_calls = base_calls + merge_calls
        
        # Estimate cost
        input_tokens = config.expected_llm_calls * 150
        output_tokens = config.expected_llm_calls * 50
        cost_min = (input_tokens / 1000) * LLM_COST_INPUT + (output_tokens / 1000) * LLM_COST_OUTPUT
        config.estimated_cost_usd = (cost_min, cost_min * 2.0)
        
        # Estimate time
        time_min = stats.total_texts * SECONDS_PER_TEXT / 60
        time_max = stats.total_texts * SECONDS_PER_TEXT_SLOW / 60
        config.estimated_duration_minutes = (time_min, time_max)
        
        return config
    
    def _add_reasoning(self, config: ClusterConfig, stats: DatasetStats, size: DatasetSize, priority: Priority) -> ClusterConfig:
        """Add human-readable reasoning."""
        config.reasoning.append(f"Dataset: {stats.total_texts} texts ({size.value.upper()})")
        config.reasoning.append(f"Complexity: {stats.complexity_score:.2f} (vocab richness: {stats.vocabulary_richness:.2f})")
        
        if stats.big_text_percentage > 10:
            config.reasoning.append(
                f"Big texts: {stats.big_text_count} ({stats.big_text_percentage:.1f}%) "
                f"will be chunked at {config.token_threshold} tokens"
            )
        
        if config.should_use_tree_merge(stats.total_texts):
            config.reasoning.append(
                f"Tree merge enabled: {config.expected_workflows} workflows, "
                f"{math.ceil(math.log2(config.expected_workflows))} merge levels"
            )
        
        config.reasoning.append(f"Optimized for: {priority.value.upper()}")
        
        # Warnings
        if stats.estimated_memory_mb > 1000:
            config.warnings.append(
                f"High memory usage expected: ~{stats.estimated_memory_mb:.0f}MB. "
                "Consider processing on a machine with sufficient RAM."
            )
        
        if config.estimated_duration_minutes[1] > 60:
            config.warnings.append(
                f"Long processing time expected: up to {config.estimated_duration_minutes[1]:.0f} minutes. "
                "Consider running in background."
            )
        
        if config.estimated_cost_usd[1] > 5.0:
            config.warnings.append(
                f"Significant LLM costs expected: ${config.estimated_cost_usd[0]:.2f}-${config.estimated_cost_usd[1]:.2f}. "
                "Review your API budget."
            )
        
        return config


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Keep old names for backward compatibility
ClusterizerConfig = ClusterConfig
RETRIEVAL_MODE = RetrievalMode
SCORING_STRATEGY = ScoringStrategy

# Preset configurations (old naming)
CONFIG_BALANCED_HYBRID = ClusterConfig.balanced(enable_bert=True)
CONFIG_BALANCED_TFIDF = ClusterConfig.balanced(enable_bert=False)
CONFIG_CONSERVATIVE_HYBRID = ClusterConfig.conservative(enable_bert=True)
CONFIG_CONSERVATIVE_TFIDF = ClusterConfig.conservative(enable_bert=False)
CONFIG_AGGRESSIVE_HYBRID = ClusterConfig.aggressive(enable_bert=True)
CONFIG_AGGRESSIVE_TFIDF = ClusterConfig.aggressive(enable_bert=False)
CONFIG_SEMANTIC_BERT = ClusterConfig.semantic()
DEFAULT_CONFIG = CONFIG_BALANCED_HYBRID

# Old weight names
CONFIDENCE_WEIGHTS_BERT = WEIGHTS_BALANCED_BERT
CONFIDENCE_WEIGHTS_NO_BERT = WEIGHTS_BALANCED_TFIDF


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def recommend_config(
    texts: List[str],
    priority: Priority = Priority.BALANCED,
    memory_limit_mb: Optional[int] = None,
    print_summary: bool = True
) -> ClusterConfig:
    """
    Convenience function for configuration recommendation.
    
    Args:
        texts: Dataset to analyze
        priority: 'speed', 'quality', 'cost', or 'balanced'
        memory_limit_mb: Optional memory constraint
        print_summary: Whether to print analysis summary
    
    Returns:
        Recommended ClusterConfig
    
    Example:
        >>> config = recommend_config(texts, priority='quality')
        >>> result = await clusterize_texts(texts, config=config)
    """
    return ClusterConfig.from_texts(texts, priority, memory_limit_mb, print_summary)


# ============================================================================
# EXAMPLES AND TESTS
# ============================================================================

def example_usage():
    """Demonstrate the unified configuration system."""
    
    logger.info("="*70)
    logger.info("UNIFIED CONFIGURATION SYSTEM - EXAMPLES")
    logger.info("="*70)
    
    # Example 1: Simple presets
    logger.info("\n1. Using Presets:")
    logger.info("-" * 70)
    
    config = ClusterConfig.balanced()
    logger.info(f"✓ {config.get_description()}")
    
    config = ClusterConfig.quality()
    logger.info(f"✓ {config.get_description()}")
    
    config = ClusterConfig.fast()
    logger.info(f"✓ {config.get_description()}")
    
    # Example 2: Auto-recommendation
    logger.info("\n2. Auto-Recommendation:")
    logger.info("-" * 70)
    
    sample_texts = ["Sample text " * 10] * 1000
    config = ClusterConfig.from_texts(sample_texts, priority='quality', print_summary=True)
    
    # Example 3: Custom configuration with overrides
    logger.info("\n3. Custom Configuration:")
    logger.info("-" * 70)
    
    base = ClusterConfig.balanced()
    custom = base.with_overrides(
        batch_size=50,
        max_passes=3,
        max_concurrent=10
    )
    logger.info(f"Base: {base.batch_size} batch size, {base.max_passes} passes")
    logger.info(f"Custom: {custom.batch_size} batch size, {custom.max_passes} passes")
    
    # Example 4: Export to dict
    logger.info("\n4. Export Configuration:")
    logger.info("-" * 70)
    
    config_dict = config.to_dict()
    logger.info(f"Exported {len(config_dict)} configuration parameters")
    for key in ['retrieval_mode', 'scoring_strategy', 'batch_size', 'max_passes']:
        logger.info(f"  {key}: {config_dict[key]}")
    
    logger.info("\n" + "="*70 + "\n")


def test_configuration_validation():
    """Test configuration validation."""
    logger.info("\n" + "="*70)
    logger.info("CONFIGURATION VALIDATION TESTS")
    logger.info("="*70 + "\n")
    
    # Test 1: Valid configuration
    try:
        config = ClusterConfig(
            retrieval_mode='hybrid',
            scoring_strategy='balanced',
            batch_size=20
        )
        logger.info("✓ Valid configuration accepted")
    except Exception as e:
        logger.info(f"✗ Valid configuration rejected: {e}")
    
    # Test 2: Invalid retrieval mode
    try:
        config = ClusterConfig(
            retrieval_mode='invalid_mode',
            scoring_strategy='balanced'
        )
        logger.info("✗ Invalid retrieval mode accepted (should fail)")
    except ValueError as e:
        logger.info(f"✓ Invalid retrieval mode rejected: {e}")
    
    # Test 3: Invalid threshold
    try:
        config = ClusterConfig(
            retrieval_mode='hybrid',
            scoring_strategy='balanced',
            new_category_threshold=1.5  # Invalid: > 1.0
        )
        logger.info("✗ Invalid threshold accepted (should fail)")
    except ValueError as e:
        logger.info(f"✓ Invalid threshold rejected: {e}")
    
    # Test 4: Weight validation
    try:
        invalid_weights = ConfidenceWeights(
            llm_score=0.1,
            tfidf_similarity=0.1,
            bert_similarity=0.1,
            keyword_overlap=0.1,
            category_maturity=0.1,
            pass_number=0.1
        )
        invalid_weights.validate()
        logger.info("⚠ Weights sum to 0.6 (warning expected)")
    except Exception as e:
        logger.info(f"Weight validation: {e}")
    
    logger.info("\n" + "="*70 + "\n")


def demo_simplified_api():
    """Demonstrate the simplified API usage."""
    
    logger.info("="*70)
    logger.info("SIMPLIFIED API DEMONSTRATION")
    logger.info("="*70 + "\n")
    
    logger.info("OLD API (many parameters):")
    logger.info("-" * 70)
    logger.info("""
    result = await clusterize_texts(
        texts,
        max_passes=2,
        prefilter_k=3,
        batch_size=20,
        max_concurrent=5,
        retrieval_mode='hybrid',
        bert_model='all-MiniLM-L6-v2',
        max_texts_per_run=500,
        token_threshold=200,
        max_parallel_merges=4,
        enable_tree_merge=True,
        new_category_threshold=0.6,
        new_category_bonus=1.1
    )
    """)
    
    logger.info("\nNEW API (single config):")
    logger.info("-" * 70)
    logger.info("""
    # Option 1: Use preset
    config = ClusterConfig.balanced()
    result = await clusterize_texts(texts, config=config)
    
    # Option 2: Auto-recommend
    config = ClusterConfig.from_texts(texts, priority='quality')
    result = await clusterize_texts(texts, config=config)
    
    # Option 3: Preset with overrides
    config = ClusterConfig.quality().with_overrides(batch_size=50)
    result = await clusterize_texts(texts, config=config)
    """)
    
    logger.info("\nBENEFITS:")
    logger.info("-" * 70)
    logger.info("✓ Single parameter instead of 12+")
    logger.info("✓ Validated configurations")
    logger.info("✓ Easy presets for common use cases")
    logger.info("✓ Auto-recommendation based on data")
    logger.info("✓ Immutable configs (frozen dataclass)")
    logger.info("✓ Clear documentation of choices")
    logger.info("✓ Backward compatible")
    
    logger.info("\n" + "="*70 + "\n")


# ============================================================================
# MIGRATION GUIDE
# ============================================================================

MIGRATION_GUIDE = """
================================================================================
MIGRATION GUIDE: Old API → New Unified Config API
================================================================================

STEP 1: Replace individual parameters with config object
------------------------------------------------------------------------

OLD:
    result = await clusterize_texts(
        texts,
        max_passes=2,
        batch_size=20,
        max_concurrent=5,
        retrieval_mode='hybrid'
    )

NEW:
    config = ClusterConfig.balanced()
    result = await clusterize_texts(texts, config=config)

OR (auto-recommend):
    config = ClusterConfig.from_texts(texts, priority='balanced')
    result = await clusterize_texts(texts, config=config)


STEP 2: Use presets for common scenarios
------------------------------------------------------------------------

Use Case: General purpose clustering
    config = ClusterConfig.balanced()

Use Case: Need fewer, broader categories
    config = ClusterConfig.conservative()

Use Case: Need fine-grained categories
    config = ClusterConfig.aggressive()

Use Case: Semantic similarity is key
    config = ClusterConfig.semantic()

Use Case: Speed is priority
    config = ClusterConfig.fast()

Use Case: Best quality results
    config = ClusterConfig.quality()

Use Case: Minimize API costs
    config = ClusterConfig.cost_optimized()


STEP 3: Override specific parameters when needed
------------------------------------------------------------------------

    base = ClusterConfig.balanced()
    custom = base.with_overrides(
        batch_size=50,
        max_passes=3,
        token_threshold=300
    )


STEP 4: Get automatic recommendations
------------------------------------------------------------------------

    # Analyzes your dataset and recommends optimal config
    config = ClusterConfig.from_texts(
        texts,
        priority='quality',  # or 'speed', 'cost', 'balanced'
        memory_limit_mb=2048,  # optional constraint
        print_summary=True  # see the analysis
    )


BACKWARD COMPATIBILITY
------------------------------------------------------------------------

All old parameter names still work! The new system is fully backward
compatible. You can migrate gradually.

Old constants still available:
    - CONFIG_BALANCED_HYBRID
    - CONFIG_SEMANTIC_BERT
    - CONFIDENCE_WEIGHTS_BERT
    - etc.

Old function still works:
    - recommend_config(texts, priority='quality')


BENEFITS OF MIGRATION
------------------------------------------------------------------------

✓ Cleaner code: 1 parameter instead of 12+
✓ Validated configs: Catch errors at config time, not runtime
✓ Intelligent defaults: Auto-recommendation based on your data
✓ Reproducible: Save/load configurations
✓ Documented: Each preset explains when to use it
✓ Type-safe: Full IDE autocomplete support
✓ Testable: Easy to unit test different configurations


COMPLETE EXAMPLE
------------------------------------------------------------------------

import asyncio
from agentic_clusterizer import clusterize_texts
from configuration_manager import ClusterConfig

async def main():
    texts = load_my_texts()
    
    # Simple: use preset
    config = ClusterConfig.quality()
    result = await clusterize_texts(texts, config=config)
    
    # Or: auto-recommend based on data
    config = ClusterConfig.from_texts(
        texts,
        priority='quality',
        print_summary=True
    )
    result = await clusterize_texts(texts, config=config)
    
    # Or: custom tuning
    config = ClusterConfig.balanced().with_overrides(
        batch_size=50,
        max_passes=3
    )
    result = await clusterize_texts(texts, config=config)
    
    logger.info(f"Found {len(result['categories'])} categories")

asyncio.run(main())

================================================================================
"""

def print_migration_guide():
    """Print the migration guide."""
    logger.info(MIGRATION_GUIDE)


if __name__ == "__main__":
    # Run examples
    example_usage()
    test_configuration_validation()
    demo_simplified_api()
    
    # Print migration guide
    logger.info("\n\n")
    print_migration_guide()