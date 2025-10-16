# Agentic Clusterizer - Quick Reference Guide

## 🚀 Quick Start

```python
import asyncio
from agentic_clusterizer import clusterize_texts, CONFIG_BALANCED_HYBRID

texts = ["Your texts here...", "More texts...", "Even more..."]

result = await clusterize_texts(texts, config=CONFIG_BALANCED_HYBRID)

# Access results
categories = result['categories']     # List[Category]
assignments = result['assignments']   # List[CategoryAssignment]
metadata = result['metadata']         # Dict with stats
```

## 📋 Preset Configurations

| Configuration | Use Case | Threshold | Bonus | Best For |
|--------------|----------|-----------|-------|----------|
| `CONFIG_BALANCED_HYBRID` | **Recommended default** | 0.60 | 1.10 | Most use cases |
| `CONFIG_CONSERVATIVE_HYBRID` | Fewer categories | 0.70 | 1.05 | High precision needed |
| `CONFIG_AGGRESSIVE_HYBRID` | More categories | 0.50 | 1.20 | Catch nuances, early exploration |
| `CONFIG_SEMANTIC_BERT` | Semantic focus | 0.65 | 1.10 | Subtle semantic differences |
| `CONFIG_BALANCED_TFIDF` | Fast, no BERT | 0.60 | 1.10 | Speed critical, simple texts |
| `CONFIG_CONSERVATIVE_TFIDF` | Conservative, no BERT | 0.70 | 1.05 | Fast + high precision |
| `CONFIG_AGGRESSIVE_TFIDF` | Aggressive, no BERT | 0.50 | 1.20 | Fast + catch nuances |

## 🔧 API Reference

### `clusterize_texts()`
Standard categorization for normal-sized texts.

```python
result = await clusterize_texts(
    texts=my_texts,                      # Required: List[str]
    config=CONFIG_BALANCED_HYBRID,       # ClusterizerConfig
    max_passes=2,                        # Number of refinement passes
    batch_size=6,                        # Parallel batch size
    max_concurrent=5,                    # Max concurrent LLM calls
    prefilter_k=3,                       # Base K for candidate selection
)
```

### `clusterize_texts_with_chunking()`
Enhanced API for large texts with automatic chunking.

```python
result = await clusterize_texts_with_chunking(
    texts=my_texts,                      # Required: List[str]
    large_text_threshold=800,            # Chunk if ≥800 tokens
    chunk_size=500,                      # Tokens per chunk
    chunk_overlap=100,                   # Overlap for context
    multi_topic_threshold=0.3,           # 30% votes = multi-topic
    config=CONFIG_BALANCED_HYBRID,       # All standard params supported
    max_passes=2,
    batch_size=6,
)

# Additional returns
multi_topic_texts = result['multi_topic_texts']  # List of multi-topic docs
```

## 🎯 Custom Configuration

```python
from agentic_clusterizer import (
    ClusterizerConfig,
    ConfidenceWeights,
    RETRIEVAL_MODE,
    SCORING_STRATEGY
)

# Create custom weights
custom_weights = ConfidenceWeights(
    llm_score=0.40,           # LLM confidence
    tfidf_similarity=0.15,    # Lexical matching
    bert_similarity=0.25,     # Semantic similarity
    keyword_overlap=0.15,     # Keyword overlap
    category_maturity=0.03,   # Mature category bonus
    pass_number=0.02          # Later pass bonus
)

# Create custom config
custom_config = ClusterizerConfig(
    retrieval_mode=RETRIEVAL_MODE.HYBRID,
    confidence_weights=custom_weights,
    scoring_strategy=SCORING_STRATEGY.BALANCED,
    new_category_threshold=0.60,
    new_category_bonus=1.15
)

result = await clusterize_texts(texts, config=custom_config)
```

## 📊 Result Structure

```python
result = {
    'categories': [
        Category(
            id='tech_a1b2c3',
            name='Technology',
            description='Technology and innovation topics',
            keywords=['AI', 'software', 'hardware'],
            text_count=5
        ),
        # ... more categories
    ],
    
    'assignments': [
        CategoryAssignment(
            text='Original text here...',
            category_id='tech_a1b2c3',
            confidence=0.87,
            reasoning='Assigned (llm=0.85, tfidf=0.65, bert=0.82, final=0.87)'
        ),
        # ... more assignments
    ],
    
    'metadata': {
        'total_passes': 2,
        'total_texts': 100,
        'total_categories': 8,
        'total_steps': 15,
        'parallel_processing': True,
        'batch_size': 6,
        'max_concurrent': 5,
        'config': {...},
        'confidence_stats': {
            'average': 0.783,
            'min': 0.521,
            'max': 0.952,
            'high_confidence_count': 75,
            'medium_confidence_count': 20,
            'low_confidence_count': 5
        }
    },
    
    # Only with clusterize_texts_with_chunking()
    'multi_topic_texts': [
        {
            'text_index': 5,
            'text': 'Preview of text...',
            'primary_category': 'finance',
            'categories': ['finance', 'technology', 'politics'],
            'confidence': 0.82
        }
    ]
}
```

## 🔍 Key Features

### 1. SmartConsolidator
Prevents duplicate categories using 4 signals:

```python
# Automatically detects and merges:
# "Finance and Markets" + "Finance and Economics" → "Finance and Markets"

# Signals used:
# - TF-IDF similarity (threshold: 0.45)
# - BERT semantic (threshold: 0.70)
# - Keyword overlap (threshold: 0.40)
# - Name similarity (threshold: 0.60)
# - Combined score (threshold: 0.55)

# Auto-approves when:
# - BERT similarity ≥ 0.75, OR
# - Combined score ≥ 0.70 (aggressive) / 0.75 (conservative)
```

### 2. AdaptiveTopKSelector
Context-aware candidate retrieval:

```python
# Base K = 3, but adjusts based on:
# - Pass number: +50% on pass 2+
# - Category count: +1 per 10 categories
# - Text complexity: +1 for complex texts
# - New category mode: +50%

# Example:
# Pass 1, 10 categories, simple text → K = 3
# Pass 2, 30 categories, complex text → K = 6
```

### 3. Semantic Chunking
Handles large texts intelligently:

```python
# Features:
# - Sentence-boundary aware (never splits mid-sentence)
# - Configurable chunk size (default: 500 tokens)
# - Overlap for context (default: 100 tokens)
# - Multi-topic detection (secondary categories)
# - Confidence-weighted aggregation

# Use when:
texts_with_long_docs = ["Short headline", "Long article..." * 100]
result = await clusterize_texts_with_chunking(
    texts_with_long_docs,
    large_text_threshold=800,  # Chunk if ≥800 tokens
)
```

## 💡 Best Practices

### When to Use Each Configuration

**Balanced (Default)** ✅
- General purpose text categorization
- News articles, social media posts
- When you need good precision + recall

**Conservative** 🔒
- High-stakes categorization
- Medical, legal, financial documents
- When false positives are costly

**Aggressive** 🚀
- Exploratory data analysis
- Want to capture subtle differences
- Early-stage category discovery

**Semantic** 🧠
- Texts with subtle semantic differences
- Academic papers, technical docs
- When meaning > keywords

### Chunking Guidelines

**Use chunking for:**
- Documents ≥800 tokens (roughly 600 words)
- Multi-topic documents (articles, papers)
- When topic diversity matters

**Skip chunking for:**
- Short texts (tweets, headlines, < 200 tokens)
- Single-topic content
- When speed is critical

### Performance Tips

```python
# Faster processing
result = await clusterize_texts(
    texts,
    config=CONFIG_BALANCED_TFIDF,  # Skip BERT
    max_passes=1,                  # Single pass
    batch_size=10,                 # Larger batches
    max_concurrent=10              # More parallel calls
)

# Better accuracy
result = await clusterize_texts(
    texts,
    config=CONFIG_SEMANTIC_BERT,   # Heavy BERT
    max_passes=2,                  # Two passes
    batch_size=4,                  # Smaller batches
    prefilter_k=5                  # More candidates
)
```

## 🐛 Troubleshooting

### Problem: Too many categories created

```python
# Solution: Use conservative config
result = await clusterize_texts(
    texts,
    config=CONFIG_CONSERVATIVE_HYBRID  # Higher threshold (0.70)
)
```

### Problem: Categories not merging

```python
# Solution: Check BERT is enabled and use aggressive mode
result = await clusterize_texts(
    texts,
    config=CONFIG_AGGRESSIVE_HYBRID  # Lower thresholds, trusts BERT
)
```

### Problem: Low confidence scores

```python
# Solutions:
# 1. More passes
result = await clusterize_texts(texts, max_passes=3)

# 2. Custom weights favoring LLM
custom_weights = ConfidenceWeights(
    llm_score=0.50,  # Increase LLM trust
    # ... other weights lower
)
```

### Problem: Multi-topic detection not working

```python
# Solution: Lower threshold
result = await clusterize_texts_with_chunking(
    texts,
    multi_topic_threshold=0.20,  # Lower from 0.30
    chunk_size=300,               # Smaller chunks
)
```

## 📚 Common Patterns

### Pattern 1: News Categorization
```python
result = await clusterize_texts_with_chunking(
    news_articles,
    large_text_threshold=500,
    config=CONFIG_BALANCED_HYBRID,
    max_passes=2
)
```

### Pattern 2: Social Media Posts
```python
result = await clusterize_texts(
    tweets,
    config=CONFIG_AGGRESSIVE_HYBRID,  # Catch nuances
    max_passes=1,                     # Fast
    batch_size=20                     # Large batches
)
```

### Pattern 3: Research Papers
```python
result = await clusterize_texts_with_chunking(
    papers,
    large_text_threshold=1000,
    chunk_size=800,
    config=CONFIG_SEMANTIC_BERT,  # Semantic focus
    max_passes=2
)
```

### Pattern 4: Customer Feedback
```python
result = await clusterize_texts(
    feedback,
    config=CONFIG_BALANCED_HYBRID,
    max_passes=2,
    batch_size=10
)

# Analyze by confidence
high_conf = [a for a in result['assignments'] if a.confidence >= 0.75]
needs_review = [a for a in result['assignments'] if a.confidence < 0.60]
```

## 🔗 Related Files

- `examples.py` - Comprehensive examples (8 scenarios)
- `integration_guide.md` - Advanced integration patterns
- `agentic_clusterizer.py` - Full implementation
- `requirements.txt` - Dependencies

## 📖 Further Reading

### Configuration Dataclasses
- `RetrievalMode`: TFIDF, BERT, HYBRID
- `ScoringStrategy`: BALANCED, CONSERVATIVE, AGGRESSIVE, SEMANTIC
- `ConfidenceWeights`: 6 factors for confidence calculation
- `ClusterizerConfig`: Main configuration object

### Core Classes
- `Category`: Category model with metadata
- `CategoryAssignment`: Text-to-category assignment
- `SmartConsolidator`: Multi-signal merge detection
- `AdaptiveTopKSelector`: Context-aware K selection
- `SemanticChunker`: Smart text chunking
- `ChunkAggregator`: Chunk result aggregation

---

**Need help?** Check `examples.py` for complete working examples!
