# Agentic Text Clusterizer

A sophisticated, scalable text clustering system that combines AI-powered categorization with hierarchical merging for large-scale document processing. Built for production use with parallel processing, dynamic workflow management, and intelligent consolidation of similar categories.

## 🚀 Key Features

- **Hybrid AI Clustering**: Combines OpenAI GPT models with BERT embeddings for accurate text categorization
- **Scalable Architecture**: Handles large datasets through intelligent chunking and parallel processing
- **Dynamic Workflow Management**: Priority queue-based execution that adapts to available resources
- **Hierarchical Merging**: Advanced tree-based consolidation of clustering results from multiple workflows
- **Big Text Support**: Automatic chunking and consolidation for documents exceeding token limits
- **Production Ready**: Comprehensive error handling, logging, and performance monitoring

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Performance & Scaling](#performance--scaling)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- CUDA-compatible GPU (recommended for BERT embeddings)

### Dependencies

```bash
pip install pydantic pydantic-ai langgraph scikit-learn sentence-transformers tiktoken
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agent_clusterizer
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt  # If requirements.txt exists, otherwise install manually
```

4. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from agentic_clusterizer import clusterize_texts, CONFIG_BALANCED_HYBRID

async def main():
    texts = [
        "Machine learning algorithms for data analysis",
        "Web development with React and Node.js",
        "Financial modeling and risk assessment",
        "Database design and optimization"
    ]
    
    result = await clusterize_texts(texts, config=CONFIG_BALANCED_HYBRID)
    
    print(f"Found {len(result['categories'])} categories:")
    for category in result['categories']:
        print(f"- {category.name}: {category.text_count} texts")

asyncio.run(main())
```

### Large-Scale Processing

```python
from text_assignment_manager import TextAssignmentManager, Text
from tree_merge_processor import TreeMergeProcessor

# Handle large datasets with chunking and parallel processing
texts = [Text(id=f"text_{i}", content=f"Large document content {i}...") 
         for i in range(1000)]

# Phase 1: Dry assignment (plan the work)
manager = TextAssignmentManager(
    batch_size=10,
    max_batches_per_workflow=5,
    token_threshold=200
)
dry_assignment = manager.dry_assign(texts)

# Phase 2: Execute with hierarchical merging
processor = TreeMergeProcessor(max_parallel_merges=4)
result = await processor.process_with_dry_assignment(
    dry_assignment,
    enable_dynamic_merging=True  # Overlapping execution and merging
)
```

## 🏗 Architecture

### Core Components

1. **Agentic Clusterizer** (`agentic_clusterizer.py`)
   - Main clustering engine using OpenAI GPT and BERT
   - Multi-pass categorization with confidence scoring
   - Category consolidation and merging

2. **Text Assignment Manager** (`text_assignment_manager.py`)
   - Intelligent workload distribution
   - Big text chunking and tracking
   - Token-aware batching

3. **Tree Merge Processor** (`tree_merge_processor.py`)
   - Hierarchical result consolidation
   - Priority queue-based dynamic execution
   - Big text chunk aggregation

4. **Configuration Manager** (`configuration_manager.py`)
   - Unified configuration system
   - Auto-recommendation based on dataset characteristics
   - Preset configurations for common use cases

### Processing Pipeline

```
Input Texts → Dry Assignment → Parallel Clustering → Hierarchical Merging → Final Result
     ↓              ↓              ↓                      ↓              ↓
  Raw Data    Workload Planning  AI Categorization   Result Consolidation  Consolidated Categories
```

## ⚙️ Configuration

### Preset Configurations

```python
from configuration_manager import ClusterConfig

# Balanced quality and speed
config = ClusterConfig.balanced()

# Maximum quality (slower)
config = ClusterConfig.quality()

# Maximum speed (lower quality)
config = ClusterConfig.fast()

# Semantic similarity focus
config = ClusterConfig.semantic()

# Cost optimized
config = ClusterConfig.cost_optimized()
```

### Custom Configuration

```python
config = ClusterConfig(
    retrieval_mode='hybrid',  # 'tfidf', 'bert', or 'hybrid'
    scoring_strategy='balanced',  # 'conservative', 'balanced', 'aggressive'
    batch_size=20,
    max_passes=3,
    max_concurrent=5,
    enable_bert=True
)
```

### Auto-Recommendation

```python
# Automatically recommend optimal config for your data
config = ClusterConfig.from_texts(
    texts,
    priority='quality',  # 'speed', 'quality', 'cost', 'balanced'
    memory_limit_mb=2048,
    print_summary=True
)
```

## 📖 Usage Examples

### CNAE Dataset Processing

```python
from run_full_cnae import CNAEProcessor

# Process Brazilian business classification data
processor = CNAEProcessor(
    input_file='cnae.txt',
    output_dir='results',
    sample_size=50  # Optional: limit for testing
)

await processor.process()
results = processor.get_results()
```

### Custom Dataset Processing

```python
import json
from pathlib import Path

async def process_custom_dataset():
    # Load your data
    with open('my_texts.json', 'r') as f:
        data = json.load(f)
    
    texts = [Text(id=item['id'], content=item['content']) for item in data]
    
    # Configure for your use case
    config = ClusterConfig.from_texts(texts, priority='quality')
    
    # Process
    result = await clusterize_texts(texts, config=config)
    
    # Save results
    output = {
        'categories': [cat.dict() for cat in result['categories']],
        'assignments': result['assignments'],
        'metadata': result['metadata']
    }
    
    Path('results').mkdir(exist_ok=True)
    with open('results/clustering_result.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

asyncio.run(process_custom_dataset())
```

## 🔧 API Reference

### Main Functions

#### `clusterize_texts(texts, config=None, **kwargs)`

Main entry point for text clustering.

**Parameters:**
- `texts`: List of strings or Text objects
- `config`: ClusterConfig object (optional)
- `batch_size`: Texts per batch (default: 10)
- `max_passes`: Number of clustering passes (default: 2)
- `max_concurrent`: Max concurrent batches (default: 5)

**Returns:**
```python
{
    'categories': [Category],
    'assignments': [CategoryAssignment],
    'metadata': dict
}
```

#### `TextAssignmentManager.dry_assign(texts)`

Plan workload distribution without execution.

#### `TreeMergeProcessor.process_with_dry_assignment(dry_assignment, **kwargs)`

Execute planned workflows with hierarchical merging.

### Key Classes

- `ClusterConfig`: Configuration management
- `TextAssignmentManager`: Workload planning
- `TreeMergeProcessor`: Result consolidation
- `Category`: Clustering result category
- `CategoryAssignment`: Text-to-category assignment

## ⚡ Performance & Scaling

### Optimization Strategies

1. **Batch Sizing**: Adjust based on API rate limits and memory
2. **Parallel Processing**: Configure `max_concurrent` and `max_parallel_merges`
3. **Token Management**: Use appropriate `token_threshold` for chunking
4. **Configuration Tuning**: Use auto-recommendation for optimal settings

### Performance Tips

```python
# For large datasets (>1000 texts)
config = ClusterConfig.from_texts(texts, priority='speed')
processor = TreeMergeProcessor(max_parallel_merges=8)

# For high-quality results
config = ClusterConfig.from_texts(texts, priority='quality')
processor = TreeMergeProcessor(max_parallel_merges=4)

# Memory-constrained environments
config = ClusterConfig.from_texts(
    texts, 
    priority='balanced', 
    memory_limit_mb=1024
)
```

### Benchmark Results

- **Small datasets** (< 100 texts): 30-60 seconds
- **Medium datasets** (100-1000 texts): 2-5 minutes
- **Large datasets** (1000+ texts): 5-15 minutes (with parallel processing)

## 🔍 Troubleshooting

### Common Issues

#### BERT Similarity = 0.000
**Symptoms:** Consolidation shows `bert=0.000` in similarity scores
**Cause:** BERT embeddings not computed or tensor conversion issues
**Fix:** Ensure CUDA is available and embeddings are properly converted to numpy

#### Logging Placeholders Not Formatted
**Symptoms:** Logs show `%d/%d` instead of actual numbers
**Cause:** Mixed f-string and printf-style formatting
**Fix:** Use consistent string formatting throughout

#### Memory Issues
**Symptoms:** Out of memory errors during processing
**Cause:** Large batch sizes or insufficient RAM
**Fix:** Reduce `batch_size`, enable chunking, or increase memory limits

#### OpenAI API Rate Limits
**Symptoms:** API throttling errors
**Cause:** Too many concurrent requests
**Fix:** Reduce `max_concurrent` and add retry logic

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
result = await clusterize_texts(texts, config=config)
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install development dependencies
4. Run tests: `python -m pytest`
5. Submit a pull request

### Code Style

- Use type hints for all function parameters and return values
- Follow PEP 8 style guidelines
- Add docstrings to all public functions and classes
- Use descriptive variable names

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance benchmarks
python benchmarks/benchmark_clustering.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models and API
- Hugging Face for BERT models
- scikit-learn for TF-IDF and similarity calculations
- Pydantic for data validation
- LangGraph for workflow orchestration

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the example scripts in the repository

---

**Note**: This system requires an OpenAI API key and works best with GPU acceleration for BERT embeddings. Performance may vary based on your hardware and API quotas.</content>
<parameter name="filePath">/home/pingu/github/experiments/data/agent_clusterizer/README.md