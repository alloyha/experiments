"""
Comprehensive Examples for Agentic Clusterizer

This file demonstrates:
1. Basic usage with preset configurations
2. Custom configuration creation
3. Chunked categorization for large texts
4. Multi-topic document detection
5. Configuration comparison
6. Advanced use cases
"""

import asyncio
from agentic_clusterizer import (
    # Main APIs
    clusterize_texts,
    clusterize_texts_with_chunking,
    
    # Configuration classes
    ClusterizerConfig,
    ConfidenceWeights,
    RetrievalMode,
    ScoringStrategy,
    
    # Preset configurations
    CONFIG_BALANCED_HYBRID,
    CONFIG_BALANCED_TFIDF,
    CONFIG_CONSERVATIVE_HYBRID,
    CONFIG_CONSERVATIVE_TFIDF,
    CONFIG_AGGRESSIVE_HYBRID,
    CONFIG_AGGRESSIVE_TFIDF,
    CONFIG_SEMANTIC_BERT,
    
    # Singletons
    RETRIEVAL_MODE,
    SCORING_STRATEGY,
    
    # Models
    Category,
    CategoryAssignment
)


# ==============================================================================
# EXAMPLE 1: Basic Usage with Preset Configurations
# ==============================================================================

async def example_1_basic_usage():
    """
    Demonstrate basic categorization with different preset configurations.
    
    Use Case: Standard text categorization with balanced settings.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage with Preset Configurations")
    print("="*70)
    
    # Sample texts covering different domains
    texts = [
        "The stock market reached new highs today with technology shares leading gains.",
        "Scientists discovered a new species of deep-sea fish near the Mariana Trench.",
        "The new iPhone features improved battery life and camera capabilities.",
        "Federal Reserve announces interest rate decision affecting global markets.",
        "Researchers found evidence of water on Jupiter's moon Europa.",
        "Samsung launches foldable smartphone with innovative display technology.",
        "Astronomers detect unusual radio signals from distant galaxy.",
        "Cryptocurrency prices surge following institutional adoption news.",
        "Marine biologists study coral reef recovery after bleaching event.",
        "Google unveils new AI model with advanced language understanding.",
    ]
    
    # Use recommended balanced hybrid configuration
    print("\n[Balanced Hybrid] - Recommended for most use cases")
    print("-" * 70)
    
    result = await clusterize_texts(
        texts=texts,
        config=CONFIG_BALANCED_HYBRID,
        max_passes=2,
        batch_size=6
    )
    
    # Display results
    print(f"\nResults:")
    print(f"  Categories created: {len(result['categories'])}")
    print(f"  Texts categorized: {len(result['assignments'])}")
    print(f"  Average confidence: {result['metadata']['confidence_stats']['average']:.3f}")
    
    print(f"\nCategories:")
    for cat in result['categories']:
        print(f"  - {cat.name} ({cat.text_count} texts)")
        print(f"    Keywords: {', '.join(cat.keywords[:5])}")
    
    print(f"\nSample Assignments:")
    for assignment in result['assignments'][:3]:
        print(f"  - \"{assignment.text[:50]}...\"")
        cat = next(c for c in result['categories'] if c.id == assignment.category_id)
        print(f"    → {cat.name} (confidence: {assignment.confidence:.3f})")
    
    return result


# ==============================================================================
# EXAMPLE 2: Configuration Comparison
# ==============================================================================

async def example_2_configuration_comparison():
    """
    Compare different configurations on the same dataset.
    
    Use Case: Finding the best configuration for your specific use case.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Configuration Comparison")
    print("="*70)
    
    # Sample texts with some ambiguous cases
    texts = [
        "Apple stock rises on strong iPhone sales",
        "New Apple variety discovered in Amazon rainforest",
        "Microsoft releases new Windows update",
        "Ancient windows discovered in Roman ruins",
        "Tesla announces new electric vehicle model",
        "Scientists discover electricity generation in marine bacteria",
    ]
    
    # Test different configurations
    configs = {
        "Balanced Hybrid": CONFIG_BALANCED_HYBRID,
        "Conservative Hybrid": CONFIG_CONSERVATIVE_HYBRID,
        "Aggressive Hybrid": CONFIG_AGGRESSIVE_HYBRID,
        "Semantic BERT": CONFIG_SEMANTIC_BERT,
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n[{name}]")
        print("-" * 70)
        
        result = await clusterize_texts(
            texts=texts,
            config=config,
            max_passes=2,
            batch_size=6
        )
        
        results[name] = result
        
        print(f"  Categories: {len(result['categories'])}")
        print(f"  Avg Confidence: {result['metadata']['confidence_stats']['average']:.3f}")
        print(f"  High Confidence (≥0.75): {result['metadata']['confidence_stats']['high_confidence_count']}")
        
        # Show category names
        cat_names = [c.name for c in result['categories']]
        print(f"  Category Names: {', '.join(cat_names)}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS:")
    print("="*70)
    print("- Conservative: Fewer categories, higher thresholds (70%), safer merges")
    print("- Balanced: Good trade-off between precision and recall (60%)")
    print("- Aggressive: More categories, lower thresholds (50%), catches nuances")
    print("- Semantic: Relies heavily on BERT, best for semantic distinctions")
    
    return results


# ==============================================================================
# EXAMPLE 3: Custom Configuration
# ==============================================================================

async def example_3_custom_configuration():
    """
    Create a custom configuration tailored to specific needs.
    
    Use Case: When preset configurations don't match your requirements.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Configuration")
    print("="*70)
    
    # Create custom confidence weights
    # Scenario: Trust LLM heavily, less weight on TF-IDF
    custom_weights = ConfidenceWeights(
        llm_score=0.50,          # Trust LLM judgement
        tfidf_similarity=0.05,    # Less weight on lexical matching
        bert_similarity=0.30,     # Good semantic understanding
        keyword_overlap=0.10,     # Some keyword consideration
        category_maturity=0.03,   # Slight preference for mature categories
        pass_number=0.02          # Minimal pass bonus
    )
    
    # Create custom configuration
    custom_config = ClusterizerConfig(
        retrieval_mode=RETRIEVAL_MODE.HYBRID,
        confidence_weights=custom_weights,
        scoring_strategy=SCORING_STRATEGY.BALANCED,
        new_category_threshold=0.55,  # Moderate threshold
        new_category_bonus=1.15        # Slight bonus for new categories
    )
    
    print(f"\nCustom Configuration:")
    print(f"  {custom_config.get_description()}")
    print(f"\nWeight Distribution:")
    weights_dict = custom_weights.to_dict()
    for key, value in weights_dict.items():
        print(f"  - {key}: {value:.0%}")
    
    # Test with sample texts
    texts = [
        "Machine learning model achieves breakthrough in image recognition",
        "Neural networks power new translation system",
        "AI system beats humans at complex strategy game",
    ]
    
    result = await clusterize_texts(
        texts=texts,
        config=custom_config,
        max_passes=1
    )
    
    print(f"\nResults:")
    print(f"  Categories: {len(result['categories'])}")
    print(f"  Avg Confidence: {result['metadata']['confidence_stats']['average']:.3f}")
    
    return result


# ==============================================================================
# EXAMPLE 4: Chunked Categorization for Large Texts
# ==============================================================================

async def example_4_chunked_categorization():
    """
    Handle large documents with automatic chunking.
    
    Use Case: Long documents, research papers, articles.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Chunked Categorization for Large Texts")
    print("="*70)
    
    # Create a large multi-topic document
    large_document = """
    The global financial markets experienced significant volatility today as the 
    Federal Reserve announced its latest interest rate decision. Stock indices 
    across major exchanges responded with mixed reactions. Technology stocks led 
    the decline with major companies seeing drops of 2-3%, while energy sector 
    stocks gained ground. Bond yields rose to their highest levels in months, 
    reflecting investor concerns about inflation. Currency markets also showed 
    movement, with the dollar strengthening against major currencies. Analysts 
    predict continued uncertainty in the coming weeks as economic data is released.
    
    In breaking science news, marine biologists have made a groundbreaking discovery 
    in the deep ocean. A new species of bioluminescent octopus was found near 
    hydrothermal vents in the Pacific Ocean at depths exceeding 3,000 meters. 
    The creature possesses unique adaptations for extreme pressure and darkness, 
    including specialized light-producing organs and enhanced sensory capabilities. 
    Researchers used advanced remotely operated vehicles equipped with high-definition 
    cameras to document the species. The discovery sheds light on how life adapts 
    to the most extreme environments on Earth. Scientists believe there may be many 
    more undiscovered species in the deep ocean trenches.
    
    Meanwhile, the technology sector continues its rapid evolution with several 
    major announcements. A leading smartphone manufacturer unveiled its latest 
    flagship device featuring revolutionary camera technology and improved AI 
    capabilities. The new phone includes advanced computational photography, 
    enabling users to capture professional-quality images in various lighting 
    conditions. Battery life has been extended through more efficient processors 
    and power management systems. The device also introduces new augmented reality 
    features that could transform how users interact with their environment. 
    Industry analysts expect strong sales during the holiday season.
    """
    
    # Small texts for comparison
    small_texts = [
        "NASA announces new Mars rover mission",
        "Streaming service raises subscription prices",
    ]
    
    # Combine small and large texts
    all_texts = small_texts + [large_document]
    
    print(f"\nProcessing {len(all_texts)} texts:")
    print(f"  - 2 small texts (~10-20 tokens)")
    print(f"  - 1 large document (~350 tokens)")
    
    # Use chunked categorization
    result = await clusterize_texts_with_chunking(
        texts=all_texts,
        large_text_threshold=100,  # Chunk texts with 100+ tokens
        chunk_size=150,             # ~100 words per chunk
        chunk_overlap=30,           # 30 tokens overlap
        multi_topic_threshold=0.25, # 25% votes = multi-topic
        config=CONFIG_BALANCED_HYBRID,
        max_passes=2
    )
    
    print(f"\nChunking Statistics:")
    print(f"  Texts chunked: {result['metadata']['texts_chunked']}")
    print(f"  Total chunks created: {result['metadata']['total_chunks_created']}")
    print(f"  Processing units: {result['metadata']['processing_units']}")
    
    print(f"\nResults:")
    print(f"  Categories: {len(result['categories'])}")
    print(f"  Assignments: {len(result['assignments'])}")
    
    # Show multi-topic detection
    if result['multi_topic_texts']:
        print(f"\nMulti-Topic Documents Detected: {len(result['multi_topic_texts'])}")
        for mt in result['multi_topic_texts']:
            print(f"\n  Text {mt['text_index']}:")
            print(f"    Preview: {mt['text'][:80]}...")
            print(f"    Primary: {mt['primary_category']}")
            print(f"    All topics: {', '.join(mt['categories'])}")
            print(f"    Confidence: {mt['confidence']:.3f}")
    
    # Show how large document was categorized
    large_doc_assignment = result['assignments'][-1]  # Last one is the large doc
    print(f"\nLarge Document Assignment:")
    print(f"  Category: {large_doc_assignment.category_id}")
    print(f"  Confidence: {large_doc_assignment.confidence:.3f}")
    print(f"  Reasoning: {large_doc_assignment.reasoning}")
    
    return result


# ==============================================================================
# EXAMPLE 5: Detecting Duplicate Categories
# ==============================================================================

async def example_5_duplicate_detection():
    """
    Demonstrate SmartConsolidator detecting and merging similar categories.
    
    Use Case: Preventing category fragmentation with semantic duplicates.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Duplicate Category Detection")
    print("="*70)
    
    # Texts designed to create similar but slightly different categories
    texts = [
        "Stock market indices rise on positive economic data",
        "Financial markets respond to interest rate changes",
        "Wall Street sees gains in technology sector",
        "Investment portfolios show strong returns",
        "Economic indicators suggest market growth",
        "Finance and banking sector shows resilience",
        "Markets and trading activity reaches new highs",
        "Financial services industry expands operations",
    ]
    
    print("\nThese texts might create duplicate categories like:")
    print("  - 'Finance and Markets'")
    print("  - 'Finance and Economics'")
    print("  - 'Financial Services'")
    print("  - 'Markets and Trading'")
    
    print("\nRunning with SmartConsolidator enabled...")
    
    result = await clusterize_texts(
        texts=texts,
        config=CONFIG_BALANCED_HYBRID,
        max_passes=2,
        batch_size=4
    )
    
    print(f"\nResults After Consolidation:")
    print(f"  Final categories: {len(result['categories'])}")
    
    for cat in result['categories']:
        print(f"\n  Category: {cat.name}")
        print(f"    Description: {cat.description}")
        print(f"    Keywords: {', '.join(cat.keywords[:8])}")
        print(f"    Texts: {cat.text_count}")
    
    print("\n" + "-"*70)
    print("How SmartConsolidator Works:")
    print("-"*70)
    print("1. TF-IDF Similarity: Lexical matching (threshold: 0.45)")
    print("2. BERT Similarity: Semantic matching (threshold: 0.70)")
    print("3. Keyword Overlap: Shared keywords (threshold: 0.40)")
    print("4. Name Similarity: Word overlap in names (threshold: 0.60)")
    print("5. Combined Score: Weighted aggregate (threshold: 0.55)")
    print("\nAuto-approves merges when:")
    print("  - BERT similarity ≥ 0.75, OR")
    print("  - Combined score ≥ 0.70 (aggressive) / 0.75 (conservative)")
    
    return result


# ==============================================================================
# EXAMPLE 6: Adaptive Top-K in Action
# ==============================================================================

async def example_6_adaptive_topk():
    """
    Demonstrate how AdaptiveTopKSelector adjusts candidate retrieval.
    
    Use Case: Better accuracy with context-aware candidate selection.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Adaptive Top-K Selection")
    print("="*70)
    
    # Create many categories to demonstrate adaptive K
    texts = [
        # Technology (5 texts)
        "Apple releases new MacBook with M3 chip",
        "Google unveils Gemini AI model",
        "Microsoft announces Windows 12",
        "Samsung launches Galaxy S24",
        "Tesla Model Y breaks sales records",
        
        # Finance (5 texts)
        "Stock market closes at record high",
        "Bitcoin reaches new valuation peak",
        "Federal Reserve adjusts interest rates",
        "Banking sector reports strong earnings",
        "Investment funds see increased inflows",
        
        # Science (5 texts)
        "CERN discovers new particle",
        "Mars rover finds organic compounds",
        "Quantum computer achieves breakthrough",
        "New vaccine shows promising results",
        "Climate study reveals warming trends",
        
        # Sports (5 texts)
        "World Cup final draws record viewers",
        "Olympic athlete breaks world record",
        "Championship game goes to overtime",
        "Tennis tournament crowns new champion",
        "Marathon runner sets new record time",
        
        # Now add ambiguous text that could fit multiple categories
        "Tech stocks soar on AI breakthrough announcement",
    ]
    
    print(f"\nProcessing {len(texts)} texts to create ~4 categories")
    print("Then testing adaptive K on ambiguous text...")
    
    result = await clusterize_texts(
        texts=texts,
        config=CONFIG_BALANCED_HYBRID,
        max_passes=2,
        batch_size=10,
        prefilter_k=3  # Base K
    )
    
    print(f"\nResults:")
    print(f"  Categories: {len(result['categories'])}")
    
    print("\nHow Adaptive K Works:")
    print("-"*70)
    print("Base K = 3, but adjusts based on:")
    print("  1. Pass Number: Pass 2 gets +50% → K=4-5")
    print("  2. Category Count: 20+ categories gets +2 → K=5-6")
    print("  3. Text Complexity: Long/complex texts get +1 → K=6-7")
    print("  4. Creating New: New category mode gets +50% → K=7-10")
    print(f"\nWith {len(result['categories'])} categories:")
    print(f"  - Pass 1, simple text: K ≈ 3-4")
    print(f"  - Pass 2, complex text: K ≈ 5-6")
    print(f"  - Creating new category: K ≈ 7-8")
    
    # Show the ambiguous text assignment
    ambiguous_assignment = result['assignments'][-1]
    cat = next(c for c in result['categories'] if c.id == ambiguous_assignment.category_id)
    
    print(f"\nAmbiguous Text Assignment:")
    print(f"  Text: \"{ambiguous_assignment.text}\"")
    print(f"  Category: {cat.name}")
    print(f"  Confidence: {ambiguous_assignment.confidence:.3f}")
    print(f"  (Higher K means more candidates considered for better accuracy)")
    
    return result


# ==============================================================================
# EXAMPLE 7: Real-World Use Case - News Categorization
# ==============================================================================

async def example_7_news_categorization():
    """
    Complete real-world example: categorizing news articles.
    
    Use Case: Production news categorization system.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Real-World News Categorization")
    print("="*70)
    
    # Mix of short headlines and longer article snippets
    news_texts = [
        # Tech short
        "Apple's new AI features boost iPhone sales",
        
        # Tech long
        """Tech giant Microsoft announced its latest quarterly earnings today, 
        surpassing analyst expectations with strong cloud computing revenue. 
        The company's Azure platform continues to gain market share against 
        competitors, driven by increasing enterprise adoption of AI services.""",
        
        # Finance short
        "Stock markets rally on positive jobs report",
        
        # Finance long  
        """The Federal Reserve signaled potential changes to its monetary policy 
        stance following recent economic data. Inflation rates have shown signs 
        of moderating, while employment numbers remain robust. Investors are 
        closely watching for signals about future interest rate decisions.""",
        
        # Sports short
        "Olympic swimmer sets new world record",
        
        # Science short
        "New exoplanet discovered in habitable zone",
        
        # Science long
        """Researchers at a leading university have developed a groundbreaking 
        treatment for a rare genetic disorder. The gene therapy approach showed 
        remarkable success in clinical trials, offering hope to thousands of 
        patients worldwide. The treatment works by correcting the underlying 
        genetic mutation responsible for the condition.""",
        
        # Health
        "WHO reports progress in malaria vaccine distribution",
        
        # Environment
        "Renewable energy capacity reaches new milestone globally",
        
        # Politics
        "International summit addresses climate change commitments",
    ]
    
    print(f"\nProcessing {len(news_texts)} news items...")
    print("  - Mix of short headlines and longer articles")
    print("  - Using chunking for texts >100 tokens")
    
    result = await clusterize_texts_with_chunking(
        texts=news_texts,
        large_text_threshold=100,
        chunk_size=150,
        chunk_overlap=30,
        config=CONFIG_BALANCED_HYBRID,
        max_passes=2,
        batch_size=5
    )
    
    print(f"\nResults:")
    print(f"  Categories: {len(result['categories'])}")
    print(f"  Texts chunked: {result['metadata']['texts_chunked']}")
    print(f"  Multi-topic: {result['metadata']['multi_topic_count']}")
    
    print(f"\nNews Categories Created:")
    for cat in result['categories']:
        print(f"\n  📁 {cat.name.upper()}")
        print(f"     {cat.description}")
        print(f"     Articles: {cat.text_count}")
        print(f"     Keywords: {', '.join(cat.keywords[:6])}")
        
        # Show articles in this category
        articles = [a for a in result['assignments'] if a.category_id == cat.id]
        for article in articles[:2]:  # Show first 2
            preview = article.text[:60] + "..." if len(article.text) > 60 else article.text
            print(f"       • {preview} (conf: {article.confidence:.2f})")
    
    # Confidence distribution
    print(f"\nConfidence Distribution:")
    stats = result['metadata']['confidence_stats']
    print(f"  High (≥0.75): {stats['high_confidence_count']} articles")
    print(f"  Medium (0.50-0.74): {stats['medium_confidence_count']} articles")
    print(f"  Low (<0.50): {stats['low_confidence_count']} articles")
    
    return result


# ==============================================================================
# EXAMPLE 8: Comparison - With vs Without Chunking
# ==============================================================================

async def example_8_chunking_comparison():
    """
    Compare results with and without chunking for large texts.
    
    Use Case: Understanding the impact of chunking.
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: With vs Without Chunking")
    print("="*70)
    
    # Create a document that spans multiple topics
    multi_topic_doc = """
    The cryptocurrency market experienced dramatic swings today as Bitcoin 
    reached a new all-time high before retreating. Ethereum and other 
    altcoins followed similar patterns. Institutional investors continue 
    to show strong interest in digital assets despite regulatory uncertainties.
    
    In completely unrelated news, marine biologists studying coral reefs 
    have discovered a new symbiotic relationship between certain fish species 
    and coral polyps. This discovery could help in coral conservation efforts.
    
    Meanwhile, tech companies are racing to develop more efficient quantum 
    computing systems. Recent breakthroughs in error correction bring us 
    closer to practical quantum computers for everyday applications.
    """
    
    # Test WITHOUT chunking (will likely pick just one category)
    print("\n[WITHOUT CHUNKING]")
    print("-" * 70)
    
    result_no_chunk = await clusterize_texts(
        texts=[multi_topic_doc],
        config=CONFIG_BALANCED_HYBRID,
        max_passes=1
    )
    
    assignment = result_no_chunk['assignments'][0]
    cat = result_no_chunk['categories'][0]
    
    print(f"Result: Assigned to single category")
    print(f"  Category: {cat.name}")
    print(f"  Confidence: {assignment.confidence:.3f}")
    print(f"  Problem: May miss other topics in the document")
    
    # Test WITH chunking (should detect multiple topics)
    print("\n[WITH CHUNKING]")
    print("-" * 70)
    
    result_with_chunk = await clusterize_texts_with_chunking(
        texts=[multi_topic_doc],
        large_text_threshold=50,   # Low threshold to force chunking
        chunk_size=100,
        chunk_overlap=20,
        multi_topic_threshold=0.2,  # More sensitive
        config=CONFIG_BALANCED_HYBRID,
        max_passes=1
    )
    
    print(f"Chunks created: {result_with_chunk['metadata']['total_chunks_created']}")
    
    if result_with_chunk['multi_topic_texts']:
        mt = result_with_chunk['multi_topic_texts'][0]
        print(f"Result: Multi-topic document detected! ✓")
        print(f"  Primary: {mt['primary_category']}")
        print(f"  All topics: {', '.join(mt['categories'])}")
        print(f"  Benefit: Captures full topical diversity")
    else:
        print(f"Result: Single topic detected")
        print(f"  (May need lower multi_topic_threshold)")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("✓ Use chunking for:")
    print("  - Documents >800 tokens")
    print("  - Multi-topic documents")
    print("  - Research papers, articles, reports")
    print("\n✓ Skip chunking for:")
    print("  - Short texts (tweets, headlines)")
    print("  - Single-topic content")
    print("  - When speed is critical")
    
    return result_with_chunk


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

async def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "="*70)
    print("AGENTIC CLUSTERIZER - COMPREHENSIVE EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate:")
    print("  1. Basic usage with preset configurations")
    print("  2. Configuration comparison")
    print("  3. Custom configuration creation")
    print("  4. Chunked categorization for large texts")
    print("  5. Duplicate category detection (SmartConsolidator)")
    print("  6. Adaptive top-K selection")
    print("  7. Real-world news categorization")
    print("  8. Chunking comparison")
    
    # Run examples (comment out any you don't want to run)
    await example_1_basic_usage()
    await example_2_configuration_comparison()
    await example_3_custom_configuration()
    await example_4_chunked_categorization()
    await example_5_duplicate_detection()
    await example_6_adaptive_topk()
    await example_7_news_categorization()
    await example_8_chunking_comparison()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nFor more information, see:")
    print("  - README.md for API documentation")
    print("  - agentic_clusterizer.py for implementation details")
    print("  - integration_guide.md for advanced use cases")


if __name__ == '__main__':
    # Run a specific example
    # asyncio.run(example_1_basic_usage())
    
    # Or run all examples
    asyncio.run(run_all_examples())
