"""
Before/After Comparison Demo

Demonstrates the improvements from the enhanced clusterizer:
1. Category consolidation (6 → 4 categories expected)
2. Better confidence scores with BERT
3. Multi-topic detection for large texts
"""

import asyncio
import logging
from typing import Dict, Any, List
from collections import Counter

# Assuming the enhanced clusterizer module is available
from agentic_clusterizer import (
    clusterize_texts,
    clusterize_texts_with_chunking,
    CONFIG_BALANCED_HYBRID,
    CONFIG_CONSERVATIVE_HYBRID,
    CONFIG_AGGRESSIVE_HYBRID,
    CONFIG_SEMANTIC_BERT,
    Category,
    CategoryAssignment
)

logger = logging.getLogger("comparison_demo")
logging.basicConfig(level=logging.INFO)

# Your original sample texts
SAMPLE_TEXTS = [
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

# Add a large multi-topic text to demonstrate chunking
LARGE_MULTI_TOPIC_TEXT = """
The Federal Reserve's recent interest rate decision has sent shockwaves through 
global financial markets, with investors reassessing their portfolios in light 
of persistent inflation concerns. Technology stocks bore the brunt of the 
sell-off, with major indices posting their worst performance in months. Market 
analysts suggest this could signal a prolonged period of volatility as central 
banks worldwide grapple with balancing economic growth and price stability.

In unrelated scientific news, marine biologists have made a groundbreaking 
discovery in the depths of the Pacific Ocean. A new species of bioluminescent 
jellyfish was found near hydrothermal vents at extreme depths, surviving in 
conditions previously thought impossible for complex life. The discovery 
highlights the incredible biodiversity that exists in Earth's least explored 
ecosystems and raises important questions about adaptation to extreme environments.
Researchers from Stanford University led the expedition using advanced deep-sea 
robotics and discovered not just the jellyfish but also three new species of 
tube worms.

Meanwhile, space agencies around the world are collaborating on an ambitious 
mission to explore Saturn's largest moon, Titan. The mission will deploy 
advanced robotic explorers to study Titan's methane lakes and thick atmosphere, 
searching for signs of prebiotic chemistry that could shed light on the origins 
of life in our solar system. NASA has committed $2 billion to the project, with 
ESA contributing additional instrumentation. Launch is scheduled for 2028.
"""

def print_section(title: str, width: int = 70):
    """Print a formatted section header."""
    print(f"\n{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}\n")

def analyze_results(result: Dict[str, Any], label: str):
    """Analyze and print results summary."""
    categories = result['categories']
    assignments = result['assignments']
    metadata = result['metadata']
    
    print_section(f"{label} - SUMMARY")
    
    print(f"📊 Overview:")
    print(f"  • Total texts: {metadata['total_texts']}")
    print(f"  • Total categories: {metadata['total_categories']}")
    print(f"  • Configuration: {metadata['config']['retrieval_mode']} / {metadata['config']['scoring_strategy']}")
    print(f"  • Processing time: {metadata.get('elapsed_time', 'N/A')}")
    
    print(f"\n📁 Categories:")
    for i, cat in enumerate(categories, 1):
        print(f"  {i}. {cat.name} ({cat.id})")
        print(f"     Texts: {cat.text_count} | Keywords: {', '.join(cat.keywords[:5])}")
    
    # Confidence statistics
    conf_stats = metadata['confidence_stats']
    print(f"\n📈 Confidence Distribution:")
    print(f"  • Average: {conf_stats['average']:.3f}")
    print(f"  • Range: {conf_stats['min']:.3f} - {conf_stats['max']:.3f}")
    print(f"  • High (≥0.75): {conf_stats['high_confidence_count']} ({conf_stats['high_confidence_count']/len(assignments)*100:.1f}%)")
    print(f"  • Medium (0.5-0.75): {conf_stats['medium_confidence_count']} ({conf_stats['medium_confidence_count']/len(assignments)*100:.1f}%)")
    print(f"  • Low (<0.5): {conf_stats['low_confidence_count']} ({conf_stats['low_confidence_count']/len(assignments)*100:.1f}%)")
    
    # Category distribution
    cat_counts = Counter(a.category_id for a in assignments)
    print(f"\n📊 Texts per Category:")
    for cat_id, count in cat_counts.most_common():
        cat_name = next((c.name for c in categories if c.id == cat_id), cat_id)
        print(f"  • {cat_name}: {count} texts")

def compare_category_merging(before: Dict[str, Any], after: Dict[str, Any]):
    """Compare category consolidation between runs."""
    print_section("CATEGORY CONSOLIDATION ANALYSIS")
    
    before_cats = before['categories']
    after_cats = after['categories']
    
    print(f"Before: {len(before_cats)} categories")
    for cat in before_cats:
        print(f"  • {cat.name}")
    
    print(f"\nAfter: {len(after_cats)} categories")
    for cat in after_cats:
        print(f"  • {cat.name}")
    
    if len(after_cats) < len(before_cats):
        print(f"\n✅ SUCCESS: Merged {len(before_cats) - len(after_cats)} duplicate categories!")
        print(f"   Consolidation rate: {(1 - len(after_cats)/len(before_cats))*100:.1f}%")
    else:
        print(f"\n⚠️  No categories were merged (this might be OK depending on data)")

def show_detailed_assignments(result: Dict[str, Any], max_show: int = 5):
    """Show detailed assignment information."""
    assignments = sorted(result['assignments'], key=lambda a: a.confidence, reverse=True)
    categories = {c.id: c for c in result['categories']}
    
    print_section("DETAILED ASSIGNMENTS (Top 5)")
    
    for i, assignment in enumerate(assignments[:max_show], 1):
        cat = categories.get(assignment.category_id)
        cat_name = cat.name if cat else assignment.category_id
        
        # Confidence indicator
        if assignment.confidence >= 0.75:
            indicator = "🟢 HIGH"
        elif assignment.confidence >= 0.5:
            indicator = "🟡 MEDIUM"
        else:
            indicator = "🔴 LOW"
        
        print(f"{i}. {indicator} | Confidence: {assignment.confidence:.3f}")
        print(f"   Text: {assignment.text[:70]}...")
        print(f"   Category: {cat_name}")
        print(f"   Reasoning: {assignment.reasoning}")
        print()

async def run_comparison():
    """Run full before/after comparison."""
    
    print_section("ENHANCED CLUSTERIZER COMPARISON DEMO", 80)
    print("This demo compares different configurations to show improvements.\n")
    
    # Test 1: Conservative vs Balanced (without large text)
    print_section("TEST 1: STANDARD TEXTS - Conservative vs Balanced")
    
    print("Running CONSERVATIVE configuration...")
    result_conservative = await clusterize_texts(
        SAMPLE_TEXTS,
        max_passes=2,
        batch_size=6,
        config=CONFIG_CONSERVATIVE_HYBRID
    )
    analyze_results(result_conservative, "CONSERVATIVE")
    
    print("\n" + "─"*70 + "\n")
    
    print("Running BALANCED configuration...")
    result_balanced = await clusterize_texts(
        SAMPLE_TEXTS,
        max_passes=2,
        batch_size=6,
        config=CONFIG_BALANCED_HYBRID
    )
    analyze_results(result_balanced, "BALANCED")
    
    # Compare merging effectiveness
    compare_category_merging(result_conservative, result_balanced)
    
    # Show detailed assignments
    show_detailed_assignments(result_balanced)
    
    # Test 2: Semantic BERT only
    print_section("TEST 2: SEMANTIC BERT (Heavy BERT Weighting)")
    
    print("Running SEMANTIC BERT configuration...")
    result_semantic = await clusterize_texts(
        SAMPLE_TEXTS,
        max_passes=2,
        batch_size=6,
        config=CONFIG_SEMANTIC_BERT
    )
    analyze_results(result_semantic, "SEMANTIC BERT")
    show_detailed_assignments(result_semantic, max_show=3)
    
    # Test 3: Chunked categorization with multi-topic detection
    print_section("TEST 3: CHUNKED CATEGORIZATION (Large Multi-Topic Text)")
    
    texts_with_large = SAMPLE_TEXTS + [LARGE_MULTI_TOPIC_TEXT]
    
    print("Running with chunking enabled...")
    result_chunked = await clusterize_texts_with_chunking(
        texts_with_large,
        large_text_threshold=100,  # Lower for demo (normally 800)
        chunk_size=150,            # Smaller for demo (normally 500)
        multi_topic_threshold=0.25,
        max_passes=2,
        batch_size=6,
        config=CONFIG_BALANCED_HYBRID
    )
    
    print(f"\n📊 Chunking Statistics:")
    metadata = result_chunked['metadata']
    print(f"  • Texts chunked: {metadata['texts_chunked']}")
    print(f"  • Total chunks created: {metadata['total_chunks_created']}")
    print(f"  • Processing units: {metadata['processing_units']}")
    print(f"  • Multi-topic texts found: {metadata['multi_topic_count']}")
    
    # Show multi-topic texts
    multi_topic_texts = result_chunked.get('multi_topic_texts', [])
    if multi_topic_texts:
        print(f"\n🔍 Multi-Topic Documents:")
        for mt in multi_topic_texts:
            print(f"\n  Text {mt['text_index']}:")
            print(f"    Preview: {mt['text'][:80]}...")
            print(f"    Primary: {mt['primary_category']}")
            print(f"    All categories: {', '.join(mt['categories'])}")
            print(f"    Confidence: {mt['confidence']:.3f}")
    
    # Test 4: Configuration comparison table
    print_section("TEST 4: CONFIGURATION COMPARISON TABLE")
    
    configs_to_test = [
        ("Conservative Hybrid", CONFIG_CONSERVATIVE_HYBRID),
        ("Balanced Hybrid", CONFIG_BALANCED_HYBRID),
        ("Aggressive Hybrid", CONFIG_AGGRESSIVE_HYBRID),
        ("Semantic BERT", CONFIG_SEMANTIC_BERT),
    ]
    
    comparison_results = []
    
    for name, config in configs_to_test:
        print(f"Testing {name}...")
        result = await clusterize_texts(
            SAMPLE_TEXTS[:6],  # Use subset for speed
            max_passes=1,
            batch_size=6,
            config=config
        )
        
        comparison_results.append({
            'name': name,
            'categories': len(result['categories']),
            'avg_confidence': result['metadata']['confidence_stats']['average'],
            'high_conf_pct': result['metadata']['confidence_stats']['high_confidence_count'] / len(result['assignments']) * 100
        })
    
    print(f"\n{'Configuration':<20} | {'Categories':<12} | {'Avg Conf':<10} | {'High Conf %':<12}")
    print("─"*70)
    for r in comparison_results:
        print(f"{r['name']:<20} | {r['categories']:<12} | {r['avg_confidence']:.3f}      | {r['high_conf_pct']:.1f}%")
    
    # Final recommendations
    print_section("RECOMMENDATIONS")
    
    print("Based on the comparison:")
    print()
    print("1. 🎯 **For most use cases**: Use CONFIG_BALANCED_HYBRID")
    print("   → Good balance of precision and consolidation")
    print("   → BERT helps catch semantic duplicates")
    print()
    print("2. 🛡️  **For established taxonomies**: Use CONFIG_CONSERVATIVE_HYBRID")
    print("   → Avoids creating too many new categories")
    print("   → Prefers existing categories with more texts")
    print()
    print("3. 🚀 **For exploratory analysis**: Use CONFIG_AGGRESSIVE_HYBRID")
    print("   → Creates granular categories")
    print("   → Good for discovering new patterns")
    print()
    print("4. 🧠 **For semantic-heavy domains**: Use CONFIG_SEMANTIC_BERT")
    print("   → Maximum BERT weight (45%)")
    print("   → Best for documents with similar concepts, different words")
    print()
    print("5. 📚 **For large/multi-topic texts**: Use clusterize_texts_with_chunking()")
    print("   → Automatically handles chunking")
    print("   → Detects multi-topic documents")
    print("   → Aggregates with confidence voting")
    
    print_section("COMPARISON COMPLETE", 80)

if __name__ == '__main__':
    asyncio.run(run_comparison())