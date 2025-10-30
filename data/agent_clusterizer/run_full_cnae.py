#!/usr/bin/env python3
"""
Optimized production-ready script for CNAE dataset processing.

Key improvements:
1. Adaptive parameters based on dataset size
2. Progress tracking and checkpointing
3. Better error handling and recovery
4. Detailed performance metrics
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from configuration_manager import ClusterConfig
from agentic_clusterizer import clusterize_texts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CNAEProcessor:
    """Intelligent CNAE dataset processor with adaptive configuration."""
    
    def __init__(
        self, 
        input_file: str,
        output_dir: str = 'results',
        sample_size: Optional[int] = None
    ):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sample_size = sample_size
        
        self.texts = []
        self.result = None
        self.start_time = None
        self.end_time = None
    
    def load_texts(self) -> int:
        """Load texts from input file."""
        logger.info(f"📂 Loading texts from: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()][:self.sample_size]
        
        logger.info(f"✓ Loaded {len(self.texts)} texts")
        return len(self.texts)
        
    async def process(self, config: Optional[ClusterConfig] = None) -> Dict[str, Any]:
        """Execute clustering with progress tracking."""
        logger.info("🚀 Starting clustering process...")
        
        self.start_time = datetime.now()
        
        try:
            config = config or ClusterConfig.from_texts(self.texts)
            self.result = await clusterize_texts(self.texts, config=config)
            
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            
            logger.info(f"\n🎉 SUCCESS! Clustering complete in {duration:.1f}s")
            
            return self.result
            
        except Exception as e:
            self.end_time = datetime.now()
            logger.error(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def print_results_summary(self):
        """Print detailed results summary."""
        if not self.result:
            logger.warning("No results to display")
            return
        
        metadata = self.result['metadata']
        categories = self.result['categories']
        assignments = self.result['assignments']
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 FINAL RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"\n📥 Input:")
        logger.info(f"  - Total texts: {len(self.texts)}")
        logger.info(f"  - Processing time: {duration:.1f}s ({duration/len(self.texts):.2f}s per text)")
        
        logger.info(f"\n📤 Output:")
        logger.info(f"  - Categories created: {len(categories)}")
        logger.info(f"  - Assignments: {len(assignments)}")
        logger.info(f"  - Compression ratio: {len(self.texts) / len(categories):.1f}:1")
        
        logger.info(f"\n⚙️ Processing Details:")
        logger.info(f"  - Mode: {metadata.get('processing_mode', 'standard')}")
        logger.info(f"  - Tree merge: {metadata.get('used_tree_merge', False)}")
        
        if metadata.get('used_tree_merge'):
            logger.info(f"  - Total workflows: {metadata.get('total_workflows', 'N/A')}")
            logger.info(f"  - Texts per workflow: {metadata.get('texts_per_workflow', 'N/A')}")
            logger.info(f"  - Big texts consolidated: {metadata.get('big_texts_consolidated', 0)}")
        
        logger.info(f"  - Total passes: {metadata.get('total_passes', 'N/A')}")
        logger.info(f"  - Batch size: {metadata.get('batch_size', 'N/A')}")
        
        # Confidence statistics
        conf_stats = metadata.get('confidence_stats', {})
        if conf_stats:
            logger.info(f"\n📈 Confidence Statistics:")
            logger.info(f"  - Average: {conf_stats.get('average', 0):.3f}")
            logger.info(f"  - High confidence (≥0.75): {conf_stats.get('high_confidence_count', 0)} texts")
            logger.info(f"  - Medium confidence (0.5-0.75): {conf_stats.get('medium_confidence_count', 0)} texts")
            logger.info(f"  - Low confidence (<0.5): {conf_stats.get('low_confidence_count', 0)} texts")
        
        # Top categories
        logger.info(f"\n🏆 Top 10 Categories by Size:")
        sorted_categories = sorted(
            categories, 
            key=lambda c: c.text_count, 
            reverse=True
        )[:10]
        
        for i, cat in enumerate(sorted_categories, 1):
            logger.info(f"  {i:2d}. {cat.name} ({cat.text_count} texts)")
            if cat.keywords:
                keywords_str = ', '.join(cat.keywords[:5])
                logger.info(f"      Keywords: {keywords_str}")
        
        logger.info(f"\n{'='*60}\n")
    
    def save_results(self):
        """Save results to multiple formats."""
        if not self.result:
            logger.warning("No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"cnae_{len(self.texts)}texts_{timestamp}"
        
        # 1. Save full JSON
        json_file = self.output_dir / f"{base_name}.json"
        serializable_result = self._make_serializable(self.result)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Saved full results: {json_file}")
        
        # 2. Save categories CSV
        csv_file = self.output_dir / f"{base_name}_categories.csv"
        self._save_categories_csv(csv_file)
        logger.info(f"✓ Saved categories: {csv_file}")
        
        # 3. Save assignments CSV
        assignments_file = self.output_dir / f"{base_name}_assignments.csv"
        self._save_assignments_csv(assignments_file)
        logger.info(f"✓ Saved assignments: {assignments_file}")
        
        # 4. Save summary report
        report_file = self.output_dir / f"{base_name}_report.txt"
        self._save_report(report_file)
        logger.info(f"✓ Saved report: {report_file}")
        
        logger.info(f"\n📁 All results saved to: {self.output_dir}/")
    
    def _make_serializable(self, result: Dict) -> Dict:
        """Convert result objects to JSON-serializable format."""
        return {
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
    
    def _save_categories_csv(self, filepath: Path):
        """Save categories to CSV."""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Name', 'Description', 'Text Count', 'Keywords'])
            
            for cat in self.result['categories']:
                writer.writerow([
                    cat.id,
                    cat.name,
                    cat.description,
                    cat.text_count,
                    ', '.join(cat.keywords) if cat.keywords else ''
                ])
    
    def _save_assignments_csv(self, filepath: Path):
        """Save assignments to CSV."""
        import csv
        
        # Build category lookup
        cat_lookup = {cat.id: cat.name for cat in self.result['categories']}
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Text', 'Category ID', 'Category Name', 'Confidence', 'Reasoning'])
            
            for assign in self.result['assignments']:
                writer.writerow([
                    assign.text,
                    assign.category_id,
                    cat_lookup.get(assign.category_id, 'Unknown'),
                    f"{assign.confidence:.3f}",
                    assign.reasoning
                ])
    
    def _save_report(self, filepath: Path):
        """Save detailed text report."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("CNAE CLUSTERING ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            
            f.write(f"SUMMARY\n")
            f.write(f"-" * 60 + "\n")
            f.write(f"Total input texts: {len(self.texts)}\n")
            f.write(f"Categories created: {len(self.result['categories'])}\n")
            f.write(f"Processing time: {(self.end_time - self.start_time).total_seconds():.1f}s\n\n")
            
            f.write(f"CATEGORIES (sorted by size)\n")
            f.write(f"-" * 60 + "\n")
            
            sorted_cats = sorted(
                self.result['categories'],
                key=lambda c: c.text_count,
                reverse=True
            )
            
            for cat in sorted_cats:
                f.write(f"\n{cat.name} (ID: {cat.id})\n")
                f.write(f"  Texts: {cat.text_count}\n")
                if cat.description:
                    f.write(f"  Description: {cat.description}\n")
                if cat.keywords:
                    f.write(f"  Keywords: {', '.join(cat.keywords)}\n")


async def main():
    """Main execution flow with interactive prompts."""
    logger.info("="*60)
    logger.info("CNAE DATASET CLUSTERING")
    logger.info("="*60)
    logger.info()
    logger.info("This tool will:")
    logger.info("  1. Load your CNAE dataset")
    logger.info("  2. Analyze dataset size and select optimal configuration")
    logger.info("  3. Process using LLM-powered hierarchical clustering")
    logger.info("  4. Save results in multiple formats")
    logger.info()
    
    # Initialize processor
    filename='cnae.txt'
    sample_size=10
    processor = CNAEProcessor(filename, sample_size=sample_size)
    config = ClusterConfig.cost_optimized()
    
    # Load texts
    try:
        num_texts = processor.load_texts()
    except FileNotFoundError:
        logger.info(f"❌ Error: File '{processor.input_file}' not found!")
        logger.info("   Please ensure cnae.txt exists in the current directory.")
        return
    
    if num_texts == 0:
        logger.info("❌ Error: No texts found in file!")
        return
    
    # Confirm processing
    logger.info("\n⚠️  This will make LLM API calls and may take time.")
    logger.info(f"   Estimated duration: {num_texts * 0.5 / 60:.1f}-{num_texts * 2 / 60:.1f} minutes")
    logger.info()
    
    # Process
    try:
        await processor.process(config)
        
        # Display results
        processor.print_results_summary()
        
        # Save results
        processor.save_results()
        
        logger.info("✅ All done! Check the 'results/' directory for outputs.")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Processing interrupted by user.")
    except Exception as e:
        logger.info(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())