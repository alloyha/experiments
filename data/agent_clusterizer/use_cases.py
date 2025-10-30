"""
Exhaustive Use Cases for Agentic Text Clusterizer

This document provides comprehensive test scenarios covering:
1. Basic clustering operations
2. Configuration variations
3. Edge cases and error handling
4. Performance scenarios
5. Integration patterns
6. Real-world applications

Each use case includes:
- Description
- Input specification
- Expected behavior
- Configuration recommendations
- Validation criteria
"""

import sys
import asyncio
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from colorama import init, Fore, Style  # pip install colorama


from agentic_clusterizer import (
    clusterize_texts,
    clusterize_texts_large,
    ClusterConfig,
    Category,
    CategoryAssignment,
)
from configuration_manager import Priority

# Set up logging
logger = logging.getLogger(__name__)

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# ============================================================================
# DUAL LOGGING SETUP
# ============================================================================

def setup_dual_logging():
    """
    Setup dual logging system:
    - File handler: All details (DEBUG level)
    - Console handler: Summary only (INFO level)
    """
    
    # Root logger for file output (everything)
    file_logger = logging.getLogger("use_case_file")
    file_logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler("use_case_detailed.log", mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    file_logger.addHandler(file_handler)
    
    # Console logger for summary only
    console_logger = logging.getLogger("use_case_console")
    console_logger.setLevel(logging.INFO)
    console_logger.propagate = False  # Don't propagate to root
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    console_logger.addHandler(console_handler)
    
    return file_logger, console_logger


# Global loggers
logger, console = setup_dual_logging()

# ============================================================================
# CONSOLE SUMMARY FORMATTER
# ============================================================================

class ConsoleSummary:
    """Improved summary with actionable failure information."""
    
    @staticmethod
    def print_header():
        """Print execution header."""
        console.info(Fore.CYAN + "=" * 80)
        console.info(Fore.CYAN + "COMPREHENSIVE USE CASE EXECUTION")
        console.info(Fore.CYAN + "=" * 80)
        console.info("")

    @staticmethod
    def print_progress(current: int, total: int, use_case_id: str, result: 'UseCaseResult'):
        """Print progress with immediate failure insight."""
        percentage = (current / total) * 100
        bar_length = 30
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        if result.passed:
            status_color = Fore.GREEN
            status = "PASS"
            detail = ""
        else:
            status_color = Fore.RED
            status = "FAIL"
            # Show key failure reason inline
            detail = f" - {ConsoleSummary._get_failure_reason(result)}"
        
        console.info(
            f"[{current:3d}/{total:3d}] {bar} {percentage:5.1f}% | "
            f"{use_case_id:20s} {status_color}{status}{Style.RESET_ALL}{detail}"
        )
    
    @staticmethod
    def _get_failure_reason(result: 'UseCaseResult') -> str:
        """Extract concise failure reason."""
        if result.errors:
            # Show first error
            return result.errors[0][:60]
        
        # Show which validation failed
        failed_validations = [
            k for k, v in result.validation_results.items() if not v
        ]
        
        if failed_validations:
            # Show the most critical failure with actual vs expected
            first_fail = failed_validations[0]
            expected = result.use_case.expected_outcomes.get(first_fail)
            actual = result.actual_outcomes.get(first_fail)
            
            # Format based on type
            if isinstance(expected, tuple):  # Range
                return f"{first_fail}: {actual} not in [{expected[0]}-{expected[1]}]"
            else:
                return f"{first_fail}: expected {expected}, got {actual}"
        
        return "Unknown failure"
    
    @staticmethod
    def print_final_summary(results: List['UseCaseResult']):
        """Enhanced final summary with failure analysis."""
        console.info("")
        console.info(Fore.CYAN + "=" * 80)
        console.info(Fore.CYAN + "EXECUTION SUMMARY")
        console.info(Fore.CYAN + "=" * 80)
        console.info("")
        
        # Overall stats
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        console.info(f"Total Use Cases:  {total}")
        console.info(
            f"Passed:           {Fore.GREEN}{passed} ({pass_rate:.1f}%){Style.RESET_ALL}"
        )
        if failed > 0:
            console.info(
                f"Failed:           {Fore.RED}{failed} ({100-pass_rate:.1f}%){Style.RESET_ALL}"
            )
        console.info("")
        
        # Category breakdown
        ConsoleSummary._print_category_breakdown(results)
        
        # Detailed failure analysis
        if failed > 0:
            ConsoleSummary._print_detailed_failures(results)
        
        # Performance summary for passed tests
        passed_results = [r for r in results if r.passed]
        if passed_results:
            ConsoleSummary._print_performance_summary(passed_results)
        
        # Bottom line
        console.info("")
        console.info(Fore.CYAN + "=" * 80)
        console.info(f"Detailed logs: {Fore.YELLOW}use_case_detailed.log")
        console.info(Fore.CYAN + "=" * 80)
    
    @staticmethod
    def _print_category_breakdown(results: List['UseCaseResult']):
        """Print results by category with details."""
        console.info(Fore.YELLOW + "Results by Category:")
        console.info("-" * 80)
        
        by_category = {}
        for result in results:
            cat = result.use_case.category.value
            if cat not in by_category:
                by_category[cat] = {'passed': 0, 'failed': 0, 'results': []}
            
            if result.passed:
                by_category[cat]['passed'] += 1
            else:
                by_category[cat]['failed'] += 1
            by_category[cat]['results'].append(result)
        
        for cat, stats in sorted(by_category.items()):
            total_cat = stats['passed'] + stats['failed']
            pass_rate = (stats['passed'] / total_cat * 100) if total_cat > 0 else 0
            
            # Color code
            if pass_rate == 100:
                color = Fore.GREEN
                icon = "✅"
            elif pass_rate >= 80:
                color = Fore.YELLOW
                icon = "⚠️ "
            else:
                color = Fore.RED
                icon = "❌"
            
            console.info(
                f"{icon} {cat:35s}: {color}{stats['passed']:2d}/{total_cat:2d} "
                f"({pass_rate:5.1f}%){Style.RESET_ALL}"
            )
        
        console.info("")
    
    @staticmethod
    def _print_detailed_failures(results: List['UseCaseResult']):
        """Print detailed analysis of failures grouped by pattern."""
        failed_results = [r for r in results if not r.passed]
        
        console.info(Fore.RED + f"\n⚠️  FAILED TESTS ANALYSIS ({len(failed_results)}):")
        console.info(Fore.RED + "=" * 80 + Style.RESET_ALL)
        
        # Group failures by pattern
        failure_patterns = EnhancedConsoleSummary._group_failures_by_pattern(failed_results)
        
        for pattern, tests in failure_patterns.items():
            console.info(f"\n{Fore.YELLOW}Issue: {pattern} ({len(tests)} tests){Style.RESET_ALL}")
            console.info("-" * 80)
            
            for result in tests[:3]:  # Show first 3 examples
                console.info(f"{Fore.RED}❌ {result.use_case.id}{Style.RESET_ALL}")
                console.info(f"   {result.use_case.name}")
                
                # Show specific validation failures
                failed_vals = {
                    k: v for k, v in result.validation_results.items() if not v
                }
                
                if failed_vals:
                    console.info(f"   {Fore.YELLOW}Validation failures:")
                    for val_name, _ in list(failed_vals.items())[:3]:
                        expected = result.use_case.expected_outcomes.get(val_name)
                        actual = result.actual_outcomes.get(val_name, "N/A")
                        
                        console.info(
                            f"      • {val_name}:"
                        )
                        console.info(
                            f"        Expected: {expected}"
                        )
                        console.info(
                            f"        Actual:   {Fore.CYAN}{actual}{Style.RESET_ALL}"
                        )
                
                # Show errors if any
                if result.errors:
                    console.info(f"   {Fore.RED}Errors:")
                    for error in result.errors[:2]:
                        console.info(f"      • {error}")
            
            if len(tests) > 3:
                console.info(f"   ... and {len(tests) - 3} more with same pattern")
        
        console.info("")
    
    @staticmethod
    def _group_failures_by_pattern(failed_results: List['UseCaseResult']) -> Dict[str, List]:
        """Group failures by common patterns."""
        patterns = {}
        
        for result in failed_results:
            # Determine failure pattern
            pattern = ConsoleSummary._identify_failure_pattern(result)
            
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(result)
        
        # Sort by frequency
        return dict(sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True))
    
    @staticmethod
    def _identify_failure_pattern(result: 'UseCaseResult') -> str:
        """Identify the type of failure."""
        if result.errors:
            error_text = result.errors[0].lower()
            if "assignment" in error_text and "mismatch" in error_text:
                return "Assignment Count Mismatch"
            elif "timeout" in error_text:
                return "LLM Timeout"
            elif "memory" in error_text:
                return "Memory Issue"
            else:
                return "Execution Error"
        
        # Check validation patterns
        failed_vals = [k for k, v in result.validation_results.items() if not v]
        
        if not failed_vals:
            return "Unknown Failure"
        
        # Common patterns
        if any("categories" in v for v in failed_vals):
            actual_cats = result.actual_outcomes.get('num_categories', 0)
            expected_range = result.use_case.expected_outcomes
            
            min_cats = expected_range.get('min_categories', 0)
            max_cats = expected_range.get('max_categories', 999)
            
            if actual_cats > max_cats:
                return "Over-Fragmentation (Too Many Categories)"
            elif actual_cats < min_cats:
                return "Under-Clustering (Too Few Categories)"
            else:
                return "Category Count Out of Range"
        
        if any("confidence" in v for v in failed_vals):
            return "Low Confidence Scores"
        
        if any("tree_merge" in v for v in failed_vals):
            return "Tree Merge Configuration Issue"
        
        if any("workflow" in v for v in failed_vals):
            return "Workflow Distribution Issue"
        
        # Default to most prominent failed validation
        return f"Validation Failed: {failed_vals[0]}"
    
    @staticmethod
    def _print_performance_summary(results: List['UseCaseResult']):
        """Print performance for successful tests."""
        if not results:
            return
        
        console.info(Fore.YELLOW + "\n⏱️  Performance Summary (Passed Tests):")
        console.info("-" * 80)
        
        total_time = sum(r.execution_time for r in results)
        avg_time = total_time / len(results)
        
        fastest = min(results, key=lambda r: r.execution_time)
        slowest = max(results, key=lambda r: r.execution_time)
        
        console.info(f"  Tests analyzed:       {len(results)}")
        console.info(f"  Total time:           {total_time:.1f}s")
        console.info(f"  Average per test:     {avg_time:.2f}s")
        console.info(
            f"  Fastest: {Fore.GREEN}{fastest.use_case.id} ({fastest.execution_time:.2f}s)"
        )
        console.info(
            f"  Slowest: {Fore.YELLOW}{slowest.use_case.id} ({slowest.execution_time:.2f}s)"
        )
        console.info("")


# ============================================================================
# FAILURE DIAGNOSIS REPORT
# ============================================================================

class FailureDiagnostics:
    """Generate actionable diagnostics for failures."""
    
    @staticmethod
    def generate_recommendations(results: List['UseCaseResult']) -> str:
        """Generate recommendations based on failure patterns."""
        failed = [r for r in results if not r.passed]
        
        if not failed:
            return "✅ All tests passed! No recommendations needed."
        
        recommendations = []
        
        # Analyze patterns
        patterns = EnhancedConsoleSummary._group_failures_by_pattern(failed)
        
        for pattern, tests in patterns.items():
            if "Over-Fragmentation" in pattern:
                recommendations.append(
                    f"\n🔧 RECOMMENDATION: Over-Fragmentation ({len(tests)} tests)\n"
                    f"   Problem: System creating too many categories\n"
                    f"   Solutions:\n"
                    f"   1. Use conservative configuration: ClusterConfig.conservative()\n"
                    f"   2. Increase new_category_threshold to 0.70+\n"
                    f"   3. Enable more aggressive consolidation\n"
                    f"   4. Reduce max_passes to prevent refinement creating new categories"
                )
            
            elif "Under-Clustering" in pattern:
                recommendations.append(
                    f"\n🔧 RECOMMENDATION: Under-Clustering ({len(tests)} tests)\n"
                    f"   Problem: System merging too aggressively\n"
                    f"   Solutions:\n"
                    f"   1. Use aggressive configuration: ClusterConfig.aggressive()\n"
                    f"   2. Decrease new_category_threshold to 0.50\n"
                    f"   3. Reduce consolidation aggressiveness"
                )
            
            elif "Assignment Count Mismatch" in pattern:
                recommendations.append(
                    f"\n🔧 RECOMMENDATION: Assignment Loss ({len(tests)} tests)\n"
                    f"   Problem: CRITICAL BUG - Assignments being dropped\n"
                    f"   Solutions:\n"
                    f"   1. Apply fixes from tree_merge_processor.py\n"
                    f"   2. Check _remap_assignments() method\n"
                    f"   3. Verify consolidation doesn't drop assignments\n"
                    f"   4. Add verification at each merge stage"
                )
            
            elif "Low Confidence" in pattern:
                recommendations.append(
                    f"\n🔧 RECOMMENDATION: Low Confidence ({len(tests)} tests)\n"
                    f"   Problem: System uncertain about categorizations\n"
                    f"   Solutions:\n"
                    f"   1. Use semantic configuration for better matching\n"
                    f"   2. Adjust confidence weights\n"
                    f"   3. Increase prefilter_k for more candidates"
                )
        
        if recommendations:
            console.info(Fore.CYAN + "\n" + "=" * 80)
            console.info(Fore.CYAN + "RECOMMENDATIONS")
            console.info(Fore.CYAN + "=" * 80)
            for rec in recommendations:
                console.info(rec)
            console.info("")
        
        return "\n".join(recommendations)

# ============================================================================
# QUICK SUMMARY VARIANTS
# ============================================================================

class QuickSummary:
    """Even more compact summary format."""
    
    @staticmethod
    def print_compact(results: List['UseCaseResult']):
        """Ultra-compact one-line per category."""
        console.info("\n" + Fore.CYAN + "QUICK SUMMARY")
        console.info(Fore.CYAN + "-" * 80)
        
        by_category = {}
        for result in results:
            cat = result.use_case.category.value
            if cat not in by_category:
                by_category[cat] = {'passed': 0, 'failed': 0, 'time': 0.0}
            
            if result.passed:
                by_category[cat]['passed'] += 1
            else:
                by_category[cat]['failed'] += 1
            by_category[cat]['time'] += result.execution_time
        
        for cat, stats in sorted(by_category.items()):
            total = stats['passed'] + stats['failed']
            rate = stats['passed'] / total * 100 if total > 0 else 0
            
            # Status emoji
            if rate == 100:
                status = "✅"
                color = Fore.GREEN
            elif rate >= 80:
                status = "⚠️ "
                color = Fore.YELLOW
            else:
                status = "❌"
                color = Fore.RED
            
            console.info(
                f"{status} {cat:35s}: "
                f"{color}{stats['passed']:2d}/{total:2d}{Style.RESET_ALL} "
                f"({rate:5.1f}%) "
                f"[{stats['time']:6.1f}s]"
            )
        
        # Overall
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        total_time = sum(r.execution_time for r in results)
        overall_rate = passed / total * 100 if total > 0 else 0
        
        console.info("")
        console.info(
            f"{'OVERALL':35s}: "
            f"{Fore.CYAN}{passed:2d}/{total:2d}{Style.RESET_ALL} "
            f"({overall_rate:5.1f}%) "
            f"[{total_time:6.1f}s]"
        )

# ============================================================================
# LIVE PROGRESS BAR (Optional)
# ============================================================================

class LiveProgress:
    """Live updating progress bar."""
    
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.passed = 0
        self.failed = 0
    
    def update(self, passed: bool):
        """Update progress."""
        self.current += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        
        self._render()
    
    def _render(self):
        """Render progress bar."""
        percentage = (self.current / self.total) * 100
        bar_length = 50
        filled = int(bar_length * self.current / self.total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Color based on pass rate
        pass_rate = self.passed / self.current * 100 if self.current > 0 else 0
        if pass_rate >= 90:
            color = Fore.GREEN
        elif pass_rate >= 70:
            color = Fore.YELLOW
        else:
            color = Fore.RED
        
        # Print with carriage return for live update
        sys.stdout.write(
            f"\r{color}[{bar}] {percentage:5.1f}% | "
            f"✅ {self.passed} ❌ {self.failed} | "
            f"{self.current}/{self.total}{Style.RESET_ALL}"
        )
        sys.stdout.flush()
        
        if self.current == self.total:
            sys.stdout.write("\n")

# ============================================================================
# USE CASE CATEGORIES
# ============================================================================

class UseCaseCategory(Enum):
    """Categories of use cases."""
    BASIC_OPERATIONS = "basic_operations"
    CONFIGURATION = "configuration"
    DATASET_CHARACTERISTICS = "dataset_characteristics"
    EDGE_CASES = "edge_cases"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"
    INTEGRATION = "integration"
    REAL_WORLD = "real_world"
    TREE_MERGE = "tree_merge"
    BIG_TEXT_HANDLING = "big_text_handling"


@dataclass
class UseCase:
    """Structured use case definition."""
    id: str
    category: UseCaseCategory
    name: str
    description: str
    input_data: List[str]
    config: ClusterConfig
    expected_outcomes: Dict[str, Any]
    validation_criteria: List[str]
    notes: str = ""
    
    def __str__(self):
        return f"[{self.id}] {self.name}\n  Category: {self.category.value}\n  {self.description}"


# ============================================================================
# SECTION 1: BASIC OPERATIONS
# ============================================================================

def basic_use_cases() -> List[UseCase]:
    """Basic clustering operations - fundamental functionality."""
    
    return [
        # UC-BASIC-001: Minimal dataset
        UseCase(
            id="UC-BASIC-001",
            category=UseCaseCategory.BASIC_OPERATIONS,
            name="Minimal Dataset (2-3 texts)",
            description="Test smallest viable dataset for clustering",
            input_data=[
                "Machine learning for predictions",
                "Web development with React",
            ],
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "min_categories": 1,
                "max_categories": 2,
                "all_texts_assigned": True,
            },
            validation_criteria=[
                "Each text must be assigned",
                "Categories should be meaningful",
                "Confidence scores > 0.5",
            ],
            notes="Edge case: Minimum viable clustering scenario"
        ),
        
        # UC-BASIC-002: Small homogeneous dataset
        UseCase(
            id="UC-BASIC-002",
            category=UseCaseCategory.BASIC_OPERATIONS,
            name="Small Homogeneous Dataset",
            description="10 texts from same domain - should create 1-2 categories",
            input_data=[
                "Machine learning algorithms",
                "Deep learning models",
                "Neural network training",
                "Supervised learning techniques",
                "Unsupervised clustering",
                "Reinforcement learning agents",
                "Model optimization methods",
                "Feature engineering strategies",
                "Cross-validation techniques",
                "Hyperparameter tuning",
            ],
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "min_categories": 1,
                "max_categories": 3,
                "dominant_category_ratio": 0.7,  # 70%+ in main category
            },
            validation_criteria=[
                "Most texts in single category (AI/ML)",
                "High confidence scores (>0.8)",
                "Minimal over-fragmentation",
            ]
        ),
        
        # UC-BASIC-003: Small heterogeneous dataset
        UseCase(
            id="UC-BASIC-003",
            category=UseCaseCategory.BASIC_OPERATIONS,
            name="Small Heterogeneous Dataset",
            description="10 texts from distinct domains - should create 3-5 categories",
            input_data=[
                "Machine learning algorithms",
                "Web development with React",
                "Financial market analysis",
                "Database normalization",
                "Cloud infrastructure setup",
                "Mobile app design patterns",
                "Legal contract review",
                "Medical diagnosis systems",
                "Agricultural crop management",
                "Supply chain optimization",
            ],
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "min_categories": 5,
                "max_categories": 10,
                "no_dominant_category": True,
            },
            validation_criteria=[
                "Each domain gets separate category",
                "No category has >30% of texts",
                "Clear category distinctions",
            ]
        ),
        
        # UC-BASIC-004: Medium dataset with natural clusters
        UseCase(
            id="UC-BASIC-004",
            category=UseCaseCategory.BASIC_OPERATIONS,
            name="Medium Dataset with Natural Clusters",
            description="50 texts with 5 clear topic clusters",
            input_data=_generate_clustered_dataset(
                clusters=[
                    ("AI/ML", 10),
                    ("Web Development", 10),
                    ("Finance", 10),
                    ("Healthcare", 10),
                    ("E-commerce", 10),
                ]
            ),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "min_categories": 5,
                "max_categories": 8,
                "category_balance": 0.3,  # No category <30% or >70% expected size
            },
            validation_criteria=[
                "5 primary categories identified",
                "Each category has ~10 texts",
                "High inter-cluster separation",
            ]
        ),
        
        # UC-BASIC-005: Standard batch processing
        UseCase(
            id="UC-BASIC-005",
            category=UseCaseCategory.BASIC_OPERATIONS,
            name="Standard Batch Processing",
            description="Test default batch size behavior with 20 texts",
            input_data=_generate_mixed_dataset(20),
            config=ClusterConfig.balanced().with_overrides(batch_size=5),
            expected_outcomes={
                "batches_processed": 4,
                "consistent_results": True,
            },
            validation_criteria=[
                "All 20 texts processed",
                "Batch boundaries don't affect clustering",
                "Reproducible results",
            ]
        ),
    ]


# ============================================================================
# SECTION 2: CONFIGURATION VARIATIONS
# ============================================================================

def configuration_use_cases() -> List[UseCase]:
    """Test different configuration presets and custom configs."""
    
    test_data = _generate_mixed_dataset(30)
    
    return [
        # UC-CONFIG-001: Balanced configuration
        UseCase(
            id="UC-CONFIG-001",
            category=UseCaseCategory.CONFIGURATION,
            name="Balanced Configuration (Default)",
            description="Standard balanced approach - hybrid retrieval, 2 passes",
            input_data=test_data,
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "retrieval_mode": "hybrid",
                "passes": 2,
                "reasonable_category_count": True,
            },
            validation_criteria=[
                "Uses both TF-IDF and BERT",
                "2-pass refinement applied",
                "Moderate category count",
            ]
        ),
        
        # UC-CONFIG-002: Conservative configuration
        UseCase(
            id="UC-CONFIG-002",
            category=UseCaseCategory.CONFIGURATION,
            name="Conservative Configuration",
            description="Favors consolidation - fewer, broader categories",
            input_data=test_data,
            config=ClusterConfig.conservative(),
            expected_outcomes={
                "categories_fewer_than_balanced": True,
                "higher_threshold": 0.70,
                "more_passes": 3,
            },
            validation_criteria=[
                "Fewer categories than balanced",
                "Higher average texts per category",
                "Strong category consolidation",
            ]
        ),
        
        # UC-CONFIG-003: Aggressive configuration
        UseCase(
            id="UC-CONFIG-003",
            category=UseCaseCategory.CONFIGURATION,
            name="Aggressive Configuration",
            description="Creates new categories easily - fine-grained clustering",
            input_data=test_data,
            config=ClusterConfig.aggressive(),
            expected_outcomes={
                "categories_more_than_balanced": True,
                "lower_threshold": 0.50,
                "fine_grained": True,
            },
            validation_criteria=[
                "More categories than balanced",
                "Specific, narrow categories",
                "Nuanced distinctions captured",
            ]
        ),
        
        # UC-CONFIG-004: Semantic-focused configuration
        UseCase(
            id="UC-CONFIG-004",
            category=UseCaseCategory.CONFIGURATION,
            name="Semantic BERT Configuration",
            description="Pure semantic similarity - BERT only, no TF-IDF",
            input_data=test_data,
            config=ClusterConfig.semantic(),
            expected_outcomes={
                "retrieval_mode": "bert",
                "semantic_grouping": True,
            },
            validation_criteria=[
                "Groups by meaning, not keywords",
                "Handles synonyms well",
                "Slower but more accurate",
            ]
        ),
        
        # UC-CONFIG-005: Fast configuration
        UseCase(
            id="UC-CONFIG-005",
            category=UseCaseCategory.CONFIGURATION,
            name="Fast Configuration (Speed-Optimized)",
            description="TF-IDF only, single pass, large batches",
            input_data=test_data,
            config=ClusterConfig.fast(),
            expected_outcomes={
                "execution_time_lowest": True,
                "passes": 1,
                "tfidf_only": True,
            },
            validation_criteria=[
                "Fastest execution time",
                "Reasonable accuracy maintained",
                "Suitable for prototyping",
            ]
        ),
        
        # UC-CONFIG-006: Quality configuration
        UseCase(
            id="UC-CONFIG-006",
            category=UseCaseCategory.CONFIGURATION,
            name="Quality Configuration (Accuracy-Optimized)",
            description="3 passes, small batches, thorough analysis",
            input_data=test_data,
            config=ClusterConfig.quality(),
            expected_outcomes={
                "passes": 3,
                "highest_confidence": True,
                "best_category_quality": True,
            },
            validation_criteria=[
                "Highest average confidence",
                "Most refined categories",
                "Best accuracy metrics",
            ]
        ),
        
        # UC-CONFIG-007: Cost-optimized configuration
        UseCase(
            id="UC-CONFIG-007",
            category=UseCaseCategory.CONFIGURATION,
            name="Cost-Optimized Configuration",
            description="Minimize LLM API calls while maintaining quality",
            input_data=test_data,
            config=ClusterConfig.cost_optimized(),
            expected_outcomes={
                "llm_calls_minimized": True,
                "acceptable_quality": True,
            },
            validation_criteria=[
                "Lowest API cost",
                "Quality within 10% of balanced",
                "Single pass execution",
            ]
        ),
        
        # UC-CONFIG-008: Custom configuration
        UseCase(
            id="UC-CONFIG-008",
            category=UseCaseCategory.CONFIGURATION,
            name="Custom Configuration with Overrides",
            description="Custom tuning for specific requirements",
            input_data=test_data,
            config=ClusterConfig.balanced().with_overrides(
                batch_size=10,
                max_passes=4,
                new_category_threshold=0.65,
                prefilter_k=5,
            ),
            expected_outcomes={
                "custom_parameters_applied": True,
                "predictable_behavior": True,
            },
            validation_criteria=[
                "All overrides respected",
                "No parameter conflicts",
                "Stable execution",
            ]
        ),
        
        # UC-CONFIG-009: Auto-recommended configuration
        UseCase(
            id="UC-CONFIG-009",
            category=UseCaseCategory.CONFIGURATION,
            name="Auto-Recommended Configuration",
            description="Let system analyze dataset and recommend config",
            input_data=test_data,
            config=ClusterConfig.from_texts(test_data, priority=Priority.BALANCED),
            expected_outcomes={
                "analysis_performed": True,
                "recommendations_appropriate": True,
            },
            validation_criteria=[
                "Dataset analyzed correctly",
                "Appropriate config selected",
                "Predictions roughly accurate",
            ]
        ),
    ]


# ============================================================================
# SECTION 3: DATASET CHARACTERISTICS
# ============================================================================

def dataset_characteristic_use_cases() -> List[UseCase]:
    """Test various dataset characteristics and their handling."""
    
    return [
        # UC-DATA-001: Very short texts
        UseCase(
            id="UC-DATA-001",
            category=UseCaseCategory.DATASET_CHARACTERISTICS,
            name="Very Short Texts (5-10 words)",
            description="Handle texts with minimal content",
            input_data=[
                "ML algorithms",
                "Web React app",
                "Finance trading",
                "Database SQL",
                "Cloud AWS setup",
            ] * 4,  # 20 short texts
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "handles_short_texts": True,
                "no_errors": True,
            },
            validation_criteria=[
                "All texts processed",
                "Meaningful categories despite brevity",
                "No quality degradation",
            ]
        ),
        
        # UC-DATA-002: Very long texts (but below chunk threshold)
        UseCase(
            id="UC-DATA-002",
            category=UseCaseCategory.DATASET_CHARACTERISTICS,
            name="Long Texts (150-200 tokens)",
            description="Handle detailed texts without chunking",
            input_data=[
                "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data. " * 3,
                "Web development encompasses all the activities involved in creating websites for hosting via internet. " * 3,
            ] * 5,  # 10 long texts
            config=ClusterConfig.balanced().with_overrides(token_threshold=250),
            expected_outcomes={
                "no_chunking_needed": True,
                "quality_maintained": True,
            },
            validation_criteria=[
                "Processes without chunking",
                "Captures full context",
                "High confidence scores",
            ]
        ),
        
        # UC-DATA-003: Highly repetitive texts
        UseCase(
            id="UC-DATA-003",
            category=UseCaseCategory.DATASET_CHARACTERISTICS,
            name="Highly Repetitive Texts",
            description="Handle texts with lots of repetition",
            input_data=[
                "machine learning machine learning machine learning",
                "web development web development web development",
                "finance analysis finance analysis finance analysis",
            ] * 5,
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "handles_repetition": True,
                "distinct_categories": True,
            },
            validation_criteria=[
                "Extracts meaningful signals",
                "Ignores repetition noise",
                "Creates valid categories",
            ]
        ),
        
        # UC-DATA-004: Mixed languages (if supported)
        UseCase(
            id="UC-DATA-004",
            category=UseCaseCategory.DATASET_CHARACTERISTICS,
            name="Multi-Language Dataset",
            description="Handle texts in different languages",
            input_data=[
                "Machine learning algorithms",
                "Algoritmos de aprendizaje automático",
                "Algorithmes d'apprentissage automatique",
                "Web development frameworks",
                "Frameworks de desarrollo web",
                "Frameworks de développement web",
            ] * 3,
            config=ClusterConfig.semantic(),  # BERT handles multilingual better
            expected_outcomes={
                "groups_by_topic_not_language": True,
                "semantic_understanding": True,
            },
            validation_criteria=[
                "Same topics grouped regardless of language",
                "No language-based categories",
                "Semantic similarity recognized",
            ],
            notes="Depends on BERT model's multilingual capability"
        ),
        
        # UC-DATA-005: Texts with special characters
        UseCase(
            id="UC-DATA-005",
            category=UseCaseCategory.DATASET_CHARACTERISTICS,
            name="Special Characters and Symbols",
            description="Handle texts with code, symbols, punctuation",
            input_data=[
                "Python: def hello(): logger.info('world')",
                "JavaScript: const hello = () => console.log('world')",
                "SQL: SELECT * FROM users WHERE id = 1",
                "Cost analysis: $1,000 + €500 = ¥150,000",
                "Email: contact@example.com, Phone: +1-555-0100",
            ] * 4,
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "handles_special_chars": True,
                "no_parsing_errors": True,
            },
            validation_criteria=[
                "Special characters don't break processing",
                "Code snippets categorized correctly",
                "Symbols preserved where meaningful",
            ]
        ),
        
        # UC-DATA-006: Texts with high vocabulary diversity
        UseCase(
            id="UC-DATA-006",
            category=UseCaseCategory.DATASET_CHARACTERISTICS,
            name="High Vocabulary Diversity",
            description="Handle texts with rich, diverse vocabulary",
            input_data=_generate_high_vocabulary_texts(20),
            config=ClusterConfig.semantic(),  # Better for complex vocabulary
            expected_outcomes={
                "captures_semantic_nuances": True,
                "vocabulary_richness_handled": True,
            },
            validation_criteria=[
                "Complex vocabulary doesn't confuse clustering",
                "Semantic relationships identified",
                "Synonyms grouped appropriately",
            ]
        ),
        
        # UC-DATA-007: Texts with high keyword overlap
        UseCase(
            id="UC-DATA-007",
            category=UseCaseCategory.DATASET_CHARACTERISTICS,
            name="High Keyword Overlap Between Domains",
            description="Handle texts where different domains share vocabulary",
            input_data=[
                "Network security protocols and encryption",
                "Social network analysis and graph theory",
                "Neural network training and optimization",
                "Computer network topology design",
                "Business network development strategies",
            ] * 4,
            config=ClusterConfig.semantic(),  # Needs semantic understanding
            expected_outcomes={
                "distinguishes_despite_overlap": True,
                "context_matters": True,
            },
            validation_criteria=[
                "Different 'network' contexts separated",
                "Semantic context drives clustering",
                "Keyword overlap doesn't cause confusion",
            ]
        ),
    ]


# ============================================================================
# SECTION 4: EDGE CASES AND ERROR HANDLING
# ============================================================================

def edge_case_use_cases() -> List[UseCase]:
    """Test edge cases, boundary conditions, and error scenarios."""
    
    return [
        # UC-EDGE-001: Single text
        UseCase(
            id="UC-EDGE-001",
            category=UseCaseCategory.EDGE_CASES,
            name="Single Text Input",
            description="Absolute minimum input - 1 text",
            input_data=["Machine learning algorithms"],
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "categories": 1,
                "assignments": 1,
                "no_errors": True,
            },
            validation_criteria=[
                "Creates single category",
                "Text assigned with confidence 1.0",
                "Graceful handling",
            ]
        ),
        
        # UC-EDGE-002: Empty strings
        UseCase(
            id="UC-EDGE-002",
            category=UseCaseCategory.EDGE_CASES,
            name="Empty String Handling",
            description="Handle empty or whitespace-only texts",
            input_data=[
                "Machine learning algorithms",
                "",
                "   ",
                "Web development",
                "",
            ],
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "filters_empty_texts": True,
                "processes_valid_texts": True,
            },
            validation_criteria=[
                "Empty texts filtered or handled gracefully",
                "Valid texts processed normally",
                "No crashes or errors",
            ]
        ),
        
        # UC-EDGE-003: Duplicate texts
        UseCase(
            id="UC-EDGE-003",
            category=UseCaseCategory.EDGE_CASES,
            name="Duplicate Text Handling",
            description="Handle exact duplicate texts in dataset",
            input_data=[
                "Machine learning algorithms",
                "Machine learning algorithms",
                "Machine learning algorithms",
                "Web development",
                "Web development",
            ],
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "assigns_all_duplicates": True,
                "same_category_for_duplicates": True,
            },
            validation_criteria=[
                "All duplicates assigned",
                "Duplicates go to same category",
                "High confidence for duplicates",
            ]
        ),
        
        # UC-EDGE-004: Near-duplicate texts
        UseCase(
            id="UC-EDGE-004",
            category=UseCaseCategory.EDGE_CASES,
            name="Near-Duplicate Texts",
            description="Handle texts that are very similar but not identical",
            input_data=[
                "Machine learning algorithms for prediction",
                "Machine learning algorithms for predictions",
                "Machine learning algorithm for prediction",
                "Machine learning algorithms for predictive analytics",
            ] * 3,
            config=ClusterConfig.conservative(),  # Should consolidate
            expected_outcomes={
                "consolidates_similar": True,
                "single_category_likely": True,
            },
            validation_criteria=[
                "Near-duplicates grouped together",
                "Minimal over-fragmentation",
                "High confidence scores",
            ]
        ),
        
        # UC-EDGE-005: All texts identical
        UseCase(
            id="UC-EDGE-005",
            category=UseCaseCategory.EDGE_CASES,
            name="All Identical Texts",
            description="Extreme case - all texts are exactly the same",
            input_data=["Machine learning algorithms"] * 20,
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "single_category": True,
                "all_high_confidence": True,
            },
            validation_criteria=[
                "Creates single category",
                "All texts assigned to it",
                "Confidence ~1.0 for all",
            ]
        ),
        
        # UC-EDGE-006: Maximum batch boundary (exact multiple)
        UseCase(
            id="UC-EDGE-006",
            category=UseCaseCategory.EDGE_CASES,
            name="Exact Batch Size Multiple",
            description="Text count exactly equals batch_size * N",
            input_data=_generate_mixed_dataset(30),  # 30 = 10 * 3
            config=ClusterConfig.balanced().with_overrides(batch_size=10),
            expected_outcomes={
                "clean_batch_division": True,
                "no_partial_batches": True,
            },
            validation_criteria=[
                "Exactly 3 batches processed",
                "No remainder handling needed",
                "Clean execution",
            ]
        ),
        
        # UC-EDGE-007: One text over batch boundary
        UseCase(
            id="UC-EDGE-007",
            category=UseCaseCategory.EDGE_CASES,
            name="Batch Boundary Plus One",
            description="Text count = batch_size * N + 1",
            input_data=_generate_mixed_dataset(31),  # 30 + 1
            config=ClusterConfig.balanced().with_overrides(batch_size=10),
            expected_outcomes={
                "handles_remainder": True,
                "last_batch_size_1": True,
            },
            validation_criteria=[
                "4 batches processed (3 full, 1 single)",
                "Single-text batch handled correctly",
                "No quality loss for last text",
            ]
        ),
        
        # UC-EDGE-008: Workflow boundary exact
        UseCase(
            id="UC-EDGE-008",
            category=UseCaseCategory.EDGE_CASES,
            name="Exact Workflow Boundary",
            description="Texts exactly equal max_texts_per_run",
            input_data=_generate_mixed_dataset(100),
            config=ClusterConfig.balanced().with_overrides(
                max_texts_per_run=100,
                enable_tree_merge=True
            ),
            expected_outcomes={
                "single_workflow": True,
                "no_tree_merge": False,  # Exactly at boundary
            },
            validation_criteria=[
                "Handled as single workflow or triggers tree merge",
                "Boundary behavior documented",
                "Consistent results",
            ]
        ),
        
        # UC-EDGE-009: Unicode and emoji handling
        UseCase(
            id="UC-EDGE-009",
            category=UseCaseCategory.EDGE_CASES,
            name="Unicode and Emoji Content",
            description="Handle texts with unicode and emoji characters",
            input_data=[
                "Machine learning 🤖 and AI 🧠",
                "Web development 💻 with React ⚛️",
                "Finance 💰 and trading 📈",
                "Café ☕ and résumé 📄 handling",
                "Chinese 中文, Japanese 日本語, Arabic العربية",
            ] * 3,
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "handles_unicode": True,
                "preserves_meaning": True,
            },
            validation_criteria=[
                "Unicode processed correctly",
                "Emojis don't break clustering",
                "International characters handled",
            ]
        ),
        
        # UC-EDGE-010: Very unbalanced distribution
        UseCase(
            id="UC-EDGE-010",
            category=UseCaseCategory.EDGE_CASES,
            name="Highly Unbalanced Dataset",
            description="95% of texts from one domain, 5% from others",
            input_data=(
                ["Machine learning algorithms"] * 95 +
                ["Web development"] * 2 +
                ["Finance"] * 2 +
                ["Healthcare"] * 1
            ),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "dominant_category_exists": True,
                "minorities_not_lost": True,
            },
            validation_criteria=[
                "Dominant category identified",
                "Minority categories still created",
                "No texts lost in dominant category",
            ]
        ),
    ]


# ============================================================================
# SECTION 5: PERFORMANCE SCENARIOS
# ============================================================================

def performance_use_cases() -> List[UseCase]:
    """Test performance characteristics and scalability."""
    
    return [
        # UC-PERF-001: Small dataset performance
        UseCase(
            id="UC-PERF-001",
            category=UseCaseCategory.PERFORMANCE,
            name="Small Dataset Performance (<100 texts)",
            description="Baseline performance for small datasets",
            input_data=_generate_mixed_dataset(50),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "execution_time_seconds": (10, 60),
                "memory_mb": (0, 500),
            },
            validation_criteria=[
                "Completes in reasonable time",
                "Low memory footprint",
                "No tree merge needed",
            ]
        ),
        
        # UC-PERF-002: Medium dataset performance
        UseCase(
            id="UC-PERF-002",
            category=UseCaseCategory.PERFORMANCE,
            name="Medium Dataset Performance (100-500 texts)",
            description="Performance at medium scale",
            input_data=_generate_mixed_dataset(250),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "execution_time_seconds": (60, 300),
                "memory_mb": (500, 2000),
            },
            validation_criteria=[
                "Reasonable execution time",
                "Efficient memory usage",
                "Quality maintained",
            ]
        ),
        
        # UC-PERF-003: Large dataset performance
        UseCase(
            id="UC-PERF-003",
            category=UseCaseCategory.PERFORMANCE,
            name="Large Dataset Performance (500-2000 texts)",
            description="Performance with tree merge activated",
            input_data=_generate_mixed_dataset(1000),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "tree_merge_activated": True,
                "execution_time_minutes": (5, 30),
                "memory_mb": (1000, 4000),
            },
            validation_criteria=[
                "Tree merge handles scale",
                "Memory usage reasonable",
                "Quality preserved",
            ]
        ),
        
        # UC-PERF-004: Speed-optimized execution
        UseCase(
            id="UC-PERF-004",
            category=UseCaseCategory.PERFORMANCE,
            name="Maximum Speed Configuration",
            description="Fastest possible execution with acceptable quality",
            input_data=_generate_mixed_dataset(500),
            config=ClusterConfig.fast(),
            expected_outcomes={
                "fastest_execution": True,
                "acceptable_quality": True,
            },
            validation_criteria=[
                "Significantly faster than balanced",
                "Quality within 15% of balanced",
                "Resource-efficient",
            ]
        ),
        
        # UC-PERF-005: Quality-optimized execution
        UseCase(
            id="UC-PERF-005",
            category=UseCaseCategory.PERFORMANCE,
            name="Maximum Quality Configuration",
            description="Best possible quality regardless of time",
            input_data=_generate_mixed_dataset(200),
            config=ClusterConfig.quality(),
            expected_outcomes={
                "highest_quality": True,
                "longer_execution": True,
            },
            validation_criteria=[
                "Best category coherence",
                "Highest confidence scores",
                "Most refined results",
            ]
        ),
        
        # UC-PERF-006: Parallel execution efficiency
        UseCase(
            id="UC-PERF-006",
            category=UseCaseCategory.PERFORMANCE,
            name="Parallel Processing Efficiency",
            description="Test parallelism with multiple concurrent batches",
            input_data=_generate_mixed_dataset(500),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=50,
                max_concurrent=10
            ),
            expected_outcomes={
                "parallel_speedup": True,
                "no_race_conditions": True,
            },
            validation_criteria=[
                "Faster than sequential",
                "Results deterministic",
                "No data corruption",
            ]
        ),
        
        # UC-PERF-007: Memory-constrained execution
        UseCase(
            id="UC-PERF-007",
            category=UseCaseCategory.PERFORMANCE,
            name="Low Memory Environment",
            description="Execute efficiently with memory constraints",
            input_data=_generate_mixed_dataset(500),
            config=ClusterConfig.from_texts(
                _generate_mixed_dataset(500),
                priority=Priority.COST,
                memory_limit_mb=1024
            ),
            expected_outcomes={
                "memory_limit_respected": True,
                "adjusted_batch_sizes": True,
            },
            validation_criteria=[
                "Stays within memory limit",
                "Completes successfully",
                "Quality acceptable",
            ]
        ),
        
        # UC-PERF-008: Cost-optimized execution
        UseCase(
            id="UC-PERF-008",
            category=UseCaseCategory.PERFORMANCE,
            name="Minimum API Cost",
            description="Minimize LLM API calls while maintaining quality",
            input_data=_generate_mixed_dataset(500),
            config=ClusterConfig.cost_optimized(),
            expected_outcomes={
                "llm_calls_minimized": True,
                "cost_lowest": True,
                "quality_acceptable": True,
            },
            validation_criteria=[
                "Fewest LLM API calls",
                "Single pass execution",
                "Quality within 20% of balanced",
            ]
        ),
    ]


# ============================================================================
# SECTION 6: TREE MERGE SCENARIOS
# ============================================================================

def tree_merge_use_cases() -> List[UseCase]:
    """Test tree merge functionality and hierarchical merging."""
    
    return [
        # UC-TREE-001: Basic tree merge (2 workflows)
        UseCase(
            id="UC-TREE-001",
            category=UseCaseCategory.TREE_MERGE,
            name="Basic Tree Merge (2 Workflows)",
            description="Simplest tree merge scenario with 2 workflows",
            input_data=_generate_mixed_dataset(15),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=3,
                max_texts_per_run=8,  # Forces 2 workflows
                enable_tree_merge=True
            ),
            expected_outcomes={
                "workflows": 2,
                "merge_levels": 1,
                "all_assignments_preserved": True,
            },
            validation_criteria=[
                "Exactly 2 workflows created",
                "Single merge operation",
                "No assignment loss",
                "Categories consolidated correctly",
            ]
        ),
        
        # UC-TREE-002: Three-way merge
        UseCase(
            id="UC-TREE-002",
            category=UseCaseCategory.TREE_MERGE,
            name="Three-Way Tree Merge",
            description="Tree merge with 3 workflows requiring 2 merge levels",
            input_data=_generate_mixed_dataset(30),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=3,
                max_texts_per_run=10,  # Forces 3 workflows
                enable_tree_merge=True
            ),
            expected_outcomes={
                "workflows": 3,
                "merge_levels": 2,
                "assignments_preserved": True,
            },
            validation_criteria=[
                "3 workflows executed in parallel",
                "Binary merge tree structure",
                "All 30 assignments preserved",
                "No data loss at any merge level",
            ]
        ),
        
        # UC-TREE-003: Deep tree merge (4+ workflows)
        UseCase(
            id="UC-TREE-003",
            category=UseCaseCategory.TREE_MERGE,
            name="Deep Tree Merge (4+ Workflows)",
            description="Complex tree merge with multiple levels",
            input_data=_generate_mixed_dataset(100),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=25,  # Forces 4 workflows
                enable_tree_merge=True
            ),
            expected_outcomes={
                "workflows": 4,
                "merge_levels": 2,  # log2(4) = 2
                "hierarchical_merging": True,
            },
            validation_criteria=[
                "4 workflows processed",
                "Hierarchical binary tree",
                "All 100 assignments preserved",
                "Efficient parallel execution",
            ]
        ),
        
        # UC-TREE-004: Unbalanced tree merge
        UseCase(
            id="UC-TREE-004",
            category=UseCaseCategory.TREE_MERGE,
            name="Unbalanced Tree Merge (Odd Workflows)",
            description="Tree merge with odd number of workflows (3, 5, 7)",
            input_data=_generate_mixed_dataset(35),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=12,  # Forces 3 workflows
                enable_tree_merge=True
            ),
            expected_outcomes={
                "workflows": 3,
                "unbalanced_tree": True,
                "handles_odd_count": True,
            },
            validation_criteria=[
                "Handles odd workflow count gracefully",
                "No hanging workflows",
                "All assignments preserved",
                "Efficient merging strategy",
            ]
        ),
        
        # UC-TREE-005: Priority queue behavior
        UseCase(
            id="UC-TREE-005",
            category=UseCaseCategory.TREE_MERGE,
            name="Priority Queue Dynamics",
            description="Test priority queue execution of workflows and merges",
            input_data=_generate_mixed_dataset(50),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=15,  # ~3-4 workflows
                max_parallel_merges=2,
                enable_tree_merge=True
            ),
            expected_outcomes={
                "priority_queue_used": True,
                "workflows_first": True,
                "merges_opportunistic": True,
            },
            validation_criteria=[
                "Workflows execute first (priority 0)",
                "Merges execute when results available",
                "Parallel execution efficient",
                "No deadlocks or stalls",
            ]
        ),
        
        # UC-TREE-006: Assignment preservation verification
        UseCase(
            id="UC-TREE-006",
            category=UseCaseCategory.TREE_MERGE,
            name="Assignment Preservation Across Merges",
            description="Verify no assignments lost during multi-level merging",
            input_data=_generate_mixed_dataset(60),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=20,  # Forces 3 workflows
                enable_tree_merge=True
            ),
            expected_outcomes={
                "input_count_matches_output": True,
                "no_assignments_lost": True,
                "no_assignments_duplicated": True,
            },
            validation_criteria=[
                "Input text count = output assignment count",
                "Each text assigned exactly once",
                "No assignments dropped at any stage",
                "Verification at each merge level",
            ]
        ),
        
        # UC-TREE-007: Category consolidation during merge
        UseCase(
            id="UC-TREE-007",
            category=UseCaseCategory.TREE_MERGE,
            name="Category Consolidation in Tree Merge",
            description="Verify duplicate categories merged correctly",
            input_data=_generate_clustered_dataset([
                ("AI/ML", 15),
                ("Web Dev", 15),
                ("Finance", 15),
            ]),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=15,  # 3 workflows
                enable_tree_merge=True
            ),
            expected_outcomes={
                "categories_consolidated": True,
                "no_duplicate_categories": True,
                "semantic_merging": True,
            },
            validation_criteria=[
                "Similar categories merged across workflows",
                "Final category count reasonable",
                "No 'AI/ML' and 'Machine Learning' duplicates",
                "Semantic similarity drives merging",
            ]
        ),
        
        # UC-TREE-008: Emergency merge handling
        UseCase(
            id="UC-TREE-008",
            category=UseCaseCategory.TREE_MERGE,
            name="Emergency Merge Scenario",
            description="Handle case where merge queue doesn't fully consolidate",
            input_data=_generate_mixed_dataset(40),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=15,
                max_parallel_merges=1,  # Limit parallelism
                enable_tree_merge=True
            ),
            expected_outcomes={
                "completes_successfully": True,
                "emergency_merge_if_needed": True,
            },
            validation_criteria=[
                "Handles queue exhaustion gracefully",
                "Emergency merge preserves all data",
                "Final result valid",
            ],
            notes="Tests fallback when priority queue doesn't consolidate to single result"
        ),
        
        # UC-TREE-009: Workflow boundary edge cases
        UseCase(
            id="UC-TREE-009",
            category=UseCaseCategory.TREE_MERGE,
            name="Workflow Boundary Edge Cases",
            description="Test exact boundaries for workflow creation",
            input_data=_generate_mixed_dataset(20),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=20,  # Exactly at boundary
                enable_tree_merge=True
            ),
            expected_outcomes={
                "boundary_handled_correctly": True,
                "consistent_behavior": True,
            },
            validation_criteria=[
                "Clear boundary behavior (1 or 2 workflows)",
                "Documented and consistent",
                "No off-by-one errors",
            ]
        ),
        
        # UC-TREE-010: Large-scale tree merge
        UseCase(
            id="UC-TREE-010",
            category=UseCaseCategory.TREE_MERGE,
            name="Large-Scale Tree Merge (10+ Workflows)",
            description="Stress test with many workflows and deep tree",
            input_data=_generate_mixed_dataset(500),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=10,
                max_texts_per_run=50,  # ~10 workflows
                max_parallel_merges=4,
                enable_tree_merge=True
            ),
            expected_outcomes={
                "workflows": 10,
                "merge_levels": 4,  # log2(10) ≈ 3-4
                "scales_well": True,
            },
            validation_criteria=[
                "Handles 10 workflows efficiently",
                "Deep tree doesn't cause issues",
                "All 500 assignments preserved",
                "Performance acceptable",
            ]
        ),
    ]


# ============================================================================
# SECTION 7: BIG TEXT HANDLING
# ============================================================================

def big_text_use_cases() -> List[UseCase]:
    """Test big text chunking and consolidation."""
    
    return [
        # UC-BIG-001: Single big text
        UseCase(
            id="UC-BIG-001",
            category=UseCaseCategory.BIG_TEXT_HANDLING,
            name="Single Big Text (Above Threshold)",
            description="Handle one text that exceeds token threshold",
            input_data=[
                "Machine learning " * 100,  # ~400 tokens
                "Web development basics",
                "Finance and trading",
            ],
            config=ClusterConfig.balanced().with_overrides(token_threshold=200),
            expected_outcomes={
                "big_text_chunked": True,
                "chunks_tracked": True,
                "consolidated_correctly": True,
            },
            validation_criteria=[
                "Big text split into chunks",
                "Chunks processed separately",
                "Final assignment consolidates chunks",
                "Single assignment for original text",
            ]
        ),
        
        # UC-BIG-002: Multiple big texts
        UseCase(
            id="UC-BIG-002",
            category=UseCaseCategory.BIG_TEXT_HANDLING,
            name="Multiple Big Texts",
            description="Handle several texts exceeding threshold",
            input_data=[
                "Machine learning " * 100,
                "Web development " * 100,
                "Finance and trading " * 100,
                "Healthcare systems " * 100,
                "Regular small text",
            ],
            config=ClusterConfig.balanced().with_overrides(token_threshold=200),
            expected_outcomes={
                "multiple_big_texts": 4,
                "all_consolidated": True,
                "mixed_with_small": True,
            },
            validation_criteria=[
                "All big texts chunked",
                "Each gets final consolidated assignment",
                "Small texts processed normally",
                "No interference between big and small",
            ]
        ),
        
        # UC-BIG-003: Very large text (many chunks)
        UseCase(
            id="UC-BIG-003",
            category=UseCaseCategory.BIG_TEXT_HANDLING,
            name="Very Large Text (10+ Chunks)",
            description="Handle text requiring many chunks",
            input_data=[
                "Machine learning algorithms " * 500,  # ~2000 tokens, 10 chunks
                "Small text 1",
                "Small text 2",
            ],
            config=ClusterConfig.balanced().with_overrides(token_threshold=200),
            expected_outcomes={
                "chunks_created": 10,
                "chunk_voting_works": True,
                "correct_category": True,
            },
            validation_criteria=[
                "Creates ~10 chunks",
                "Chunk voting determines final category",
                "Confidence aggregated correctly",
                "Handles large chunk count",
            ]
        ),
        
        # UC-BIG-004: Big text across multiple workflows
        UseCase(
            id="UC-BIG-004",
            category=UseCaseCategory.BIG_TEXT_HANDLING,
            name="Big Text Chunks Span Multiple Workflows",
            description="Big text chunks distributed across workflows in tree merge",
            input_data=(
                ["Machine learning algorithms " * 200] +  # Big text, many chunks
                _generate_mixed_dataset(30)
            ),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=10,
                token_threshold=200,
                enable_tree_merge=True
            ),
            expected_outcomes={
                "chunks_span_workflows": True,
                "consolidation_works": True,
                "tree_merge_compatible": True,
            },
            validation_criteria=[
                "Chunks distributed across workflows",
                "Tracking works across workflow boundaries",
                "Final consolidation correct",
                "Single assignment for big text",
            ]
        ),
        
        # UC-BIG-005: Multi-topic big text
        UseCase(
            id="UC-BIG-005",
            category=UseCaseCategory.BIG_TEXT_HANDLING,
            name="Multi-Topic Big Text",
            description="Big text covering multiple topics - chunk voting",
            input_data=[
                (
                    "Machine learning is a field of AI. " * 50 +
                    "Web development uses frameworks. " * 50 +
                    "Finance and trading strategies. " * 50
                ),  # Multi-topic, ~600 tokens
                "Pure ML text",
                "Pure web text",
                "Pure finance text",
            ],
            config=ClusterConfig.balanced().with_overrides(token_threshold=150),
            expected_outcomes={
                "multi_topic_detected": True,
                "dominant_topic_identified": True,
                "secondary_topics_tracked": True,
            },
            validation_criteria=[
                "Primary category assigned",
                "Secondary categories noted if significant",
                "Chunk voting reflects topic distribution",
                "Metadata shows multi-topic nature",
            ]
        ),
        
        # UC-BIG-006: Edge case - text exactly at threshold
        UseCase(
            id="UC-BIG-006",
            category=UseCaseCategory.BIG_TEXT_HANDLING,
            name="Text Exactly at Token Threshold",
            description="Text with exactly threshold tokens",
            input_data=[
                "Machine learning " * 67,  # Exactly 200 tokens
                "Other text",
            ],
            config=ClusterConfig.balanced().with_overrides(token_threshold=200),
            expected_outcomes={
                "boundary_behavior_clear": True,
                "no_unnecessary_chunking": True,
            },
            validation_criteria=[
                "Boundary behavior documented (chunk or not)",
                "Consistent handling",
                "No errors at boundary",
            ]
        ),
        
        # UC-BIG-007: Big text with special structure
        UseCase(
            id="UC-BIG-007",
            category=UseCaseCategory.BIG_TEXT_HANDLING,
            name="Structured Big Text (Code, Lists, etc.)",
            description="Big text with structured content like code or lists",
            input_data=[
                (
                    "```python\n" +
                    "def example():\n    " +
                    "# comment\n    " * 100 +
                    "```"
                ),  # Code block
                "Regular text about programming",
            ],
            config=ClusterConfig.balanced().with_overrides(token_threshold=200),
            expected_outcomes={
                "structure_preserved": True,
                "sensible_chunking": True,
            },
            validation_criteria=[
                "Doesn't break code structure badly",
                "Chunks at reasonable boundaries",
                "Structure aids categorization",
            ]
        ),
        
        # UC-BIG-008: Chunk consolidation edge cases
        UseCase(
            id="UC-BIG-008",
            category=UseCaseCategory.BIG_TEXT_HANDLING,
            name="Chunk Consolidation Edge Cases",
            description="Test edge cases in chunk voting and consolidation",
            input_data=[
                "Topic A " * 100 + "Topic B " * 100,  # Equal split
                "Pure topic A",
                "Pure topic B",
            ],
            config=ClusterConfig.balanced().with_overrides(token_threshold=150),
            expected_outcomes={
                "handles_ties": True,
                "deterministic_tiebreak": True,
            },
            validation_criteria=[
                "Handles equal vote counts",
                "Tiebreaking is deterministic",
                "Reasonable category chosen",
            ]
        ),
    ]


# ============================================================================
# SECTION 8: INTEGRATION PATTERNS
# ============================================================================

def integration_use_cases() -> List[UseCase]:
    """Test integration with external systems and workflows."""
    
    return [
        # UC-INT-001: Batch processing pipeline
        UseCase(
            id="UC-INT-001",
            category=UseCaseCategory.INTEGRATION,
            name="Batch Processing Pipeline",
            description="Process data in batches from external source",
            input_data=_generate_mixed_dataset(100),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "batch_processing": True,
                "incremental_compatible": True,
            },
            validation_criteria=[
                "Can process in chunks",
                "Results can be accumulated",
                "Consistent across batches",
            ],
            notes="Simulates ETL pipeline integration"
        ),
        
        # UC-INT-002: Real-time streaming
        UseCase(
            id="UC-INT-002",
            category=UseCaseCategory.INTEGRATION,
            name="Real-Time Streaming Classification",
            description="Classify new texts against existing categories",
            input_data=_generate_mixed_dataset(50),
            config=ClusterConfig.fast(),  # Speed matters for streaming
            expected_outcomes={
                "low_latency": True,
                "suitable_for_streaming": True,
            },
            validation_criteria=[
                "Fast enough for real-time",
                "Can use existing categories",
                "Consistent classification",
            ],
            notes="Tests suitability for stream processing"
        ),
        
        # UC-INT-003: Database integration
        UseCase(
            id="UC-INT-003",
            category=UseCaseCategory.INTEGRATION,
            name="Database Storage and Retrieval",
            description="Store results in database and retrieve for analysis",
            input_data=_generate_mixed_dataset(30),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "serializable_results": True,
                "queryable_structure": True,
            },
            validation_criteria=[
                "Results can be serialized to JSON",
                "Structure supports SQL queries",
                "Can reconstruct from storage",
            ],
            notes="Tests data persistence patterns"
        ),
        
        # UC-INT-004: API endpoint integration
        UseCase(
            id="UC-INT-004",
            category=UseCaseCategory.INTEGRATION,
            name="REST API Endpoint",
            description="Expose clustering as REST API endpoint",
            input_data=_generate_mixed_dataset(20),
            config=ClusterConfig.fast(),
            expected_outcomes={
                "response_time_acceptable": True,
                "api_friendly_format": True,
            },
            validation_criteria=[
                "Response time < 30 seconds",
                "Results in JSON format",
                "Error handling appropriate",
            ],
            notes="Tests API deployment scenario"
        ),
        
        # UC-INT-005: Incremental updates
        UseCase(
            id="UC-INT-005",
            category=UseCaseCategory.INTEGRATION,
            name="Incremental Category Updates",
            description="Add new texts to existing category structure",
            input_data=_generate_mixed_dataset(50),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "supports_incremental": True,
                "existing_categories_preserved": True,
            },
            validation_criteria=[
                "Can seed with existing categories",
                "New texts classified against existing",
                "Categories evolve sensibly",
            ],
            notes="Tests incremental learning scenario"
        ),
        
        # UC-INT-006: Multi-tenant isolation
        UseCase(
            id="UC-INT-006",
            category=UseCaseCategory.INTEGRATION,
            name="Multi-Tenant Data Isolation",
            description="Handle data from multiple tenants/customers",
            input_data=_generate_mixed_dataset(40),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "isolation_maintained": True,
                "no_cross_contamination": True,
            },
            validation_criteria=[
                "Each tenant's data independent",
                "No category leakage",
                "Results properly tagged",
            ],
            notes="Tests SaaS deployment pattern"
        ),
        
        # UC-INT-007: Export to external formats
        UseCase(
            id="UC-INT-007",
            category=UseCaseCategory.INTEGRATION,
            name="Export to CSV/Excel/JSON",
            description="Export results in various formats",
            input_data=_generate_mixed_dataset(30),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "csv_export": True,
                "json_export": True,
                "excel_compatible": True,
            },
            validation_criteria=[
                "CSV format valid",
                "JSON properly structured",
                "Excel can open",
            ],
            notes="Tests data export capabilities"
        ),
        
        # UC-INT-008: Integration with existing taxonomies
        UseCase(
            id="UC-INT-008",
            category=UseCaseCategory.INTEGRATION,
            name="Map to Existing Taxonomy",
            description="Map discovered categories to predefined taxonomy",
            input_data=_generate_mixed_dataset(40),
            config=ClusterConfig.semantic(),  # Better for taxonomy matching
            expected_outcomes={
                "taxonomy_mapping": True,
                "hierarchical_compatible": True,
            },
            validation_criteria=[
                "Categories can map to taxonomy",
                "Hierarchical relationships preserved",
                "Unmapped categories identified",
            ],
            notes="Tests enterprise taxonomy integration"
        ),
    ]


# ============================================================================
# SECTION 9: REAL-WORLD APPLICATIONS
# ============================================================================

def real_world_use_cases() -> List[UseCase]:
    """Test realistic application scenarios."""
    
    return [
        # UC-REAL-001: Customer feedback categorization
        UseCase(
            id="UC-REAL-001",
            category=UseCaseCategory.REAL_WORLD,
            name="Customer Feedback Categorization",
            description="Categorize customer support tickets/feedback",
            input_data=[
                "Cannot login to my account, password reset not working",
                "Billing issue - charged twice for same subscription",
                "Feature request: would love dark mode",
                "Bug report: app crashes on iPhone 13",
                "Refund request for unused service",
                "How do I export my data?",
                "Compliment: love the new interface!",
                "Security concern: unauthorized access attempt",
                "Integration with Slack not working",
                "Mobile app loads very slowly",
            ] * 5,  # 50 tickets
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "issue_categories_identified": True,
                "actionable_groups": True,
            },
            validation_criteria=[
                "Technical issues grouped",
                "Billing/payment separate",
                "Feature requests identified",
                "Bugs vs. questions distinguished",
            ],
            notes="Common customer support use case"
        ),
        
        # UC-REAL-002: Document classification
        UseCase(
            id="UC-REAL-002",
            category=UseCaseCategory.REAL_WORLD,
            name="Legal/Business Document Classification",
            description="Categorize business documents by type",
            input_data=[
                "This employment contract outlines terms and conditions...",
                "Invoice #12345 for services rendered...",
                "Non-disclosure agreement between parties...",
                "Quarterly financial report for Q3 2024...",
                "Meeting minutes from board meeting dated...",
                "Purchase order for office supplies...",
                "Customer proposal for project XYZ...",
                "Privacy policy update notification...",
                "Performance review for employee...",
                "Vendor service level agreement...",
            ] * 4,  # 40 documents
            config=ClusterConfig.conservative(),  # Prefer broad categories
            expected_outcomes={
                "document_types_identified": True,
                "legal_vs_financial": True,
            },
            validation_criteria=[
                "Contracts grouped together",
                "Financial docs separate",
                "Internal vs. external docs",
                "Compliance docs identified",
            ],
            notes="Document management system integration"
        ),
        
        # UC-REAL-003: News article categorization
        UseCase(
            id="UC-REAL-003",
            category=UseCaseCategory.REAL_WORLD,
            name="News Article Categorization",
            description="Categorize news articles by topic",
            input_data=_generate_news_articles(60),
            config=ClusterConfig.aggressive(),  # Fine-grained topics
            expected_outcomes={
                "topic_categories": True,
                "subtopic_granularity": True,
            },
            validation_criteria=[
                "Politics separate from business",
                "Technology subcategories",
                "Sports by type",
                "Local vs. international",
            ],
            notes="News aggregation/recommendation systems"
        ),
        
        # UC-REAL-004: E-commerce product categorization
        UseCase(
            id="UC-REAL-004",
            category=UseCaseCategory.REAL_WORLD,
            name="Product Description Categorization",
            description="Categorize products from descriptions",
            input_data=[
                "Wireless Bluetooth headphones with noise cancellation, 30hr battery",
                "Men's cotton t-shirt, blue, size large, crew neck",
                "Kitchen blender 1000W, 6 speeds, glass jar",
                "Mystery thriller novel, bestseller, 400 pages",
                "Yoga mat non-slip 6mm thick, eco-friendly",
                "Smartphone case leather wallet, card slots",
                "Coffee maker programmable 12-cup, stainless steel",
                "Running shoes men's size 10, breathable mesh",
                "Science fiction book trilogy, paperback",
                "Laptop bag waterproof 15 inch, multiple compartments",
            ] * 5,  # 50 products
            config=ClusterConfig.semantic(),  # Product similarity
            expected_outcomes={
                "product_categories": True,
                "attributes_considered": True,
            },
            validation_criteria=[
                "Electronics grouped",
                "Clothing separate",
                "Books by genre",
                "Home goods identified",
            ],
            notes="E-commerce catalog management"
        ),
        
        # UC-REAL-005: Research paper classification
        UseCase(
            id="UC-REAL-005",
            category=UseCaseCategory.REAL_WORLD,
            name="Academic Paper Categorization",
            description="Categorize research papers by field/topic",
            input_data=_generate_research_abstracts(40),
            config=ClusterConfig.semantic(),  # Academic similarity
            expected_outcomes={
                "research_fields": True,
                "interdisciplinary_handled": True,
            },
            validation_criteria=[
                "CS separate from biology",
                "Subfields identified",
                "Cross-disciplinary papers handled",
                "Methodology vs. application",
            ],
            notes="Academic database/library systems"
        ),
        
        # UC-REAL-006: Social media content moderation
        UseCase(
            id="UC-REAL-006",
            category=UseCaseCategory.REAL_WORLD,
            name="Social Media Post Categorization",
            description="Categorize posts for moderation/analysis",
            input_data=_generate_social_posts(100),
            config=ClusterConfig.fast(),  # Need speed for social
            expected_outcomes={
                "content_types_identified": True,
                "tone_categories": True,
            },
            validation_criteria=[
                "Questions vs. statements",
                "Promotional content identified",
                "Discussion topics grouped",
                "Sentiment-aware grouping",
            ],
            notes="Social media management tools"
        ),
        
        # UC-REAL-007: Email inbox categorization
        UseCase(
            id="UC-REAL-007",
            category=UseCaseCategory.REAL_WORLD,
            name="Email Auto-Categorization",
            description="Automatically categorize incoming emails",
            input_data=[
                "RE: Project deadline update - need your input by Friday",
                "Your invoice #5678 is ready for payment",
                "Newsletter: Top 10 productivity tips this week",
                "Meeting invitation: Q4 Planning Session",
                "Security alert: unusual login activity detected",
                "Customer inquiry: question about pricing plans",
                "Shipping notification: your package has been delivered",
                "Job application: Senior Developer position",
                "Social: LinkedIn - you have new connections",
                "Receipt: Your Amazon.com order confirmation",
            ] * 6,  # 60 emails
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "email_categories": True,
                "priority_distinction": True,
            },
            validation_criteria=[
                "Work vs. personal",
                "Urgent vs. informational",
                "Newsletters separate",
                "Transactional vs. conversational",
            ],
            notes="Email client smart folders"
        ),
        
        # UC-REAL-008: Job posting categorization
        UseCase(
            id="UC-REAL-008",
            category=UseCaseCategory.REAL_WORLD,
            name="Job Posting Categorization",
            description="Categorize job postings by role/industry",
            input_data=_generate_job_postings(50),
            config=ClusterConfig.semantic(),
            expected_outcomes={
                "job_categories": True,
                "seniority_levels": True,
            },
            validation_criteria=[
                "Tech roles grouped",
                "Junior vs. senior",
                "Industry sectors",
                "Remote vs. onsite",
            ],
            notes="Job board/recruitment platforms"
        ),
        
        # UC-REAL-009: FAQ auto-generation
        UseCase(
            id="UC-REAL-009",
            category=UseCaseCategory.REAL_WORLD,
            name="FAQ Generation from Questions",
            description="Group similar questions for FAQ creation",
            input_data=[
                "How do I reset my password?",
                "What is your refund policy?",
                "Can I change my subscription plan?",
                "Why was I charged twice?",
                "How to delete my account?",
                "What payment methods do you accept?",
                "Is there a mobile app?",
                "How do I contact support?",
                "What are the system requirements?",
                "Can I export my data?",
            ] * 5,  # 50 questions
            config=ClusterConfig.conservative(),  # Group similar questions
            expected_outcomes={
                "question_topics": True,
                "duplicate_questions": True,
            },
            validation_criteria=[
                "Similar questions grouped",
                "Account management separate from billing",
                "Technical vs. policy questions",
                "Minimal over-splitting",
            ],
            notes="Knowledge base/FAQ automation"
        ),
        
        # UC-REAL-010: Medical record categorization
        UseCase(
            id="UC-REAL-010",
            category=UseCaseCategory.REAL_WORLD,
            name="Medical Record Categorization",
            description="Categorize medical notes/records by type",
            input_data=[
                "Patient presents with acute chest pain, elevated BP 140/90",
                "Routine annual physical examination, all vitals normal",
                "Follow-up visit for diabetes management, A1C levels improving",
                "Lab results: complete blood count within normal limits",
                "Prescription refill: metformin 500mg twice daily",
                "Radiology report: chest X-ray shows no abnormalities",
                "Surgical consultation for knee replacement procedure",
                "Emergency visit: severe allergic reaction to food",
                "Mental health assessment: anxiety disorder screening",
                "Vaccination record: flu shot administered",
            ] * 4,  # 40 records
            config=ClusterConfig.semantic(),  # Medical terminology
            expected_outcomes={
                "record_types_identified": True,
                "medical_categories": True,
            },
            validation_criteria=[
                "Emergencies vs. routine",
                "Labs vs. notes",
                "Procedures vs. consultations",
                "Diagnostic vs. treatment",
            ],
            notes="Healthcare EHR systems - HIPAA compliant deployment required"
        ),
    ]


# ============================================================================
# SECTION 10: ERROR HANDLING AND RECOVERY
# ============================================================================

def error_handling_use_cases() -> List[UseCase]:
    """Test error scenarios and recovery mechanisms."""
    
    return [
        # UC-ERR-001: Invalid input handling
        UseCase(
            id="UC-ERR-001",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Invalid Input Handling",
            description="Handle various invalid inputs gracefully",
            input_data=[
                None,  # None value
                "",    # Empty string
                "   ", # Whitespace only
                "Valid text",
            ],
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "filters_invalid": True,
                "processes_valid": True,
                "no_crash": True,
            },
            validation_criteria=[
                "Invalid inputs filtered or handled",
                "Valid inputs processed",
                "Clear error messages",
                "No exceptions raised",
            ]
        ),
        
        # UC-ERR-002: Configuration validation
        UseCase(
            id="UC-ERR-002",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Configuration Validation",
            description="Validate configuration parameters",
            input_data=_generate_mixed_dataset(20),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=-5,  # Invalid
            ),
            expected_outcomes={
                "validation_error": True,
                "clear_message": True,
            },
            validation_criteria=[
                "Invalid config rejected",
                "Error message explains issue",
                "Suggests valid values",
            ],
            notes="Should raise ValueError with clear message"
        ),
        
        # UC-ERR-003: API timeout handling
        UseCase(
            id="UC-ERR-003",
            category=UseCaseCategory.ERROR_HANDLING,
            name="LLM API Timeout Handling",
            description="Handle LLM API timeouts gracefully",
            input_data=_generate_mixed_dataset(20),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "timeout_handled": True,
                "fallback_used": True,
                "partial_results": True,
            },
            validation_criteria=[
                "Timeout doesn't crash system",
                "Fallback categorization used",
                "Progress preserved",
            ],
            notes="Simulates LLM_TIMEOUT_SECONDS exceeded"
        ),
        
        # UC-ERR-004: Memory overflow prevention
        UseCase(
            id="UC-ERR-004",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Memory Overflow Prevention",
            description="Handle near-memory-limit scenarios",
            input_data=_generate_mixed_dataset(1000),  # Large dataset
            config=ClusterConfig.balanced().with_overrides(
                batch_size=500  # Risky large batch
            ),
            expected_outcomes={
                "memory_managed": True,
                "no_oom_crash": True,
            },
            validation_criteria=[
                "Doesn't exceed available memory",
                "Adjusts batch size if needed",
                "Completes successfully",
            ],
            notes="Tests defensive memory management"
        ),
        
        # UC-ERR-005: Network interruption recovery
        UseCase(
            id="UC-ERR-005",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Network Interruption Recovery",
            description="Handle intermittent network issues",
            input_data=_generate_mixed_dataset(30),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "retries_on_failure": True,
                "eventual_success": True,
            },
            validation_criteria=[
                "Transient failures retried",
                "Permanent failures reported",
                "Progress not lost",
            ],
            notes="Tests retry logic for API calls"
        ),
        
        # UC-ERR-006: Partial workflow failure
        UseCase(
            id="UC-ERR-006",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Partial Workflow Failure in Tree Merge",
            description="Handle failure of one workflow in tree merge",
            input_data=_generate_mixed_dataset(50),
            config=ClusterConfig.balanced().with_overrides(
                batch_size=5,
                max_texts_per_run=15,
                enable_tree_merge=True
            ),
            expected_outcomes={
                "continues_with_success": True,
                "reports_failure": True,
            },
            validation_criteria=[
                "Other workflows complete",
                "Failed workflow reported",
                "Partial results available",
            ],
            notes="Tests fault tolerance in tree merge"
        ),
        
        # UC-ERR-007: Category ID collision
        UseCase(
            id="UC-ERR-007",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Category ID Collision Handling",
            description="Handle duplicate category IDs gracefully",
            input_data=_generate_mixed_dataset(20),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "detects_collision": True,
                "resolves_automatically": True,
            },
            validation_criteria=[
                "Collision detected",
                "Unique IDs generated",
                "No data loss",
            ],
            notes="Tests ID deduplication logic"
        ),
        
        # UC-ERR-008: Malformed LLM response
        UseCase(
            id="UC-ERR-008",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Malformed LLM Response Handling",
            description="Handle invalid/malformed responses from LLM",
            input_data=_generate_mixed_dataset(10),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "validates_response": True,
                "falls_back_gracefully": True,
            },
            validation_criteria=[
                "Invalid responses detected",
                "Fallback categorization used",
                "Processing continues",
            ],
            notes="Tests response validation"
        ),
        
        # UC-ERR-009: Concurrent execution conflicts
        UseCase(
            id="UC-ERR-009",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Concurrent Execution Conflict Resolution",
            description="Handle race conditions in parallel execution",
            input_data=_generate_mixed_dataset(50),
            config=ClusterConfig.balanced().with_overrides(
                max_concurrent=20  # High parallelism
            ),
            expected_outcomes={
                "no_race_conditions": True,
                "deterministic_results": True,
            },
            validation_criteria=[
                "Results are deterministic",
                "No data corruption",
                "Thread-safe operations",
            ],
            notes="Tests concurrency safety"
        ),
        
        # UC-ERR-010: Graceful degradation
        UseCase(
            id="UC-ERR-010",
            category=UseCaseCategory.ERROR_HANDLING,
            name="Graceful Degradation Under Load",
            description="Degrade gracefully when system under stress",
            input_data=_generate_mixed_dataset(500),
            config=ClusterConfig.balanced(),
            expected_outcomes={
                "completes_eventually": True,
                "quality_acceptable": True,
            },
            validation_criteria=[
                "Doesn't crash under load",
                "May be slower but completes",
                "Quality maintained",
            ],
            notes="Tests system resilience"
        ),
    ]


# ============================================================================
# HELPER FUNCTIONS FOR TEST DATA GENERATION
# ============================================================================

def _generate_mixed_dataset(size: int) -> List[str]:
    """Generate mixed-topic dataset of specified size."""
    topics = {
        "ml": "Machine learning algorithms and neural networks for predictive analytics",
        "web": "Web development frameworks and frontend technologies like React",
        "finance": "Financial markets analysis and trading strategies for investments",
        "health": "Healthcare systems and medical diagnostic technologies",
        "ecommerce": "E-commerce platforms and online retail management solutions",
        "education": "Educational technology and online learning platforms",
        "security": "Cybersecurity protocols and network protection systems",
        "cloud": "Cloud infrastructure and distributed computing architectures",
        "mobile": "Mobile application development for iOS and Android platforms",
        "data": "Data analytics and business intelligence reporting tools",
    }
    
    result = []
    for i in range(size):
        topic = list(topics.values())[i % len(topics)]
        result.append(f"{topic} - variant {i // len(topics)}")
    
    return result


def _generate_clustered_dataset(clusters: List[Tuple[str, int]]) -> List[str]:
    """
    Generate dataset with specific cluster structure.
    
    Args:
        clusters: List of (topic_name, count) tuples
    
    Returns:
        List of texts with specified cluster distribution
    """
    result = []
    
    topic_templates = {
        "AI/ML": "Machine learning {} algorithms for {} prediction tasks",
        "Web Development": "Web development using {} framework for {} applications",
        "Finance": "Financial {} analysis and {} trading strategies",
        "Healthcare": "Healthcare {} systems and {} medical diagnostics",
        "E-commerce": "E-commerce {} platforms for {} online retail",
    }
    
    modifiers = ["advanced", "modern", "efficient", "scalable", "robust", 
                 "innovative", "practical", "enterprise", "cloud-based", "real-time"]
    
    for topic, count in clusters:
        template = topic_templates.get(topic, "{} technology for {} solutions")
        for i in range(count):
            mod1 = modifiers[i % len(modifiers)]
            mod2 = modifiers[(i + 1) % len(modifiers)]
            result.append(template.format(mod1, mod2))
    
    return result


def _generate_high_vocabulary_texts(count: int) -> List[str]:
    """Generate texts with rich, diverse vocabulary."""
    sophisticated_texts = [
        "The epistemological paradigm of contemporary machine learning necessitates multifaceted algorithmic approaches",
        "Architectural frameworks for distributed web applications require meticulous consideration of scalability vectors",
        "Quantitative financial modeling leverages stochastic processes and econometric methodologies",
        "Pharmaceutical research paradigms integrate bioinformatics with clinical trial optimization strategies",
        "Comprehensive enterprise resource planning systems orchestrate multidimensional business processes",
        "Quantum computational architectures promise exponential acceleration for cryptographic applications",
        "Neurological diagnostic methodologies incorporate advanced imaging and biomarker analysis techniques",
        "Supply chain optimization employs sophisticated heuristic algorithms and predictive analytics",
        "Educational pedagogy evolves through integration of adaptive learning technologies",
        "Environmental sustainability initiatives require holistic approaches to resource management",
    ]
    
    return sophisticated_texts * (count // len(sophisticated_texts) + 1)


def _generate_news_articles(count: int) -> List[str]:
    """Generate news article excerpts."""
    articles = [
        "Breaking: Political tensions rise as diplomatic negotiations reach critical impasse",
        "Tech giant announces revolutionary AI breakthrough in natural language processing",
        "Stock markets rally on positive economic indicators and inflation data",
        "Championship finals: Underdog team secures victory in thrilling overtime match",
        "Scientific discovery: Researchers identify potential treatment for rare disease",
        "Climate summit concludes with landmark agreement on emission reductions",
        "Cultural festival celebrates diversity with international performances and cuisine",
        "Local community rallies to support small businesses after natural disaster",
        "Education reform bill passes, promising major changes to curriculum standards",
        "Transportation infrastructure project receives funding for modernization efforts",
    ]
    
    return articles * (count // len(articles) + 1)


def _generate_research_abstracts(count: int) -> List[str]:
    """Generate research paper abstract excerpts."""
    abstracts = [
        "This paper presents a novel approach to deep reinforcement learning using attention mechanisms",
        "We investigate the effects of climate change on marine biodiversity in coral reef ecosystems",
        "Our study examines the sociological impact of social media on adolescent development",
        "This research develops new mathematical frameworks for quantum error correction codes",
        "We analyze the economic implications of automation on labor market dynamics",
        "This work explores advanced materials for next-generation battery technologies",
        "Our findings demonstrate the efficacy of gene therapy for inherited disorders",
        "This paper proposes a new cryptographic protocol for secure multi-party computation",
        "We investigate neural correlates of decision-making using functional MRI imaging",
        "This study examines the historical evolution of democratic institutions",
    ]
    
    return abstracts * (count // len(abstracts) + 1)


def _generate_social_posts(count: int) -> List[str]:
    """Generate social media post examples."""
    posts = [
        "Just finished reading an amazing book! Highly recommend it 📚",
        "Anyone else having issues with the new app update? It keeps crashing",
        "Looking for recommendations for a good Italian restaurant downtown",
        "Excited to announce I'm starting a new job next month! 🎉",
        "Does anyone know how to fix this error message I'm getting?",
        "Check out my latest blog post on productivity tips",
        "This weather is absolutely perfect for a weekend hike! ⛰️",
        "Shoutout to the amazing customer service team at @company",
        "Hot take: pineapple on pizza is actually delicious",
        "Can't believe it's already Friday! Hope everyone has a great weekend",
    ]
    
    return posts * (count // len(posts) + 1)


def _generate_job_postings(count: int) -> List[str]:
    """Generate job posting descriptions."""
    postings = [
        "Senior Software Engineer - Full stack development with React and Node.js, 5+ years experience",
        "Marketing Manager - Lead digital campaigns, manage team, B2B experience required",
        "Data Scientist - ML/AI experience, Python, statistical analysis, PhD preferred",
        "Product Designer - UX/UI design for mobile apps, portfolio required, mid-level",
        "Sales Representative - SaaS sales, meet quotas, excellent communication skills",
        "DevOps Engineer - AWS, Kubernetes, CI/CD pipelines, infrastructure as code",
        "Content Writer - Technical writing, SEO knowledge, 3+ years experience",
        "Customer Success Manager - Enterprise clients, relationship building, remote ok",
        "Junior Frontend Developer - HTML/CSS/JavaScript, eager to learn, entry level",
        "Business Analyst - Requirements gathering, stakeholder management, Agile",
    ]
    
    return postings * (count // len(postings) + 1)


# ============================================================================
# USE CASE EXECUTION FRAMEWORK
# ============================================================================

@dataclass
class UseCaseResult:
    """Result from executing a use case."""
    use_case: UseCase
    passed: bool
    execution_time: float
    actual_outcomes: Dict[str, Any]
    validation_results: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    clustering_result: Dict[str, Any] = None


class UseCaseExecutor:
    """Executes use cases and validates results."""
    
    async def execute_use_case(self, use_case: UseCase) -> UseCaseResult:
        """Execute a single use case and validate results."""
        import time
        
        start_time = time.time()
        errors = []
        warnings = []
        actual_outcomes = {}
        validation_results = {}
        clustering_result = None

        logger.info(f"=" * 80)
        logger.info(f"Executing: {use_case.id} - {use_case.name}")
        logger.info(f"Category: {use_case.category.value}")
        logger.info(f"Description: {use_case.description}")
        logger.info(f"Input size: {len(use_case.input_data)} texts")
        logger.info(f"Config: {use_case.config.get_description()}")
        logger.info(f"=" * 80)
        
        try:
            # Execute clustering
            clustering_result = await clusterize_texts(
                use_case.input_data,
                config=use_case.config
            )
            
            # Extract actual outcomes
            actual_outcomes = self._extract_outcomes(clustering_result, use_case)
            
            # Validate results
            validation_results = self._validate_results(
                clustering_result,
                use_case,
                actual_outcomes
            )
            
            # Check for failures
            passed = all(validation_results.values()) and len(errors) == 0
            
        except Exception as e:
            errors.append(f"Execution failed: {str(e)}")
            passed = False
        
        execution_time = time.time() - start_time

        result = UseCaseResult(
            use_case=use_case,
            passed=passed,
            execution_time=execution_time,
            actual_outcomes=actual_outcomes,
            validation_results=validation_results,
            errors=errors,
            warnings=warnings,
            clustering_result=clustering_result
        )


        # Log detailed results to file
        logger.info(f"\nResult: {'PASSED' if result.passed else 'FAILED'}")
        logger.info(f"Execution time: {result.execution_time:.2f}s")
        logger.info(f"Actual outcomes: {result.actual_outcomes}")
        
        if result.validation_results:
            logger.info(f"Validation results: {result.validation_results}")
        
        if result.errors:
            logger.error(f"Errors: {result.errors}")
        
        if result.warnings:
            logger.warning(f"Warnings: {result.warnings}")
        
        if result.clustering_result:
            metadata = result.clustering_result.get('metadata', {})
            logger.info(f"Clustering metadata: {metadata}")
        
        logger.info("\n")
        
        return result
    
    def _extract_outcomes(
        self, 
        result: Dict[str, Any], 
        use_case: UseCase
    ) -> Dict[str, Any]:
        """Extract actual outcomes from clustering result."""
        metadata = result.get('metadata', {})
        categories = result.get('categories', [])
        assignments = result.get('assignments', [])
        
        return {
            'num_categories': len(categories),
            'num_assignments': len(assignments),
            'used_tree_merge': metadata.get('used_tree_merge', False),
            'total_workflows': metadata.get('total_workflows', 1),
            'passes': metadata.get('total_passes', 1),
            'avg_confidence': metadata.get('confidence_stats', {}).get('average', 0),
            'retrieval_mode': metadata.get('retrieval_mode', ''),
        }
    
    def _validate_results(
        self,
        result: Dict[str, Any],
        use_case: UseCase,
        actual_outcomes: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Validate results against expected outcomes."""
        validation = {}
        
        # Validate each expected outcome
        for key, expected_value in use_case.expected_outcomes.items():
            if key in actual_outcomes:
                actual_value = actual_outcomes[key]
                
                # Handle different validation types
                if isinstance(expected_value, bool):
                    validation[key] = actual_value == expected_value
                elif isinstance(expected_value, (int, float)):
                    validation[key] = actual_value == expected_value
                elif isinstance(expected_value, tuple):  # Range check
                    validation[key] = expected_value[0] <= actual_value <= expected_value[1]
                else:
                    validation[key] = True  # Can't validate, assume pass
            else:
                validation[key] = False  # Expected outcome not present
        
        return validation


class UseCaseReport:
    """Generates comprehensive reports from use case execution."""
    
    @staticmethod
    def generate_summary(results: List[UseCaseResult]) -> str:
        """Generate summary report."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        report = [
            "=" * 80,
            "USE CASE EXECUTION SUMMARY",
            "=" * 80,
            f"Total use cases: {total}",
            f"Passed: {passed} ({100 * passed / total:.1f}%)",
            f"Failed: {failed} ({100 * failed / total:.1f}%)",
            "",
        ]
        
        # Group by category
        by_category = {}
        for result in results:
            cat = result.use_case.category.value
            if cat not in by_category:
                by_category[cat] = {'passed': 0, 'failed': 0}
            
            if result.passed:
                by_category[cat]['passed'] += 1
            else:
                by_category[cat]['failed'] += 1
        
        report.append("Results by Category:")
        report.append("-" * 80)
        for cat, stats in sorted(by_category.items()):
            total_cat = stats['passed'] + stats['failed']
            pass_rate = 100 * stats['passed'] / total_cat if total_cat > 0 else 0
            report.append(
                f"  {cat:30s}: {stats['passed']:2d}/{total_cat:2d} passed ({pass_rate:.0f}%)"
            )
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    @staticmethod
    def generate_detailed_report(results: List[UseCaseResult]) -> str:
        """Generate detailed report with all use cases."""
        report = ["=" * 80, "DETAILED USE CASE REPORT", "=" * 80, ""]
        
        # Group by category
        by_category = {}
        for result in results:
            cat = result.use_case.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)
        
        # Report each category
        for category, cat_results in sorted(by_category.items()):
            report.append(f"\n{'='*80}")
            report.append(f"CATEGORY: {category.upper()}")
            report.append(f"{'='*80}\n")
            
            for result in cat_results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                report.append(f"{status} | {result.use_case.id} - {result.use_case.name}")
                report.append(f"  Time: {result.execution_time:.2f}s")
                
                if result.actual_outcomes:
                    report.append(f"  Outcomes: {result.actual_outcomes}")
                
                if result.errors:
                    report.append(f"  Errors:")
                    for error in result.errors:
                        report.append(f"    - {error}")
                
                if result.warnings:
                    report.append(f"  Warnings:")
                    for warning in result.warnings:
                        report.append(f"    - {warning}")
                
                report.append("")
        
        return "\n".join(report)

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_results_json(results: List['UseCaseResult'], filename: str = "results.json"):
    """Export results to JSON for further analysis."""
    import json
    
    export_data = {
        'summary': {
            'total': len(results),
            'passed': sum(1 for r in results if r.passed),
            'failed': sum(1 for r in results if not r.passed),
            'total_time': sum(r.execution_time for r in results),
        },
        'results': [
            {
                'id': r.use_case.id,
                'name': r.use_case.name,
                'category': r.use_case.category.value,
                'passed': r.passed,
                'execution_time': r.execution_time,
                'errors': r.errors,
                'warnings': r.warnings,
                'actual_outcomes': r.actual_outcomes,
            }
            for r in results
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    console.info(f"Results exported to: {Fore.YELLOW}{filename}")


def export_results_html(results: List['UseCaseResult'], filename: str = "results.html"):
    """Export results to HTML report."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Use Case Test Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Use Case Test Results</h1>
        <p>Total: {len(results)} | Passed: {sum(1 for r in results if r.passed)} | Failed: {sum(1 for r in results if not r.passed)}</p>
        <table>
            <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Category</th>
                <th>Status</th>
                <th>Time (s)</th>
            </tr>
    """
    
    for r in results:
        status_class = "pass" if r.passed else "fail"
        status_text = "✅ PASS" if r.passed else "❌ FAIL"
        
        html += f"""
            <tr>
                <td>{r.use_case.id}</td>
                <td>{r.use_case.name}</td>
                <td>{r.use_case.category.value}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{r.execution_time:.2f}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(filename, 'w') as f:
        f.write(html)
    
    console.info(f"HTML report exported to: {Fore.YELLOW}{filename}")

# ============================================================================
# ENHANCED MAIN EXECUTION
# ============================================================================

async def run_all_use_cases():
    """Execute all use cases with clean console output + detailed file logs."""
    
    ConsoleSummary.print_header()
    
    # Collect all use cases
    all_use_cases = []
    all_use_cases.extend(basic_use_cases())
    all_use_cases.extend(configuration_use_cases())
    all_use_cases.extend(dataset_characteristic_use_cases())
    all_use_cases.extend(edge_case_use_cases())
    all_use_cases.extend(performance_use_cases())
    all_use_cases.extend(tree_merge_use_cases())
    all_use_cases.extend(big_text_use_cases())
    all_use_cases.extend(integration_use_cases())
    all_use_cases.extend(real_world_use_cases())
    all_use_cases.extend(error_handling_use_cases())
    
    total = len(all_use_cases)
    console.info(f"Total use cases: {total}")
    console.info(f"Detailed logs: use_case_detailed.log")
    console.info("")
    
    # Execute
    executor = UseCaseExecutor()
    results = []
    
    for i, use_case in enumerate(all_use_cases, 1):
        try:
            result = await executor.execute_use_case(use_case)
            results.append(result)
            
            # Enhanced progress display
            UseCaseExecutor.print_progress(i, total, use_case.id, result)
            
        except Exception as e:
            logger.exception(f"Fatal error executing {use_case.id}")
            console.info(
                f"[{i:3d}/{total:3d}] {use_case.id:20s} "
                f"{Fore.RED}EXCEPTION: {str(e)[:50]}"
            )
    
    # Enhanced summary
    ConsoleSummary.print_final_summary(results)
    
    # Generate recommendations
    FailureDiagnostics.generate_recommendations(results)
    
    # Export options
    console.info(f"\n{Fore.CYAN}Export Results:")
    console.info(f"  JSON: python use_cases.py export")
    console.info(f"  HTML: Open use_case_report.html in browser")
    
    return results


async def run_category_use_cases(category: UseCaseCategory):
    """Run specific category with enhanced output."""
    
    category_map = {
        UseCaseCategory.BASIC_OPERATIONS: basic_use_cases,
        UseCaseCategory.CONFIGURATION: configuration_use_cases,
        UseCaseCategory.DATASET_CHARACTERISTICS: dataset_characteristic_use_cases,
        UseCaseCategory.EDGE_CASES: edge_case_use_cases,
        UseCaseCategory.PERFORMANCE: performance_use_cases,
        UseCaseCategory.TREE_MERGE: tree_merge_use_cases,
        UseCaseCategory.BIG_TEXT_HANDLING: big_text_use_cases,
        UseCaseCategory.INTEGRATION: integration_use_cases,
        UseCaseCategory.REAL_WORLD: real_world_use_cases,
        UseCaseCategory.ERROR_HANDLING: error_handling_use_cases,
    }
    
    if category not in category_map:
        console.info(f"{Fore.RED}Unknown category: {category}")
        return
    
    use_cases = category_map[category]()
    
    console.info(Fore.CYAN + "=" * 80)
    console.info(Fore.CYAN + f"Running Category: {category.value.upper()}")
    console.info(Fore.CYAN + "=" * 80)
    console.info(f"Use cases: {len(use_cases)}")
    console.info("")
    
    executor = UseCaseExecutor()
    results = []
    
    for i, use_case in enumerate(use_cases, 1):
        result = await executor.execute_use_case(use_case)
        results.append(result)
        
        status = "PASS" if result.passed else "FAIL"
        ConsoleSummary.print_progress(i, len(use_cases), use_case.id, status)
    
    ConsoleSummary.print_final_summary(results)
    
    return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            # Quick run with compact output
            results = asyncio.run(run_all_use_cases())
            QuickSummary.print_compact(results)
            
        elif command == "export":
            # Run and export results
            results = asyncio.run(run_all_use_cases())
            export_results_json(results)
            export_results_html(results)
            
        elif command in [cat.name.lower() for cat in UseCaseCategory]:
            # Run specific category
            category = UseCaseCategory[command.upper()]
            asyncio.run(run_category_use_cases(category))
            
        else:
            console.info(f"{Fore.RED}Unknown command: {command}")
            console.info("Usage:")
            console.info("  python use_cases.py              # Run all")
            console.info("  python use_cases.py quick        # Quick compact output")
            console.info("  python use_cases.py export       # Run and export")
            console.info("  python use_cases.py tree_merge   # Run specific category")
    else:
        # Default: run all with full summary
        asyncio.run(run_all_use_cases())