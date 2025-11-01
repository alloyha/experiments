"""
Enumerate all currently available query combinations in the refined ontology.

After removing fuzzy points and splitting query types, we have:
- Physical extents: point, region
- Epistemic types: known, uncertain (points); known, uncertain, fuzzy (regions)
- Query types: MEMBERSHIP, SUBSET, INTERSECTION, OVERLAP_FRACTION, PROXIMITY, DISTANCE
"""

from enum import Enum
from itertools import product


class PhysicalExtent(Enum):
    POINT = "point"
    REGION = "region"


class PointEpistemic(Enum):
    KNOWN = "known"
    UNCERTAIN = "uncertain"


class RegionEpistemic(Enum):
    KNOWN = "known"
    UNCERTAIN = "uncertain"
    FUZZY = "fuzzy"


class QueryType(Enum):
    MEMBERSHIP = "membership"
    SUBSET = "subset"
    INTERSECTION = "intersection"
    OVERLAP_FRACTION = "overlap_fraction"
    PROXIMITY = "proximity"
    DISTANCE = "distance"


def get_epistemic_types(extent):
    """Get valid epistemic types for a physical extent"""
    if extent == PhysicalExtent.POINT:
        return [e.value for e in PointEpistemic]
    else:  # REGION
        return [e.value for e in RegionEpistemic]


def is_valid_combination(s_ext, s_epi, t_ext, t_epi, q_type):
    """Check if a combination is semantically valid"""
    
    # Rule 1: SUBSET requires both regions
    if q_type == QueryType.SUBSET.value:
        return s_ext == "region" and t_ext == "region"
    
    # Rule 2: INTERSECTION requires both regions
    if q_type == QueryType.INTERSECTION.value:
        return s_ext == "region" and t_ext == "region"
    
    # Rule 3: OVERLAP_FRACTION requires both regions
    if q_type == QueryType.OVERLAP_FRACTION.value:
        return s_ext == "region" and t_ext == "region"
    
    # Rule 4: MEMBERSHIP requires at least one region
    if q_type == QueryType.MEMBERSHIP.value:
        if s_ext == "point" and t_ext == "point":
            return False  # point × point membership invalid
        return True
    
    # Rule 5: PROXIMITY and DISTANCE are valid for all extent combinations
    if q_type in [QueryType.PROXIMITY.value, QueryType.DISTANCE.value]:
        return True
    
    return False


def enumerate_all_combinations():
    """Enumerate all valid combinations"""
    
    combinations = []
    
    for s_ext in [e.value for e in PhysicalExtent]:
        s_epistemic_types = get_epistemic_types(PhysicalExtent(s_ext))
        
        for t_ext in [e.value for e in PhysicalExtent]:
            t_epistemic_types = get_epistemic_types(PhysicalExtent(t_ext))
            
            for q_type in [q.value for q in QueryType]:
                for s_epi in s_epistemic_types:
                    for t_epi in t_epistemic_types:
                        
                        if is_valid_combination(s_ext, s_epi, t_ext, t_epi, q_type):
                            combinations.append({
                                'subject_extent': s_ext,
                                'subject_epistemic': s_epi,
                                'target_extent': t_ext,
                                'target_epistemic': t_epi,
                                'query_type': q_type
                            })
    
    return combinations


def group_by_query_type(combinations):
    """Group combinations by query type"""
    groups = {}
    for combo in combinations:
        q_type = combo['query_type']
        if q_type not in groups:
            groups[q_type] = []
        groups[q_type].append(combo)
    return groups


def print_summary(combinations):
    """Print summary statistics"""
    
    print("="*70)
    print("AVAILABLE QUERY COMBINATIONS (After Refinement)")
    print("="*70)
    print()
    
    print(f"Total valid combinations: {len(combinations)}")
    print()
    
    # Group by query type
    groups = group_by_query_type(combinations)
    
    print("Breakdown by Query Type:")
    print("-" * 70)
    for q_type in QueryType:
        count = len(groups.get(q_type.value, []))
        print(f"  {q_type.value:20s}: {count:3d} combinations")
    print()
    
    # Count by extent patterns
    extent_patterns = {}
    for combo in combinations:
        pattern = f"{combo['subject_extent']} × {combo['target_extent']}"
        extent_patterns[pattern] = extent_patterns.get(pattern, 0) + 1
    
    print("Breakdown by Extent Pattern:")
    print("-" * 70)
    for pattern, count in sorted(extent_patterns.items()):
        print(f"  {pattern:20s}: {count:3d} combinations")
    print()


def print_detailed_table(combinations):
    """Print detailed table of all combinations"""
    
    groups = group_by_query_type(combinations)
    
    print("="*70)
    print("DETAILED COMBINATIONS BY QUERY TYPE")
    print("="*70)
    print()
    
    for q_type in QueryType:
        combos = groups.get(q_type.value, [])
        if not combos:
            continue
            
        print(f"\n{q_type.value.upper()} ({len(combos)} combinations)")
        print("-" * 70)
        
        for i, combo in enumerate(combos, 1):
            s = f"({combo['subject_extent']}, {combo['subject_epistemic']})"
            t = f"({combo['target_extent']}, {combo['target_epistemic']})"
            print(f"  {i:2d}. {s:20s} × {t:20s}")


def compare_with_original():
    """Compare with original 144-combination matrix"""
    
    print("\n" + "="*70)
    print("COMPARISON WITH ORIGINAL MATRIX")
    print("="*70)
    print()
    
    # Original matrix: 2 extents × 3 epistemic (with fuzzy points) × 2 extents × 3 epistemic × 4 query types
    # = 2 × 3 × 2 × 3 × 4 = 144
    
    original_total = 144
    original_questionable = 27  # Fuzzy point cases
    original_invalid = 36  # Semantic errors (point contains point, point overlap, etc.)
    original_valid = original_total - original_questionable - original_invalid  # 81
    
    # Refined matrix
    refined_total = len(enumerate_all_combinations())
    
    print(f"Original Matrix:")
    print(f"  Total combinations: {original_total}")
    print(f"  Questionable (fuzzy points): {original_questionable} ({original_questionable/original_total*100:.1f}%)")
    print(f"  Invalid (semantic errors): {original_invalid} ({original_invalid/original_total*100:.1f}%)")
    print(f"  Valid: {original_valid} ({original_valid/original_total*100:.1f}%)")
    print()
    
    print(f"Refined Matrix:")
    print(f"  Total combinations: {refined_total}")
    print(f"  Questionable: 0 (0.0%)")
    print(f"  Invalid: 0 (0.0%)")
    print(f"  Valid: {refined_total} (100.0%)")
    print()
    
    print(f"Reduction: {original_total} → {refined_total} ({refined_total/original_total*100:.1f}%)")
    print(f"  Eliminated fuzzy points: -{original_questionable}")
    print(f"  Eliminated semantic errors: -{original_invalid}")
    print(f"  Split CONTAINMENT into 4 types: +3 per region×region case")
    print()


if __name__ == "__main__":
    # Enumerate all combinations
    combinations = enumerate_all_combinations()
    
    # Print summary
    print_summary(combinations)
    
    # Print detailed table
    print_detailed_table(combinations)
    
    # Compare with original
    compare_with_original()
    
    # Print key observations
    print("="*70)
    print("KEY OBSERVATIONS")
    print("="*70)
    print()
    print("1. Fuzzy only applies to REGIONS (no fuzzy points)")
    print("   - Point epistemic: known, uncertain (2 types)")
    print("   - Region epistemic: known, uncertain, fuzzy (3 types)")
    print()
    print("2. Query types are now more specific:")
    print("   - MEMBERSHIP: point in region OR uniform sampling interpretation")
    print("   - SUBSET: R₁ ⊆ R₂ (regions only)")
    print("   - INTERSECTION: R₁ ∩ R₂ ≠ ∅ (regions only)")
    print("   - OVERLAP_FRACTION: |R₁∩R₂|/|R₁| (regions only)")
    print("   - PROXIMITY: d(subject, target) ≤ δ (all combinations)")
    print("   - DISTANCE: d(subject, target) (all combinations)")
    print()
    print("3. Every combination has clear semantics (no ambiguity)")
    print()
    print("4. Coverage can reach 100% with computational strategies:")
    print("   - Analytical: known × known cases")
    print("   - Framework: uncertain point × region cases")
    print("   - Monte Carlo: uncertain × uncertain cases")
    print("   - Hybrid: complex mixed cases")
    print()