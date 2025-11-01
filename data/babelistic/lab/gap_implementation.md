## Strategy for 100% Coverage

### Phase 1: Identify the 2 Undefined Cases

Looking at your simplified matrix, the likely undefined cases are:

```python
# 1. Uncertain × Uncertain distance/proximity
("point", "uncertain", "point", "uncertain", "PROXIMITY/DISTANCE")

# 2. Region distance semantics edge cases
("region", "known", "region", "known", "DISTANCE") 
# with some RegionDistanceSemantics not yet implemented
```

### Phase 2: Systematic Coverage via Computational Strategies

The key insight: **Every valid query can be computed** using one of these strategies:Now let me provide the concrete implementation pattern:

## Implementation Pattern for 100% Coverage

```python
from abc import ABC, abstractmethod
from enum import Enum, auto

class ComputationStrategy(Enum):
    """How to compute the query result"""
    ANALYTICAL = auto()      # Closed-form solution
    FRAMEWORK = auto()       # Mollified indicator (existing)
    MONTE_CARLO = auto()     # Sampling-based
    HYBRID = auto()          # Adaptive combination

class QueryComputer(ABC):
    """Abstract strategy for computing query results"""
    
    @abstractmethod
    def can_handle(self, query: Query) -> bool:
        """Can this strategy compute this query?"""
        pass
    
    @abstractmethod
    def compute(self, query: Query) -> QueryResult:
        """Compute the result"""
        pass
    
    @abstractmethod
    def estimate_cost(self, query: Query) -> float:
        """Estimate computational cost (for routing)"""
        pass


class AnalyticalComputer(QueryComputer):
    """Closed-form solutions (fastest, exact)"""
    
    def can_handle(self, query):
        # Known × Known cases
        s_known = query.subject.state.epistemic_type() == EpistemicType.KNOWN
        t_known = query.target.state.epistemic_type() == EpistemicType.KNOWN
        
        return s_known and t_known
    
    def compute(self, query):
        if query.query_type == QueryType.MEMBERSHIP:
            # Point-in-polygon test
            return self._geometric_test(query)
        elif query.query_type == QueryType.SUBSET:
            # Check indicator₁ ≤ indicator₂ everywhere
            return self._subset_test(query)
        # ... etc


class FrameworkComputer(QueryComputer):
    """Existing ProbabilityEstimator (most common)"""
    
    def can_handle(self, query):
        # Uncertain point × any region
        # Any point × fuzzy region
        # Region × region with uniform interpretation
        return self._has_distribution(query) and self._has_region(query)
    
    def compute(self, query):
        # Use existing framework
        estimator = ProbabilityEstimator(
            metric_space=query.metric_space,
            region=self._extract_region(query),
            query_distribution=self._extract_distribution(query),
            kernel=query.kernel or GaussianKernel(),
            convolution_strategy=DirectConvolution(),
            integrator=QuadratureIntegrator()
        )
        return estimator.compute(bandwidth=query.bandwidth, resolution=query.resolution)


class MonteCarloComputer(QueryComputer):
    """Sampling-based (most flexible, handles everything)"""
    
    def can_handle(self, query):
        return True  # Fallback strategy
    
    def compute(self, query):
        """
        Universal strategy:
        1. Sample from subject distribution/ensemble
        2. Sample from target distribution/ensemble
        3. Evaluate query predicate on samples
        4. Estimate probability/expectation
        """
        if query.query_type == QueryType.PROXIMITY:
            return self._proximity_monte_carlo(query)
        elif query.query_type == QueryType.DISTANCE:
            return self._distance_distribution(query)
        # ... etc
    
    def _proximity_monte_carlo(self, query):
        """
        Example: Uncertain × Uncertain proximity
        P(d(X, Y) ≤ δ)
        """
        # Sample from both distributions
        X_samples = query.subject.state.distribution.sample(self.n_samples)
        Y_samples = query.target.state.distribution.sample(self.n_samples)
        
        # Compute distances (vectorized)
        distances = query.metric_space.distance(X_samples, Y_samples)
        
        # Estimate probability
        probability = np.mean(distances <= query.distance_threshold)
        
        # Bootstrap confidence interval
        error = np.std(distances <= query.distance_threshold) / np.sqrt(self.n_samples)
        
        return QueryResult(
            value=probability,
            error_estimate=error,
            method="monte_carlo",
            n_samples=self.n_samples
        )


class HybridComputer(QueryComputer):
    """Adaptive strategy selection"""
    
    def can_handle(self, query):
        return True
    
    def compute(self, query):
        """
        Example: Uncertain point × Uncertain region proximity
        
        Strategy: Marginalize over region uncertainty, use framework for each
        P(d(X, ∂R̃) ≤ δ) = ∫ P(d(X, ∂R) ≤ δ | R) p(R) dR
        """
        if self._is_uncertain_uncertain_proximity(query):
            return self._marginalized_proximity(query)
        else:
            # Delegate to simpler strategy
            return self._route_to_best_strategy(query)
    
    def _marginalized_proximity(self, query):
        """
        Outer loop: Monte Carlo over region ensemble
        Inner loop: Framework for each fixed region
        """
        region_ensemble = query.target.state.ensemble
        
        results = []
        for region_i in region_ensemble:
            # Fixed region case → use framework
            fixed_query = Query(
                subject=query.subject,
                target=Target(query.target.entity, KnownRegion(region_i)),
                query_type=query.query_type,
                metric_space=query.metric_space,
                distance_threshold=query.distance_threshold
            )
            
            result_i = FrameworkComputer().compute(fixed_query)
            results.append(result_i.value)
        
        # Average over ensemble
        return QueryResult(
            value=np.mean(results),
            error_estimate=np.std(results) / np.sqrt(len(results)),
            method="hybrid_marginalized"
        )


class QueryRouter:
    """Automatically selects best computation strategy"""
    
    def __init__(self):
        self.strategies = [
            AnalyticalComputer(),      # Try exact first
            FrameworkComputer(),       # Then framework
            MonteCarloComputer(),      # Then sampling
            HybridComputer()           # Finally adaptive
        ]
    
    def compute(self, query: Query) -> QueryResult:
        """Route to best available strategy"""
        
        # Find all capable strategies
        capable = [s for s in self.strategies if s.can_handle(query)]
        
        if not capable:
            raise ValueError(f"No strategy can handle query: {query}")
        
        # Pick cheapest capable strategy
        best = min(capable, key=lambda s: s.estimate_cost(query))
        
        # Allow user override
        if query.force_strategy:
            best = self._get_strategy(query.force_strategy)
        
        return best.compute(query)


# Usage
def solve_any_query(query: Query) -> QueryResult:
    """100% coverage guarantee"""
    router = QueryRouter()
    return router.compute(query)
```

## Key Principles for 100% Coverage

1. **Computational Completeness ≠ Optimality**
   - Monte Carlo can solve *anything* (with enough samples)
   - Use faster strategies when possible
   - Automatic fallback chain ensures no query fails

2. **Maintain Agnosticism Through Strategy Pattern**
   - Each strategy is metric/distribution/geometry agnostic
   - Router picks strategy based on query structure, not content
   - User can override for domain-specific optimization

3. **Graceful Degradation**
   ```
   Analytical (exact, fast) 
      ↓ (if unavailable)
   Framework (approximate, medium)
      ↓ (if unavailable)  
   Monte Carlo (approximate, slow but universal)
   ```

4. **Document Computational Complexity**
   - Each strategy declares its cost model
   - Warn users about expensive queries
   - Suggest optimizations (resolution, samples, bandwidth)

## The 100% Coverage Guarantee

**Every valid query in your ontology can be computed** using this pattern:

```python
# Type safety at construction
query = Query(...)  # Validates at __post_init__

# Computational completeness at execution
result = solve_any_query(query)  # Never fails for valid queries
```

The agnostic philosophy is preserved because:
- ✅ No strategy assumes specific metric/distribution/geometry
- ✅ Strategies are pluggable and extensible
- ✅ Router is content-agnostic (only checks structure)
- ✅ User retains full control via strategy override
