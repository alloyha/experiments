import numpy as np
import time
from typing import Dict, List, Tuple
import sys

# Import the main functions (assuming they're in the same file or imported)
from scipy.stats import rice

# Earth radius in meters
R_EARTH = 6371000.0


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters using haversine formula."""
    phi1 = np.deg2rad(lat1)
    lam1 = np.deg2rad(lon1)
    phi2 = np.deg2rad(lat2)
    lam2 = np.deg2rad(lon2)
    
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R_EARTH * c


def check_geo_prob_both_uncertain(mu_P_latlon, Sigma_P, mu_Q_latlon, Sigma_Q, 
                                  d0_meters, prob_threshold=0.95, 
                                  n_mc=200_000, random_state=None):
    """Main function from the implementation."""
    if d0_meters <= 0:
        raise ValueError(f"d0_meters must be positive, got {d0_meters}")
    if not 0 <= prob_threshold <= 1:
        raise ValueError(f"prob_threshold must be in [0,1], got {prob_threshold}")
    
    mu_P = np.asarray(mu_P_latlon, dtype=float)
    mu_Q = np.asarray(mu_Q_latlon, dtype=float)
    
    Sigma_P_is_scalar = np.isscalar(Sigma_P)
    Sigma_Q_is_scalar = np.isscalar(Sigma_Q)
    
    if Sigma_P_is_scalar:
        sigma_P = float(Sigma_P)
        if sigma_P <= 0:
            raise ValueError(f"Sigma_P must be positive, got {sigma_P}")
        Sigma_P_mat = np.eye(2) * sigma_P**2
    else:
        Sigma_P_mat = np.asarray(Sigma_P, dtype=float)
        if Sigma_P_mat.shape != (2, 2):
            raise ValueError(f"Sigma_P must be scalar or 2x2, got shape {Sigma_P_mat.shape}")
    
    if Sigma_Q_is_scalar:
        sigma_Q = float(Sigma_Q)
        if sigma_Q <= 0:
            raise ValueError(f"Sigma_Q must be positive, got {sigma_Q}")
        Sigma_Q_mat = np.eye(2) * sigma_Q**2
    else:
        Sigma_Q_mat = np.asarray(Sigma_Q, dtype=float)
        if Sigma_Q_mat.shape != (2, 2):
            raise ValueError(f"Sigma_Q must be scalar or 2x2, got shape {Sigma_Q_mat.shape}")
    
    for name, mat in [("Sigma_P", Sigma_P_mat), ("Sigma_Q", Sigma_Q_mat)]:
        eigvals = np.linalg.eigvalsh(mat)
        if np.any(eigvals <= 0):
            raise ValueError(f"{name} must be positive definite, got eigenvalues {eigvals}")
    
    mu_X_deg = mu_P - mu_Q
    Sigma_X_m2 = Sigma_P_mat + Sigma_Q_mat
    
    Sigma_X_eigs = np.linalg.eigvalsh(Sigma_X_m2)
    is_isotropic = np.allclose(Sigma_X_eigs[0], Sigma_X_eigs[1], rtol=1e-6)
    
    if is_isotropic and Sigma_P_is_scalar and Sigma_Q_is_scalar:
        sigma_X = np.sqrt(sigma_P**2 + sigma_Q**2)
        delta = haversine_distance_m(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
        b = delta / sigma_X
        prob = float(rice.cdf(d0_meters / sigma_X, b))
        fulfilled = prob >= prob_threshold
        return fulfilled, prob
    else:
        rng = np.random.default_rng(random_state)
        delta_mean_m = haversine_distance_m(mu_P[0], mu_P[1], mu_Q[0], mu_Q[1])
        lat_rad = np.deg2rad((mu_P[0] + mu_Q[0]) / 2)
        meters_per_deg_lat = 111132.954
        meters_per_deg_lon = 111132.954 * np.cos(lat_rad)
        scale_to_deg = np.diag([1.0 / meters_per_deg_lat, 1.0 / meters_per_deg_lon])
        Sigma_X_deg2 = scale_to_deg @ Sigma_X_m2 @ scale_to_deg.T
        samples_X_deg = rng.multivariate_normal(mu_X_deg, Sigma_X_deg2, size=n_mc)
        samples_P = mu_Q + samples_X_deg
        dists = haversine_distance_m(samples_P[:, 0], samples_P[:, 1], mu_Q[0], mu_Q[1])
        prob = float(np.mean(dists <= d0_meters))
        fulfilled = prob >= prob_threshold
        return fulfilled, prob


class PerformanceProfiler:
    """Profile geospatial probability checker performance and accuracy."""
    
    def __init__(self):
        self.results = []
    
    def time_function(self, func, *args, **kwargs) -> Tuple[float, any]:
        """Time a function call and return (elapsed_time, result)."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return elapsed, result
    
    def profile_isotropic_case(self, n_trials: int = 100) -> Dict:
        """Profile the isotropic (Rice distribution) case."""
        print("\n" + "="*70)
        print("PROFILING: Isotropic Case (Rice Distribution - Analytical)")
        print("="*70)
        
        mu_P = (37.7749, -122.4194)
        mu_Q = (37.77495, -122.41945)
        sigma_P = 5.0
        sigma_Q = 3.0
        d0 = 20.0
        
        times = []
        for i in range(n_trials):
            elapsed, (ok, prob) = self.time_function(
                check_geo_prob_both_uncertain,
                mu_P, sigma_P, mu_Q, sigma_Q, d0
            )
            times.append(elapsed)
        
        times = np.array(times)
        result = {
            'method': 'Isotropic (Rice)',
            'n_trials': n_trials,
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'median_time_ms': np.median(times) * 1000,
            'probability': prob,
            'fulfilled': ok
        }
        
        self._print_timing_results(result)
        self.results.append(result)
        return result
    
    def profile_anisotropic_case(self, n_mc_values: List[int] = [1000, 10000, 100000, 500000]) -> Dict:
        """Profile the anisotropic (Monte Carlo) case with different sample sizes."""
        print("\n" + "="*70)
        print("PROFILING: Anisotropic Case (Monte Carlo)")
        print("="*70)
        
        mu_P = (37.7749, -122.4194)
        mu_Q = (37.77495, -122.41945)
        Sigma_P = np.array([[25.0, 5.0], [5.0, 16.0]])
        Sigma_Q = np.array([[9.0, 0.0], [0.0, 9.0]])
        d0 = 20.0
        
        results_mc = []
        
        for n_mc in n_mc_values:
            print(f"\nTesting with N={n_mc:,} Monte Carlo samples:")
            
            # Run multiple trials to get timing statistics
            n_timing_trials = 10 if n_mc <= 100000 else 5
            times = []
            probs = []
            
            for trial in range(n_timing_trials):
                elapsed, (ok, prob) = self.time_function(
                    check_geo_prob_both_uncertain,
                    mu_P, Sigma_P, mu_Q, Sigma_Q, d0,
                    n_mc=n_mc, random_state=42 + trial
                )
                times.append(elapsed)
                probs.append(prob)
            
            times = np.array(times)
            probs = np.array(probs)
            
            result = {
                'method': f'Anisotropic MC (N={n_mc:,})',
                'n_mc': n_mc,
                'n_timing_trials': n_timing_trials,
                'mean_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000,
                'min_time_ms': np.min(times) * 1000,
                'max_time_ms': np.max(times) * 1000,
                'prob_mean': np.mean(probs),
                'prob_std': np.std(probs),
                'prob_ci_95': 1.96 * np.std(probs),  # 95% confidence interval half-width
            }
            
            self._print_mc_results(result)
            results_mc.append(result)
        
        self.results.extend(results_mc)
        return results_mc
    
    def profile_accuracy_convergence(self, max_n_mc: int = 1000000) -> Dict:
        """Test Monte Carlo convergence by comparing to analytical solution."""
        print("\n" + "="*70)
        print("ACCURACY TEST: Monte Carlo Convergence vs Analytical")
        print("="*70)
        
        # Use isotropic case where we have analytical solution
        mu_P = (37.7749, -122.4194)
        mu_Q = (37.77495, -122.41945)
        sigma_P = 5.0
        sigma_Q = 3.0
        d0 = 20.0
        
        # Get analytical solution
        print("\nComputing analytical solution (Rice distribution)...")
        _, (_, prob_analytical) = self.time_function(
            check_geo_prob_both_uncertain,
            mu_P, sigma_P, mu_Q, sigma_Q, d0
        )
        print(f"  Analytical probability: {prob_analytical:.6f}")
        
        # Test MC convergence
        n_mc_values = [1000, 5000, 10000, 50000, 100000, 500000]
        if max_n_mc >= 1000000:
            n_mc_values.append(1000000)
        
        print("\nTesting Monte Carlo convergence:")
        convergence_results = []
        
        for n_mc in n_mc_values:
            # Force anisotropic path by using matrix (even though it's isotropic)
            Sigma_P_mat = np.eye(2) * sigma_P**2
            Sigma_Q_mat = np.eye(2) * sigma_Q**2
            
            elapsed, (ok, prob_mc) = self.time_function(
                check_geo_prob_both_uncertain,
                mu_P, Sigma_P_mat, mu_Q, Sigma_Q_mat, d0,
                n_mc=n_mc, random_state=42
            )
            
            error = abs(prob_mc - prob_analytical)
            rel_error = error / prob_analytical * 100
            
            result = {
                'n_mc': n_mc,
                'prob_mc': prob_mc,
                'prob_analytical': prob_analytical,
                'absolute_error': error,
                'relative_error_pct': rel_error,
                'time_ms': elapsed * 1000
            }
            
            print(f"  N={n_mc:>7,}: P={prob_mc:.6f}, Error={error:.6f} ({rel_error:.3f}%), Time={elapsed*1000:.1f}ms")
            convergence_results.append(result)
        
        return convergence_results
    
    def profile_edge_cases(self) -> Dict:
        """Profile edge cases and boundary conditions."""
        print("\n" + "="*70)
        print("EDGE CASES: Testing Boundary Conditions")
        print("="*70)
        
        edge_cases = []
        
        # Case 1: Zero separation (coincident points)
        print("\n1. Coincident points (δ = 0):")
        mu_P = (37.7749, -122.4194)
        mu_Q = (37.7749, -122.4194)  # Same location
        elapsed, (ok, prob) = self.time_function(
            check_geo_prob_both_uncertain,
            mu_P, 5.0, mu_Q, 3.0, d0_meters=10.0
        )
        print(f"   Probability: {prob:.6f}, Time: {elapsed*1000:.2f}ms")
        print(f"   Expected: High probability (δ=0, d₀=2σ_X)")
        edge_cases.append({'case': 'coincident', 'prob': prob, 'time_ms': elapsed*1000})
        
        # Case 2: Very large separation
        print("\n2. Large separation (δ >> σ):")
        mu_Q_far = (38.0, -122.0)
        elapsed, (ok, prob) = self.time_function(
            check_geo_prob_both_uncertain,
            mu_P, 5.0, mu_Q_far, 3.0, d0_meters=100.0
        )
        delta = haversine_distance_m(mu_P[0], mu_P[1], mu_Q_far[0], mu_Q_far[1])
        print(f"   Separation: {delta:.0f}m, Probability: {prob:.8f}, Time: {elapsed*1000:.2f}ms")
        print(f"   Expected: Very low probability")
        edge_cases.append({'case': 'large_separation', 'prob': prob, 'time_ms': elapsed*1000})
        
        # Case 3: Very small uncertainty
        print("\n3. Very small uncertainty (σ → 0):")
        mu_Q_close = (37.77495, -122.41945)
        elapsed, (ok, prob) = self.time_function(
            check_geo_prob_both_uncertain,
            mu_P, 0.5, mu_Q_close, 0.3, d0_meters=20.0
        )
        delta_close = haversine_distance_m(mu_P[0], mu_P[1], mu_Q_close[0], mu_Q_close[1])
        print(f"   Separation: {delta_close:.2f}m, σ_X: {np.sqrt(0.5**2 + 0.3**2):.2f}m")
        print(f"   Probability: {prob:.6f}, Time: {elapsed*1000:.2f}ms")
        edge_cases.append({'case': 'small_uncertainty', 'prob': prob, 'time_ms': elapsed*1000})
        
        # Case 4: Very large uncertainty
        print("\n4. Very large uncertainty (σ >> δ):")
        elapsed, (ok, prob) = self.time_function(
            check_geo_prob_both_uncertain,
            mu_P, 50.0, mu_Q_close, 30.0, d0_meters=20.0
        )
        sigma_X_large = np.sqrt(50**2 + 30**2)
        print(f"   σ_X: {sigma_X_large:.1f}m, δ: {delta_close:.2f}m")
        print(f"   Probability: {prob:.6f}, Time: {elapsed*1000:.2f}ms")
        print(f"   Expected: Moderate probability (d₀ < σ_X)")
        edge_cases.append({'case': 'large_uncertainty', 'prob': prob, 'time_ms': elapsed*1000})
        
        return edge_cases
    
    def _print_timing_results(self, result: Dict):
        """Print formatted timing results."""
        print(f"\nResults ({result['n_trials']} trials):")
        print(f"  Mean time:   {result['mean_time_ms']:.3f} ms")
        print(f"  Std dev:     {result['std_time_ms']:.3f} ms")
        print(f"  Min time:    {result['min_time_ms']:.3f} ms")
        print(f"  Max time:    {result['max_time_ms']:.3f} ms")
        print(f"  Median time: {result['median_time_ms']:.3f} ms")
        print(f"  Probability: {result['probability']:.6f}")
        print(f"  Fulfilled:   {result['fulfilled']}")
    
    def _print_mc_results(self, result: Dict):
        """Print formatted Monte Carlo results."""
        print(f"  Mean time:    {result['mean_time_ms']:.2f} ms ± {result['std_time_ms']:.2f} ms")
        print(f"  Probability:  {result['prob_mean']:.6f} ± {result['prob_ci_95']:.6f} (95% CI)")
        print(f"  MC std dev:   {result['prob_std']:.6f}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY REPORT")
        print("="*70)
        
        # Find isotropic result
        isotropic = [r for r in self.results if 'Rice' in r.get('method', '')]
        if isotropic:
            iso = isotropic[0]
            print(f"\nIsotropic (Analytical) Performance:")
            print(f"  ✓ Mean time: {iso['mean_time_ms']:.3f} ms")
            print(f"  ✓ Deterministic result (no sampling variance)")
            print(f"  ✓ Exact mathematical solution")
        
        # Find MC results
        mc_results = [r for r in self.results if 'MC' in r.get('method', '')]
        if mc_results:
            print(f"\nAnisotropic (Monte Carlo) Performance:")
            for mc in mc_results:
                speedup = iso['mean_time_ms'] / mc['mean_time_ms'] if isotropic else 1.0
                print(f"  • N={mc['n_mc']:>7,}: {mc['mean_time_ms']:>7.1f} ms "
                      f"(±{mc['prob_ci_95']:.4f} accuracy)")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS:")
        print("="*70)
        print("✓ Use isotropic (Rice) method when possible - ~100-1000× faster")
        print("✓ For anisotropic: N=100k gives good balance of speed/accuracy")
        print("✓ For high precision: N=500k or implement Imhof's method")
        print("✓ Monte Carlo error scales as O(1/√N)")
        print("="*70)


def run_full_profile():
    """Run complete profiling suite."""
    print("="*70)
    print("GEOSPATIAL PROBABILITY CHECKER - PERFORMANCE PROFILING")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    profiler = PerformanceProfiler()
    
    # Profile isotropic case
    profiler.profile_isotropic_case(n_trials=100)
    
    # Profile anisotropic case with different sample sizes
    profiler.profile_anisotropic_case(n_mc_values=[1000, 10000, 100000, 500000])
    
    # Test accuracy convergence
    profiler.profile_accuracy_convergence(max_n_mc=500000)
    
    # Test edge cases
    profiler.profile_edge_cases()
    
    # Generate summary
    profiler.generate_summary_report()
    
    return profiler


if __name__ == "__main__":
    profiler = run_full_profile()