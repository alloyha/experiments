"""Tests for app/core/llm_router.py - LLM routing logic."""

import pytest
from app.core.llm_router import LLMRouter, ModelProfile, RoutingDecision


class TestLLMRouter:
    """Test suite for LLM router."""

    def test_router_initialization(self):
        """Test router initializes with default config."""
        router = LLMRouter()
        assert router.config == {}
        assert router.model_configs is not None
        assert router.circuit_breakers == {}

    def test_router_initialization_with_config(self):
        """Test router initializes with custom config."""
        config = {"custom": "value"}
        router = LLMRouter(config=config)
        assert router.config == config

    def test_model_configs_loaded(self):
        """Test model configurations are loaded correctly."""
        router = LLMRouter()
        configs = router.model_configs
        
        # Check all profiles are present
        assert ModelProfile.FAST in configs
        assert ModelProfile.BALANCED in configs
        assert ModelProfile.POWERFUL in configs
        
        # Check structure for FAST profile
        fast_config = configs[ModelProfile.FAST]
        assert "primary" in fast_config
        assert "fallback" in fast_config
        assert fast_config["primary"]["provider"] == "openai"
        assert fast_config["primary"]["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_route_fast_profile(self):
        """Test routing with FAST profile."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Hello"}]
        
        decision = await router.route(messages, ModelProfile.FAST)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.provider == "openai"
        assert decision.model == "gpt-3.5-turbo"
        assert decision.profile == ModelProfile.FAST
        assert decision.estimated_cost > 0
        assert decision.estimated_latency > 0

    @pytest.mark.asyncio
    async def test_route_balanced_profile(self):
        """Test routing with BALANCED profile."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Hello"}]
        
        decision = await router.route(messages, ModelProfile.BALANCED)
        
        assert decision.provider == "openai"
        assert decision.model == "gpt-4-turbo"
        assert decision.profile == ModelProfile.BALANCED

    @pytest.mark.asyncio
    async def test_route_powerful_profile(self):
        """Test routing with POWERFUL profile."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Hello"}]
        
        decision = await router.route(messages, ModelProfile.POWERFUL)
        
        assert decision.provider == "openai"
        assert decision.model == "gpt-4"
        assert decision.profile == ModelProfile.POWERFUL

    @pytest.mark.asyncio
    async def test_route_with_circuit_breaker_open(self):
        """Test routing falls back when circuit breaker is open."""
        router = LLMRouter()
        # Open circuit breaker for primary model
        router.circuit_breakers["openai:gpt-3.5-turbo"] = True
        
        messages = [{"role": "user", "content": "Hello"}]
        decision = await router.route(messages, ModelProfile.FAST)
        
        # Should use fallback
        assert decision.provider == "anthropic"
        assert decision.model == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_route_default_profile(self):
        """Test routing with default profile."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Hello"}]
        
        # Default should be FAST
        decision = await router.route(messages)
        
        assert decision.profile == ModelProfile.FAST

    def test_estimate_tokens_simple(self):
        """Test token estimation with simple message."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Hello"}]
        
        tokens = router._estimate_tokens(messages)
        
        # "Hello" is 5 chars, should be ~2 tokens (5//4 = 1, but min is 10)
        assert tokens == 10  # Minimum

    def test_estimate_tokens_longer(self):
        """Test token estimation with longer message."""
        router = LLMRouter()
        # 100 characters
        messages = [{"role": "user", "content": "x" * 100}]
        
        tokens = router._estimate_tokens(messages)
        
        # 100 chars / 4 = 25 tokens
        assert tokens == 25

    def test_estimate_tokens_multiple_messages(self):
        """Test token estimation with multiple messages."""
        router = LLMRouter()
        messages = [
            {"role": "user", "content": "x" * 40},
            {"role": "assistant", "content": "x" * 40},
            {"role": "user", "content": "x" * 40},
        ]
        
        tokens = router._estimate_tokens(messages)
        
        # 120 chars / 4 = 30 tokens
        assert tokens == 30

    def test_estimate_latency_fast(self):
        """Test latency estimation for FAST profile."""
        router = LLMRouter()
        
        latency = router._estimate_latency(ModelProfile.FAST, 100)
        
        # Base 800ms + 100 tokens * 2ms = 1000ms
        assert latency == 1000

    def test_estimate_latency_balanced(self):
        """Test latency estimation for BALANCED profile."""
        router = LLMRouter()
        
        latency = router._estimate_latency(ModelProfile.BALANCED, 100)
        
        # Base 2000ms + 100 tokens * 2ms = 2200ms
        assert latency == 2200

    def test_estimate_latency_powerful(self):
        """Test latency estimation for POWERFUL profile."""
        router = LLMRouter()
        
        latency = router._estimate_latency(ModelProfile.POWERFUL, 100)
        
        # Base 5000ms + 100 tokens * 2ms = 5200ms
        assert latency == 5200

    def test_is_circuit_open_false(self):
        """Test circuit breaker check when closed."""
        router = LLMRouter()
        
        is_open = router._is_circuit_open("openai", "gpt-3.5-turbo")
        
        assert is_open is False

    def test_is_circuit_open_true(self):
        """Test circuit breaker check when open."""
        router = LLMRouter()
        router.circuit_breakers["openai:gpt-3.5-turbo"] = True
        
        is_open = router._is_circuit_open("openai", "gpt-3.5-turbo")
        
        assert is_open is True

    @pytest.mark.asyncio
    async def test_cost_calculation(self):
        """Test cost is calculated correctly."""
        router = LLMRouter()
        # 4000 characters = 1000 tokens
        messages = [{"role": "user", "content": "x" * 4000}]
        
        decision = await router.route(messages, ModelProfile.FAST)
        
        # 1000 tokens at 0.0015 per 1k = 0.0015
        assert decision.estimated_cost == pytest.approx(0.0015, abs=0.0001)

    @pytest.mark.asyncio
    async def test_route_with_kwargs(self):
        """Test routing accepts additional kwargs."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Hello"}]
        
        # Should not fail with additional kwargs
        decision = await router.route(
            messages, 
            ModelProfile.FAST,
            temperature=0.7,
            max_tokens=100
        )
        
        assert decision is not None

    def test_estimate_tokens_empty_content(self):
        """Test token estimation with empty content."""
        router = LLMRouter()
        messages = [{"role": "user", "content": ""}]
        
        tokens = router._estimate_tokens(messages)
        
        # Empty content should still return minimum
        assert tokens == 10

    def test_estimate_tokens_missing_content(self):
        """Test token estimation with missing content key."""
        router = LLMRouter()
        messages = [{"role": "user"}]
        
        tokens = router._estimate_tokens(messages)
        
        # Missing content should be handled gracefully
        assert tokens == 10
