"""
Fixed tests for LLM router that actually test the real interface.
"""

import pytest

from app.core.llm_router import LLMRouter, ModelProfile, RoutingDecision


class TestModelProfile:
    """Test ModelProfile enum."""

    def test_model_profile_values(self):
        """Test ModelProfile enum values."""
        assert ModelProfile.FAST == "fast"
        assert ModelProfile.BALANCED == "balanced"
        assert ModelProfile.POWERFUL == "powerful"


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """Test creating RoutingDecision."""
        decision = RoutingDecision(
            provider="openai",
            model="gpt-4",
            estimated_cost=0.05,
            estimated_latency=2000,
            profile=ModelProfile.BALANCED,
        )
        assert decision.provider == "openai"
        assert decision.model == "gpt-4"
        assert decision.estimated_cost == 0.05
        assert decision.estimated_latency == 2000
        assert decision.profile == ModelProfile.BALANCED


class TestLLMRouter:
    """Test LLM Router functionality."""

    def test_router_initialization(self):
        """Test router initialization."""
        router = LLMRouter()
        assert router.config == {}
        assert hasattr(router, "model_configs")
        assert hasattr(router, "circuit_breakers")

    def test_router_initialization_with_config(self):
        """Test router initialization with config."""
        config = {"test": "value"}
        router = LLMRouter(config)
        assert router.config == config

    def test_load_model_configs(self):
        """Test model configurations loading."""
        router = LLMRouter()
        configs = router.model_configs

        # Check all profiles are present
        assert ModelProfile.FAST in configs
        assert ModelProfile.BALANCED in configs
        assert ModelProfile.POWERFUL in configs

        # Check structure of fast profile
        fast_config = configs[ModelProfile.FAST]
        assert "primary" in fast_config
        assert "fallback" in fast_config
        assert fast_config["primary"]["provider"] == "openai"
        assert fast_config["primary"]["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_route_fast_profile(self):
        """Test routing with fast profile."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Hello world"}]

        decision = await router.route(messages, ModelProfile.FAST)

        assert isinstance(decision, RoutingDecision)
        assert decision.provider == "openai"
        assert decision.model == "gpt-3.5-turbo"
        assert decision.profile == ModelProfile.FAST
        assert decision.estimated_cost > 0
        assert decision.estimated_latency > 0

    @pytest.mark.asyncio
    async def test_route_balanced_profile(self):
        """Test routing with balanced profile."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Analyze this complex text"}]

        decision = await router.route(messages, ModelProfile.BALANCED)

        assert decision.provider == "openai"
        assert decision.model == "gpt-4-turbo"
        assert decision.profile == ModelProfile.BALANCED

    @pytest.mark.asyncio
    async def test_route_powerful_profile(self):
        """Test routing with powerful profile."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Complex reasoning task"}]

        decision = await router.route(messages, ModelProfile.POWERFUL)

        assert decision.provider == "openai"
        assert decision.model == "gpt-4"
        assert decision.profile == ModelProfile.POWERFUL

    @pytest.mark.asyncio
    async def test_route_default_profile(self):
        """Test routing with default (fast) profile."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Quick question"}]

        decision = await router.route(messages)

        assert decision.profile == ModelProfile.FAST

    def test_estimate_tokens(self):
        """Test token estimation."""
        router = LLMRouter()

        # Test basic estimation
        messages = [{"role": "user", "content": "Hello"}]
        tokens = router._estimate_tokens(messages)
        assert tokens >= 10  # Minimum 10 tokens

        # Test longer content
        long_messages = [{"role": "user", "content": "A" * 100}]
        long_tokens = router._estimate_tokens(long_messages)
        assert long_tokens > tokens

    def test_estimate_latency(self):
        """Test latency estimation."""
        router = LLMRouter()

        # Test fast profile
        fast_latency = router._estimate_latency(ModelProfile.FAST, 100)
        assert fast_latency >= 1000  # Base 800 + token latency

        # Test balanced profile
        balanced_latency = router._estimate_latency(ModelProfile.BALANCED, 100)
        assert balanced_latency > fast_latency

        # Test powerful profile
        powerful_latency = router._estimate_latency(ModelProfile.POWERFUL, 100)
        assert powerful_latency > balanced_latency

    def test_circuit_breaker_check(self):
        """Test circuit breaker functionality."""
        router = LLMRouter()

        # Initially no circuit breakers are open
        assert not router._is_circuit_open("openai", "gpt-4")

        # Set a circuit breaker
        router.circuit_breakers["openai:gpt-4"] = True
        assert router._is_circuit_open("openai", "gpt-4")

    @pytest.mark.asyncio
    async def test_route_with_circuit_breaker(self):
        """Test routing when circuit breaker is open."""
        router = LLMRouter()
        messages = [{"role": "user", "content": "Test"}]

        # Open circuit breaker for primary model
        router.circuit_breakers["openai:gpt-3.5-turbo"] = True

        decision = await router.route(messages, ModelProfile.FAST)

        # Should use fallback
        assert decision.provider == "anthropic"
        assert decision.model == "claude-3-haiku"

    @pytest.mark.asyncio
    async def test_cost_calculation(self):
        """Test cost calculation based on tokens."""
        router = LLMRouter()

        # Short message
        short_messages = [{"role": "user", "content": "Hi"}]
        short_decision = await router.route(short_messages, ModelProfile.FAST)

        # Long message
        long_messages = [{"role": "user", "content": "A" * 1000}]
        long_decision = await router.route(long_messages, ModelProfile.FAST)

        # Longer message should cost more
        assert long_decision.estimated_cost > short_decision.estimated_cost
