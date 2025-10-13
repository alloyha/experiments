"""
LLM Router for optimal model selection and routing.

Based on HIL Agent System specifications for intelligent LLM routing
and cost optimization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ModelProfile(str, Enum):
    """Model performance profiles for different use cases."""

    FAST = "fast"  # GPT-3.5, Claude Haiku - low cost, high speed
    BALANCED = "balanced"  # GPT-4-turbo, Claude Sonnet - balanced cost/quality
    POWERFUL = "powerful"  # GPT-4, Claude Opus - high quality, higher cost


@dataclass
class RoutingDecision:
    """Decision made by LLM router."""

    provider: str
    model: str
    estimated_cost: float
    estimated_latency: int  # milliseconds
    profile: ModelProfile


class LLMRouter:
    """
    Intelligent LLM routing for cost and performance optimization.

    Routes requests to optimal models based on:
    - Performance profile (fast/balanced/powerful)
    - Token estimation and budgeting
    - Circuit breakers per model
    - Automatic fallback chains
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize LLM router with configuration."""
        self.config = config or {}
        self.model_configs = self._load_model_configs()
        self.circuit_breakers = {}

    def _load_model_configs(self) -> dict[str, dict[str, Any]]:
        """Load model configurations for routing decisions."""
        return {
            ModelProfile.FAST: {
                "primary": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "cost_per_1k": 0.0015,
                },
                "fallback": {
                    "provider": "anthropic",
                    "model": "claude-3-haiku",
                    "cost_per_1k": 0.0025,
                },
            },
            ModelProfile.BALANCED: {
                "primary": {
                    "provider": "openai",
                    "model": "gpt-4-turbo",
                    "cost_per_1k": 0.01,
                },
                "fallback": {
                    "provider": "anthropic",
                    "model": "claude-3-sonnet",
                    "cost_per_1k": 0.03,
                },
            },
            ModelProfile.POWERFUL: {
                "primary": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "cost_per_1k": 0.03,
                },
                "fallback": {
                    "provider": "anthropic",
                    "model": "claude-3-opus",
                    "cost_per_1k": 0.075,
                },
            },
        }

    async def route(
        self,
        messages: list[dict[str, str]],
        profile: ModelProfile = ModelProfile.FAST,
        **kwargs,
    ) -> RoutingDecision:
        """
        Route request to optimal model based on profile and current state.

        Args:
            messages: List of chat messages
            profile: Performance profile (fast/balanced/powerful)
            **kwargs: Additional routing parameters

        Returns:
            RoutingDecision with selected provider and model
        """
        # Estimate token count for cost calculation
        estimated_tokens = self._estimate_tokens(messages)

        # Get model config for profile
        config = self.model_configs[profile]

        # Check circuit breaker status
        primary_config = config["primary"]
        if self._is_circuit_open(primary_config["provider"], primary_config["model"]):
            # Use fallback
            selected_config = config["fallback"]
        else:
            selected_config = primary_config

        # Calculate estimated cost and latency
        estimated_cost = (estimated_tokens / 1000) * selected_config["cost_per_1k"]
        estimated_latency = self._estimate_latency(profile, estimated_tokens)

        return RoutingDecision(
            provider=selected_config["provider"],
            model=selected_config["model"],
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            profile=profile,
        )

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate token count for messages."""
        # Simple estimation: ~4 chars per token
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return max(total_chars // 4, 10)  # Minimum 10 tokens

    def _estimate_latency(self, profile: ModelProfile, tokens: int) -> int:
        """Estimate response latency in milliseconds."""
        base_latency = {
            ModelProfile.FAST: 800,
            ModelProfile.BALANCED: 2000,
            ModelProfile.POWERFUL: 5000,
        }

        # Add token-based latency
        token_latency = tokens * 2  # ~2ms per token
        return base_latency[profile] + token_latency

    def _is_circuit_open(self, provider: str, model: str) -> bool:
        """Check if circuit breaker is open for provider/model."""
        key = f"{provider}:{model}"
        return self.circuit_breakers.get(key, False)
