"""
LLM Provider factory for managing different LLM providers.

Supports OpenAI, Anthropic, and other providers with unified interface.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, **kwargs):
        """Initialize provider with model and configuration."""
        self.model = model
        self.config = kwargs

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Complete chat with unstructured response."""

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Complete chat with structured response."""


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and development."""

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Mock completion returning test response."""
        return "Mock LLM response for testing"

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Mock structured completion."""
        # Return mock data that matches common schemas
        if schema:
            # Try to generate mock data based on schema
            return self._generate_mock_from_schema(schema)

        # Default mock response for intent classification
        return {
            "intent": "mock_intent",
            "confidence": 0.95,
            "entities": {"test": "value"},
        }

    def _generate_mock_from_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Generate mock data from JSON schema."""
        properties = schema.get("properties", {})
        mock_data = {}

        for field, field_schema in properties.items():
            field_type = field_schema.get("type", "string")

            if field_type == "string":
                mock_data[field] = f"mock_{field}"
            elif field_type == "number":
                mock_data[field] = 0.95
            elif field_type == "integer":
                mock_data[field] = 1
            elif field_type == "boolean":
                mock_data[field] = True
            elif field_type == "object":
                mock_data[field] = {"test": "value"}
            elif field_type == "array":
                mock_data[field] = ["test"]
            else:
                mock_data[field] = None

        return mock_data


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Complete using OpenAI API."""
        # TODO: Implement actual OpenAI integration
        # For now, return mock response
        return "OpenAI response (not implemented)"

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Complete with structured output using OpenAI."""
        # TODO: Implement actual OpenAI structured output
        # For now, return mock response
        return {"intent": "openai_mock_intent", "confidence": 0.92, "entities": {}}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider."""

    async def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Complete using Anthropic API."""
        # TODO: Implement actual Anthropic integration
        return "Anthropic response (not implemented)"

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Complete with structured output using Anthropic."""
        # TODO: Implement actual Anthropic structured output
        return {"intent": "anthropic_mock_intent", "confidence": 0.88, "entities": {}}


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mock": MockLLMProvider,
    }

    @classmethod
    def create(cls, provider: str, model: str, **kwargs) -> LLMProvider:
        """
        Create LLM provider instance.

        Args:
            provider: Provider name (openai, anthropic, mock)
            model: Model name
            **kwargs: Additional configuration

        Returns:
            LLMProvider instance
        """
        if provider not in cls._providers:
            raise ValueError(f"Unknown provider: {provider}")

        provider_class = cls._providers[provider]
        return provider_class(model=model, **kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type[LLMProvider]) -> None:
        """Register a new provider."""
        cls._providers[name] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        """List available providers."""
        return list(cls._providers.keys())
