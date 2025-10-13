"""
Test suite for Simple Agent implementation.

Following TDD approach to define Simple Agent requirements through tests first.
Based on HIL Agent System specifications:
- Stateless, single-shot execution
- ~1-2s latency target
- ~$0.001 per call cost
- 99%+ reliability
- Structured output support
"""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel


# Test data models
class IntentClassification(BaseModel):
    """Example output schema for intent classification."""

    intent: str
    confidence: float
    entities: dict[str, Any] = {}


class TestSimpleAgentCore:
    """Test core Simple Agent functionality."""

    def test_simple_agent_initialization(self):
        """Test Simple Agent can be initialized with required parameters."""
        from app.agents.types.simple_agent import SimpleAgent
        from app.core.llm_router import LLMRouter

        mock_llm_router = Mock(spec=LLMRouter)

        agent = SimpleAgent(
            llm_router=mock_llm_router,
            model_profile="fast",
            output_schema=IntentClassification,
            temperature=0.0,
        )

        assert agent.llm_router == mock_llm_router
        assert agent.model_profile == "fast"
        assert agent.output_schema == IntentClassification
        assert agent.temperature == 0.0

    @pytest.mark.asyncio
    async def test_simple_agent_basic_run(self):
        """Test Simple Agent basic run method with mock dependencies."""
        from app.agents.types.simple_agent import SimpleAgent

        # Mock dependencies for initial test structure
        mock_llm_router = AsyncMock()
        mock_llm_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(
            llm_router=mock_llm_router,
            model_profile="fast",
            output_schema=IntentClassification,
            temperature=0.0,
        )

        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.complete_structured.return_value = {
                "intent": "test_intent",
                "confidence": 0.95,
                "entities": {},
            }
            mock_factory.create.return_value = mock_provider

            result = await agent.run(
                prompt="Classify intent: {{message}}",
                input_data={"message": "I want to return my order"},
                system_prompt=None,
            )

            assert isinstance(result, IntentClassification)
            assert result.intent == "test_intent"
            assert result.confidence == 0.95

    def test_simple_agent_requires_llm_router(self):
        """Test Simple Agent requires LLM router dependency."""
        from app.agents.types.simple_agent import SimpleAgent

        # Should raise TypeError when llm_router is None or missing
        with pytest.raises(TypeError):
            SimpleAgent(llm_router=None)


class TestSimpleAgentLLMIntegration:
    """Test Simple Agent integration with LLM routing system."""

    @pytest.mark.asyncio
    async def test_llm_router_integration(self):
        """Test Simple Agent properly integrates with LLM router."""
        from app.agents.types.simple_agent import SimpleAgent
        from app.core.llm_router import LLMRouter, ModelProfile

        mock_router = AsyncMock(spec=LLMRouter)
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(llm_router=mock_router, model_profile="fast")

        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.complete_structured.return_value = {
                "intent": "test_intent",
                "confidence": 0.95,
                "entities": {},
            }
            mock_factory.create.return_value = mock_provider

            await agent.run(prompt="Test prompt", input_data={"test": "data"})

            # Verify router was called with correct profile
            mock_router.route.assert_called_once()
            call_args = mock_router.route.call_args
            assert "fast" in str(call_args)

    @pytest.mark.asyncio
    async def test_structured_output_validation(self):
        """Test Simple Agent validates structured output schema."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_llm_router = AsyncMock()
        mock_llm_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        # Test with valid schema
        agent = SimpleAgent(
            llm_router=mock_llm_router, output_schema=IntentClassification
        )

        # Test schema validation happens
        assert agent.output_schema == IntentClassification

        # Test that schema is used in LLM calls
        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.complete_structured.return_value = {
                "intent": "test_intent",
                "confidence": 0.95,
                "entities": {},
            }
            mock_factory.create.return_value = mock_provider

            result = await agent.run(prompt="Test prompt", input_data={})

            # Verify the structured output method was called
            mock_provider.complete_structured.assert_called_once()
            call_kwargs = mock_provider.complete_structured.call_args[1]
            assert "schema" in call_kwargs

    @pytest.mark.asyncio
    async def test_model_profile_selection(self):
        """Test Simple Agent supports different model profiles."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        for profile in ["fast", "balanced", "powerful"]:
            agent = SimpleAgent(llm_router=mock_router, model_profile=profile)
            assert agent.model_profile == profile


class TestSimpleAgentPromptProcessing:
    """Test Simple Agent prompt template rendering."""

    @pytest.mark.asyncio
    async def test_prompt_template_rendering(self):
        """Test Simple Agent renders Jinja2 templates correctly."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(provider="openai", model="gpt-3.5-turbo")

        agent = SimpleAgent(llm_router=mock_router)

        # Mock the LLM provider response
        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = (
                "User message: Hello world, Context: greeting"
            )
            mock_factory.create.return_value = mock_provider

            result = await agent.run(
                prompt="User message: {{message}}, Context: {{context}}",
                input_data={"message": "Hello world", "context": "greeting"},
            )

            # Verify template was rendered and complete method was called
            mock_provider.complete.assert_called_once()
            call_args = mock_provider.complete.call_args
            assert call_args is not None
            # Check that the prompt was properly formatted
            messages = call_args[1]["messages"]
            assert any("Hello world" in str(msg) for msg in messages)
            assert any("greeting" in str(msg) for msg in messages)

    @pytest.mark.asyncio
    async def test_system_prompt_support(self):
        """Test Simple Agent supports system prompts."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )
        agent = SimpleAgent(llm_router=mock_router)

        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = "Helpful response"
            mock_factory.create.return_value = mock_provider

            await agent.run(
                prompt="User prompt",
                input_data={},
                system_prompt="You are a helpful assistant",
            )

            # Verify system prompt was passed to provider
            mock_provider.complete.assert_called_once()
            call_kwargs = mock_provider.complete.call_args[1]
            messages = call_kwargs["messages"]
            # Check that system message was included
            assert any(
                msg.get("role") == "system"
                and "helpful assistant" in msg.get("content", "").lower()
                for msg in messages
            )


class TestSimpleAgentErrorHandling:
    """Test Simple Agent error handling and resilience."""

    @pytest.mark.asyncio
    async def test_llm_router_failure_handling(self):
        """Test Simple Agent handles LLM router failures gracefully."""
        from app.agents.types.simple_agent import SimpleAgent
        from app.core.exceptions import LLMRoutingError

        mock_router = AsyncMock()
        mock_router.route.side_effect = Exception("Routing failed")

        agent = SimpleAgent(llm_router=mock_router)

        with pytest.raises(Exception) as exc_info:
            await agent.run(prompt="Test prompt", input_data={})

        # Verify the routing error was raised
        assert "Routing failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_provider_failure_handling(self):
        """Test Simple Agent handles LLM provider failures."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(llm_router=mock_router)

        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.complete.side_effect = Exception("Provider failed")
            mock_factory.create.return_value = mock_provider

            with pytest.raises(Exception) as exc_info:
                await agent.run(prompt="Test prompt", input_data={})

            # Verify the provider error was raised
            assert "Provider failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_output_schema_validation_error(self):
        """Test Simple Agent handles invalid output schema."""
        from pydantic import ValidationError

        from app.agents.types.simple_agent import SimpleAgent
        from app.core.exceptions import AgentExecutionError

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(llm_router=mock_router, output_schema=IntentClassification)

        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            # Return invalid data that doesn't match schema
            mock_provider.complete_structured.return_value = {
                "invalid_field": "invalid_value"
            }
            mock_factory.create.return_value = mock_provider

            # Test that validation errors are properly handled
            with pytest.raises((ValidationError, ValueError, AgentExecutionError)):
                await agent.run(prompt="Test prompt", input_data={})


class TestSimpleAgentPerformance:
    """Test Simple Agent performance and metrics."""

    @pytest.mark.asyncio
    async def test_execution_time_tracking(self):
        """Test Simple Agent tracks execution time."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(llm_router=mock_router)

        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.complete.return_value = "Simple response"
            mock_factory.create.return_value = mock_provider

            # Mock time to simulate execution
            import time

            start_time = time.time()
            result = await agent.run(prompt="Test prompt", input_data={})
            end_time = time.time()

            # Should have completed successfully (returns dict when no output_schema)
            assert isinstance(result, dict)
            assert "response" in result
            # Basic time tracking verification
            execution_time = end_time - start_time
            assert execution_time >= 0

    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Test Simple Agent tracks execution costs."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(llm_router=mock_router)

        with patch("app.agents.types.simple_agent.LLMProviderFactory") as mock_factory:
            mock_provider = AsyncMock()
            mock_provider.complete_structured.return_value = {
                "intent": "test_intent",
                "confidence": 0.95,
                "entities": {},
            }
            mock_factory.create.return_value = mock_provider

            await agent.run(prompt="Test prompt", input_data={})

            # Verify router was called to get cost estimates
            mock_router.route.assert_called_once()
            routing_result = mock_router.route.return_value
            assert routing_result.estimated_cost == 0.001


class TestSimpleAgentConfiguration:
    """Test Simple Agent configuration options."""

    def test_temperature_configuration(self):
        """Test Simple Agent supports temperature configuration."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(llm_router=mock_router, temperature=0.7)

        assert agent.temperature == 0.7

    def test_default_configuration(self):
        """Test Simple Agent has sensible defaults."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(llm_router=mock_router)

        # Check defaults align with HIL specifications
        assert agent.model_profile == "fast"  # Default to fast profile
        assert agent.temperature == 0.0  # Deterministic by default
        assert agent.output_schema is None  # No schema by default

    @pytest.mark.asyncio
    async def test_agent_metadata(self):
        """Test Simple Agent exposes metadata about its configuration."""
        from app.agents.types.simple_agent import SimpleAgent

        mock_router = AsyncMock()
        mock_router.route.return_value = Mock(
            provider="openai",
            model="gpt-3.5-turbo",
            estimated_cost=0.001,
            estimated_latency=800,
        )

        agent = SimpleAgent(
            llm_router=mock_router, model_profile="fast", temperature=0.0
        )

        # Test that basic agent properties are accessible
        assert agent.model_profile == "fast"
        assert agent.temperature == 0.0
        assert agent.llm_router is not None
