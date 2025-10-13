"""
Simple Agent implementation for HIL Agent System.

Based on HIL Agent System specifications:
- Stateless, single-shot execution
- ~1-2s latency target
- ~$0.001 per call cost
- 99%+ reliability
- Structured output support

Use Cases:
- Intent classification
- Sentiment analysis
- Entity extraction
- Simple Q&A
"""

import time
from typing import Any, Dict, List, Optional, Type

from jinja2 import Template
from pydantic import BaseModel

from app.core.exceptions import AgentExecutionError, LLMProviderError, LLMRoutingError
from app.core.llm_providers import LLMProviderFactory
from app.core.llm_router import LLMRouter, ModelProfile
from app.core.logging import get_logger

logger = get_logger(__name__)


class SimpleAgent:
    """
    Simple Agent for stateless, single-shot AI tasks.

    Characteristics:
    - ⚡ Latency: ~1-2s
    - 💰 Cost: ~$0.001 per call
    - 🎯 Reliability: 99%+
    - 🔄 No iteration loop
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        model_profile: str = "fast",
        output_schema: type[BaseModel] | None = None,
        temperature: float = 0.0,
    ):
        """
        Initialize Simple Agent.

        Args:
            llm_router: LLM router for model selection
            model_profile: Performance profile (fast/balanced/powerful)
            output_schema: Pydantic model for structured output
            temperature: LLM temperature (0.0 for deterministic)
        """
        if llm_router is None:
            raise TypeError("llm_router cannot be None")

        self.llm_router = llm_router
        self.model_profile = model_profile
        self.output_schema = output_schema
        self.temperature = temperature

        # Execution tracking
        self.last_execution_time: float | None = None
        self.last_execution_cost: float | None = None

        logger.info(
            "Simple Agent initialized",
            extra={
                "model_profile": model_profile,
                "temperature": temperature,
                "has_output_schema": output_schema is not None,
            },
        )

    async def run(
        self,
        prompt: str,
        input_data: dict[str, Any],
        system_prompt: str | None = None,
    ) -> Any:
        """
        Execute simple agent task.

        Args:
            prompt: Jinja2 template string for user prompt
            input_data: Data to render in prompt template
            system_prompt: Optional system prompt

        Returns:
            Response (BaseModel if output_schema provided, else dict)
        """
        start_time = time.time()

        try:
            # 1. Render prompt template
            rendered_prompt = self._render_prompt(prompt, input_data)
            logger.debug("Prompt rendered", extra={"length": len(rendered_prompt)})

            # 2. Build messages
            messages = self._build_messages(rendered_prompt, system_prompt)

            # 3. Route to optimal model
            try:
                routing_decision = await self.llm_router.route(
                    messages=messages, profile=ModelProfile(self.model_profile)
                )
                logger.debug(
                    "LLM routing decision",
                    extra={
                        "provider": routing_decision.provider,
                        "model": routing_decision.model,
                        "estimated_cost": routing_decision.estimated_cost,
                    },
                )
            except Exception as e:
                logger.error(f"LLM routing failed: {e}")
                raise LLMRoutingError(f"Failed to route request: {e}")

            # 4. Create provider and call LLM
            try:
                provider = LLMProviderFactory.create(
                    provider=routing_decision.provider,
                    model=routing_decision.model,
                    temperature=self.temperature,
                )

                if self.output_schema:
                    response = await provider.complete_structured(
                        messages=messages,
                        schema=self.output_schema.model_json_schema()
                        if self.output_schema
                        else None,
                    )
                else:
                    response_text = await provider.complete(messages=messages)
                    response = {"response": response_text}

                logger.debug(
                    "LLM response received",
                    extra={"response_keys": list(response.keys())},
                )

            except Exception as e:
                logger.error(f"LLM provider failed: {e}")
                raise LLMProviderError(f"LLM provider error: {e}")

            # 5. Validate and return structured output
            if self.output_schema:
                try:
                    validated_response = self.output_schema(**response)
                    logger.debug("Response validated against schema")
                    result = validated_response
                except Exception as e:
                    logger.error(f"Output validation failed: {e}")
                    raise ValueError(f"Output validation failed: {e}")
            else:
                result = response

            # 6. Track execution metrics
            self.last_execution_time = time.time() - start_time
            self.last_execution_cost = routing_decision.estimated_cost

            logger.info(
                "Simple Agent execution completed",
                extra={
                    "execution_time": self.last_execution_time,
                    "cost": self.last_execution_cost,
                    "model": routing_decision.model,
                },
            )

            return result

        except (LLMRoutingError, LLMProviderError) as e:
            # Re-raise known errors
            raise e
        except Exception as e:
            logger.error(f"Simple Agent execution failed: {e}")
            raise AgentExecutionError(f"Agent execution failed: {e}")

    def _render_prompt(self, prompt: str, input_data: dict[str, Any]) -> str:
        """Render Jinja2 prompt template with input data."""
        try:
            template = Template(prompt)
            return template.render(**input_data)
        except Exception as e:
            raise AgentExecutionError(f"Prompt template rendering failed: {e}")

    def _build_messages(
        self, user_prompt: str, system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        """Build messages list for LLM."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def get_metadata(self) -> dict[str, Any]:
        """Get agent metadata and configuration."""
        return {
            "type": "simple",
            "model_profile": self.model_profile,
            "temperature": self.temperature,
            "has_output_schema": self.output_schema is not None,
            "version": "1.0.0",
            "last_execution_time": self.last_execution_time,
            "last_execution_cost": self.last_execution_cost,
        }
