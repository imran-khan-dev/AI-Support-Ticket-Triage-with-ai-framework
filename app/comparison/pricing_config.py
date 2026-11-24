"""Model pricing configuration for F4.5 cost estimation.

Prices are per 1 million tokens, in USD.  Sources are documented per model.
Prices are kept separate from execution logic so they can be updated without
touching the benchmark or metrics code.

Never hardcode an unexplained price.  If a model is not listed, its cost is
reported as unavailable (``None``) rather than estimated.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelPrice(BaseModel):
    """Pricing for a single model/provider combination."""

    model: str = Field(..., description="Model identifier.")
    provider: str = Field(..., description="Provider name (e.g. 'groq').")
    input_price_per_1m_tokens: float = Field(
        ..., ge=0, description="Input token price per 1M tokens (USD)."
    )
    output_price_per_1m_tokens: float = Field(
        ..., ge=0, description="Output token price per 1M tokens (USD)."
    )
    pricing_source: str = Field(
        ..., description="URL or document where the pricing was obtained."
    )
    pricing_date: str = Field(
        ..., description="Date the pricing was verified (YYYY-MM-DD)."
    )
    currency: str = Field(
        default="USD", description="Currency of the prices (ISO 4217 code)."
    )

    def input_cost_usd(self, tokens: int) -> float:
        """Estimated input cost for the given token count."""
        return tokens * self.input_price_per_1m_tokens / 1_000_000

    def output_cost_usd(self, tokens: int) -> float:
        """Estimated output cost for the given token count."""
        return tokens * self.output_price_per_1m_tokens / 1_000_000

    def total_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Estimated total cost for the given token counts."""
        return self.input_cost_usd(input_tokens) + self.output_cost_usd(output_tokens)


# Groq pricing (per 1M tokens, USD).
# Source: https://groq.com/pricing (verified 2026-01-15).
GROQ_PRICING: dict[str, ModelPrice] = {
    "llama-3.1-8b-instant": ModelPrice(
        model="llama-3.1-8b-instant",
        provider="groq",
        input_price_per_1m_tokens=0.05,
        output_price_per_1m_tokens=0.05,
        pricing_source="https://groq.com/pricing",
        pricing_date="2026-01-15",
    ),
    "llama-3.1-70b-versatile": ModelPrice(
        model="llama-3.1-70b-versatile",
        provider="groq",
        input_price_per_1m_tokens=0.59,
        output_price_per_1m_tokens=0.79,
        pricing_source="https://groq.com/pricing",
        pricing_date="2026-01-15",
    ),
    "mixtral-8x7b-32768": ModelPrice(
        model="mixtral-8x7b-32768",
        provider="groq",
        input_price_per_1m_tokens=0.24,
        output_price_per_1m_tokens=0.24,
        pricing_source="https://groq.com/pricing",
        pricing_date="2026-01-15",
    ),
    "gemma2-9b-it": ModelPrice(
        model="gemma2-9b-it",
        provider="groq",
        input_price_per_1m_tokens=0.05,
        output_price_per_1m_tokens=0.05,
        pricing_source="https://groq.com/pricing",
        pricing_date="2026-01-15",
    ),
}


def get_model_price(model: str, provider: str = "groq") -> ModelPrice | None:
    """Return pricing for a model, or ``None`` if not configured."""
    return GROQ_PRICING.get(model)


def list_known_models() -> list[ModelPrice]:
    """Return all configured model prices."""
    return list(GROQ_PRICING.values())


class PricingConfig(BaseModel):
    """Pricing configuration for cost estimation.

    Holds the model, provider, and price lookup so cost calculation can be
    performed without hardcoding prices in execution logic.
    """

    model: str | None = Field(
        default=None, description="Model identifier used for pricing."
    )
    provider: str = Field(
        default="groq", description="Provider name used for pricing."
    )
    pricing_source: str | None = Field(
        default=None, description="Source of the pricing data."
    )
    pricing_date: str | None = Field(
        default=None, description="Date the pricing was verified."
    )
    input_price_per_1m_tokens: float | None = Field(
        default=None, ge=0, description="Input token price per 1M tokens (USD)."
    )
    output_price_per_1m_tokens: float | None = Field(
        default=None, ge=0, description="Output token price per 1M tokens (USD)."
    )
    currency: str = Field(
        default="USD", description="Currency of the prices (ISO 4217 code)."
    )

    def model_price(self) -> ModelPrice | None:
        """Return the configured model's price, or None if not configured."""
        if self.model is None:
            return None
        price = get_model_price(self.model, self.provider)
        if price is not None:
            return price
        # Fall back to explicit fields if set.
        if (
            self.input_price_per_1m_tokens is not None
            and self.output_price_per_1m_tokens is not None
        ):
            return ModelPrice(
                model=self.model or "unknown",
                provider=self.provider,
                input_price_per_1m_tokens=self.input_price_per_1m_tokens,
                output_price_per_1m_tokens=self.output_price_per_1m_tokens,
                pricing_source=self.pricing_source or "user_configured",
                pricing_date=self.pricing_date or "unknown",
                currency=self.currency,
            )
        return None

    def estimate_cost_usd(
        self, input_tokens: int, output_tokens: int
    ) -> float | None:
        """Estimated cost in USD, or None if pricing is unavailable."""
        price = self.model_price()
        if price is None:
            return None
        return price.total_cost_usd(input_tokens, output_tokens)
