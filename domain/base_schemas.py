"""
Base Pydantic schemas for Aspect-Based Sentiment Analysis.

These models define the core data contracts used across the entire engine.
Every ABSA pipeline reads and produces these shapes.
"""

from pydantic import BaseModel, Field
from typing import List


class AspectSentiment(BaseModel):
    """A single aspect (feature) of a product/service and its sentiment.

    Example:
        aspect="battery_life", sentiment="positive", confidence=0.95
    """

    # The feature being discussed, e.g. "battery_life", "customer_service".
    aspect: str = Field(
        ...,
        description=(
            "The product or service aspect being evaluated. "
            "Must be one of the strict canonical categories for the sector: "
            "FINTECH: transaction_speed, payment_gateway_reliability, "
            "customer_support, onboarding, fees_and_charges, app_ux. "
            "ECOMMERCE: delivery_speed, product_quality, packaging_integrity, "
            "customer_support, refunds_and_returns, app_ux."
        ),
        min_length=1,
    )

    # Sentiment label: positive, negative, or neutral.
    sentiment: str = Field(
        ...,
        description="Predicted sentiment for this aspect.",
        pattern=r"^(positive|negative|neutral)$",
    )

    # Confidence score between 0.0 and 1.0.
    confidence: float = Field(
        ...,
        description="Confidence of the sentiment prediction (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )


class ExtractedReview(BaseModel):
    """A full review that has been analysed for aspect-based sentiment.

    Contains the original text plus the list of aspect-sentiment pairs.
    """

    # Unique identifier for the review (e.g. a database primary key).
    review_id: str = Field(
        ...,
        description="Unique identifier for the review.",
        min_length=1,
    )

    # The raw review text as provided by the user.
    text: str = Field(
        ...,
        description="Original review text.",
        min_length=1,
    )

    # One or more aspect-sentiment pairs extracted from the text.
    aspects: List[AspectSentiment] = Field(
        ...,
        description="List of aspect-sentiment pairs extracted from the review.",
        min_length=1,
    )
