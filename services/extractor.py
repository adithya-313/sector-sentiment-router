"""
LLM API Adapter for Aspect-Based Sentiment Analysis.

We use the `instructor` library to patch the OpenAI client so that it
automatically validates LLM responses against our Pydantic schemas.
The client is configured to talk to a local Ollama instance (not the
real OpenAI API), which keeps inference free and data private.

Retry with exponential backoff makes the system resilient to transient
network issues or temporary LLM overload.
"""

import time
import traceback

import instructor
from instructor import Mode
from openai import OpenAI

from config import settings
from domain.base_schemas import AspectSentiment, ExtractedReview
from domain.tenant_schemas import TenantSector


# ── Sector-specific aspect rules injected into every prompt ──────────
# These are STRICT canonical categories. The LLM MUST NOT invent new ones.
SECTOR_RULES: dict[TenantSector, list[str]] = {
    TenantSector.FINTECH: [
        "transaction_speed",
        "payment_gateway_reliability",
        "customer_support",
        "onboarding",
        "fees_and_charges",
        "app_ux",
    ],
    TenantSector.ECOMMERCE: [
        "delivery_speed",
        "product_quality",
        "packaging_integrity",
        "customer_support",
        "refunds_and_returns",
        "app_ux",
    ],
}


def _build_system_prompt(sector: TenantSector) -> str:
    """Create the system prompt that tells the LLM which aspects to look for.

    The prompt includes sector-specific example aspects so the model
    focuses on domain-relevant features.
    """
    aspects = ", ".join(SECTOR_RULES[sector])
    return (
        f"You are an ABSA (Aspect-Based Sentiment Analysis) assistant. "
        f"Analyse the given review and extract sentiment for each relevant aspect. "
        f"For the {sector.value} sector, focus on aspects such as: {aspects}. "
        f"You MUST categorize the aspect into one of these strict canonical categories "
        f"based on the sector. DO NOT invent new categories. "
        f"Always respond with a valid JSON object that matches the required schema."
    )


# Maximum number of retry attempts for LLM calls.
_MAX_RETRIES = 3

# Base delay in seconds before the first retry (doubles each attempt).
_BASE_DELAY_S = 1.0


class AbsaExtractor:
    """Service that extracts aspect sentiments from review text using an LLM.

    Uses instructor-patched OpenAI client pointed at a local Ollama server
    for zero-cost, private inference.
    """

    def __init__(self) -> None:
        # ── Create a raw OpenAI client pointed at Ollama ──────────────
        raw_client = OpenAI(
            base_url=settings.OLLAMA_BASE_URL + "/v1",
            # Ollama does not require an API key, but the SDK expects one.
            api_key="ollama",
        )

        # ── Patch with instructor for automatic schema validation ─────
        # This means we can pass our Pydantic model as `response_model`
        # and instructor will validate + retry the LLM response for us.
        self.client = instructor.from_openai(raw_client, mode=Mode.JSON)

    def extract_aspects(
        self,
        text: str,
        tenant_id: str,
        tenant_sector: TenantSector,
    ) -> ExtractedReview:
        """Analyse a review and return structured aspect-sentiment pairs.

        Implements retry with exponential backoff for resilience against
        transient LLM failures.

        Args:
            text:          The raw review text.
            tenant_id:     Identifies which tenant the review belongs to.
            tenant_sector: Determines which aspects are relevant.

        Returns:
            An ExtractedReview containing aspect-sentiment pairs.
        """
        system_prompt = _build_system_prompt(tenant_sector)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                # Instructor will call the LLM and validate the response
                # against the ExtractedReview schema automatically.
                result: ExtractedReview = self.client.chat.completions.create(
                    model="phi3",
                    response_model=ExtractedReview,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                )
                return result

            except Exception:
                # Write full diagnostic trace to a file for debugging.
                with open("diagnostic_trace.txt", "w", encoding="utf-8") as f:
                    f.write("=== EXCEPTION TRACE ===\n")
                    f.write(traceback.format_exc())
                    f.write("\n\n=== RAW TEXT PAYLOAD ===\n")
                    f.write(text)
                    f.write("\n\n=== TENANT CONTEXT ===\n")
                    f.write(f"tenant_id={tenant_id}, sector={tenant_sector.value}\n")
                if attempt < _MAX_RETRIES:
                    # Exponential backoff: wait 1s, 2s, 4s before retrying.
                    delay = _BASE_DELAY_S * (2 ** (attempt - 1))
                    time.sleep(delay)

        # All retries exhausted — return a safe fallback so the API
        # can still respond gracefully instead of crashing.
        return ExtractedReview(
            review_id=tenant_id,
            text=text,
            aspects=[
                AspectSentiment(
                    aspect="fallback",
                    sentiment="neutral",
                    confidence=0.0,
                )
            ],
        )
