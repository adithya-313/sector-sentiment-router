"""
Text ingestion router.

Exposes a POST endpoint that accepts raw review text, checks a
deduplication cache, and — on a cache miss — invokes the LLM-based
extractor to produce aspect-sentiment pairs.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from fastapi import HTTPException

from domain.base_schemas import ExtractedReview
from domain.tenant_schemas import TenantSector
from services.cache import LocalSemanticCache
from services.database import AnalyticsDatabase
from services.extractor import AbsaExtractor

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


# ── Request / Response schemas ───────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Payload accepted by the POST /analyze endpoint."""

    # The raw review text submitted by the user.
    text: str = Field(..., min_length=1, description="Raw review text.")

    # Unique identifier for the tenant (used for cache key scoping).
    tenant_id: str = Field(..., min_length=1, description="Tenant identifier.")

    # Sector the tenant operates in (controls aspect relevance).
    tenant_sector: TenantSector = Field(
        ...,
        description="Sector that determines which aspects are relevant.",
    )


class AnalyzeResponse(BaseModel):
    """Response returned after analysing a review."""

    # Indicates whether the result came from cache or a fresh extraction.
    cached: bool = Field(..., description="Whether the result was served from cache.")

    # The extracted review with aspect-sentiment pairs.
    result: ExtractedReview


# ── Dependency factories (singletons) ────────────────────────────────
# Using simple functions so FastAPI's Depends() can wire them automatically.
# This is Dependency Injection — it decouples route logic from service
# initialisation and makes testing easier (just swap the factory).

def get_cache() -> LocalSemanticCache:
    """Return the singleton cache instance."""
    return LocalSemanticCache()


def get_extractor() -> AbsaExtractor:
    """Return the singleton extractor instance."""
    return AbsaExtractor()


# Singleton database instance — reuses the same DuckDB connection.
_db_instance: AnalyticsDatabase | None = None


def get_database() -> AnalyticsDatabase:
    """Return the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = AnalyticsDatabase()
    return _db_instance


# ── Routes ───────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_review(
    request: AnalyzeRequest,
    cache: LocalSemanticCache = Depends(get_cache),
    extractor: AbsaExtractor = Depends(get_extractor),
    database: AnalyticsDatabase = Depends(get_database),
) -> AnalyzeResponse:
    """Analyse a review for aspect-based sentiment.

    Workflow:
      1. Check the idempotency cache for (text, tenant_id).
      2. On cache HIT  → return the cached result immediately.
      3. On cache MISS → call the LLM extractor, store the result.
      4. Persist the result to DuckDB for analytical queries.
      5. Return the response.
    """
    # Step 1: Check cache for an existing result (idempotency).
    cached_result = cache.get(request.text, request.tenant_id)
    if cached_result is not None:
        extracted = ExtractedReview(**cached_result)
        is_cached = True
    else:
        # Step 2: Cache miss — run extraction via the LLM adapter.
        extracted = extractor.extract_aspects(
            text=request.text,
            tenant_id=request.tenant_id,
            tenant_sector=request.tenant_sector,
        )
        # Step 3: Store in cache so future identical calls are idempotent.
        cache.set(request.text, request.tenant_id, extracted.model_dump())
        is_cached = False

    # Step 4: Persist to DuckDB for downstream analytics.
    # INSERT OR IGNORE makes this safe on re-runs from cache.
    database.save_extracted_review(extracted, request.tenant_sector)

    return AnalyzeResponse(cached=is_cached, result=extracted)


@router.get("/summary/{sector}")
def get_sector_summary(
    sector: TenantSector,
    database: AnalyticsDatabase = Depends(get_database),
) -> list[dict[str, object]]:
    """Return analytical summary for a given sector."""
    try:
        return database.get_sector_summary(sector)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
