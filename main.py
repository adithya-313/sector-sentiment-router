"""
FastAPI application entry-point for the Multi-Tenant ABSA Engine.

Initialises the web server, configures middleware, and exposes
observability endpoints.
"""

from fastapi import FastAPI
from config import settings
from api.ingestion import router as ingestion_router

# ── FastAPI application instance ──────────────────────────────────────
app = FastAPI(
    title="Multi-Tenant ABSA Engine",
    description="Aspect-Based Sentiment Analysis for multiple tenants.",
    version="0.1.0",
)


# ── Health check endpoint (observability / uptime monitoring) ─────────
# ── Mount ingestion routes ───────────────────────────────────────────
app.include_router(ingestion_router)


# ── Health check endpoint (observability / uptime monitoring) ─────────
@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a structured JSON response for load-balancers and monitors."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "service": "absa-engine",
    }
