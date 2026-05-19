"""
End-to-End (E2E) simulation script for the Multi-Tenant ABSA Engine.

This script:
  1. Starts the FastAPI server in the background.
  2. Submits fake reviews for two sectors: Fintech (Razorpay) and
     Quick-Commerce / E-Commerce (Zepto).
  3. Queries the DuckDB analytics database directly and prints a
     formatted summary table.

Run it from the project root:
    python test_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path so that domain / services imports work.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
from domain.tenant_schemas import TenantSector  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 18999
BASE_URL = f"http://{HOST}:{PORT}"
ANALYZE_URL = f"{BASE_URL}/api/v1/analyze"

# ── Test data mimicking real-world reviews ────────────────────────────

TEST_REVIEWS: list[dict[str, Any]] = [
    # ── Fintech reviews (Razorpay / PayU style) ──────────────────────
    {
        "text": "Razorpay's transaction fees are way too high for small businesses.",
        "tenant_id": "razorpay_001",
        "tenant_sector": "fintech",
    },
    {
        "text": "Love the mobile app UX — super clean and easy to navigate.",
        "tenant_id": "razorpay_002",
        "tenant_sector": "fintech",
    },
    {
        "text": "Customer support resolved my payment issue within minutes.",
        "tenant_id": "razorpay_003",
        "tenant_sector": "fintech",
    },
    # ── Quick-Commerce reviews (Zepto / Blinkit style) ───────────────
    {
        "text": "Zepto delivered my groceries in 8 minutes, incredible speed!",
        "tenant_id": "zepto_001",
        "tenant_sector": "ecommerce",
    },
    {
        "text": "The vegetables were stale and the packaging was damaged.",
        "tenant_id": "zepto_002",
        "tenant_sector": "ecommerce",
    },
    {
        "text": "Returns process was smooth — they picked up the wrong item the next day.",
        "tenant_id": "zepto_003",
        "tenant_sector": "ecommerce",
    },
]


def _print_separator(char: str = "=", width: int = 72) -> None:
    """Print a horizontal rule for clean terminal output."""
    print(char * width)


def _print_table(rows: list[dict[str, object]]) -> None:
    """Print a human-readable table from the summary rows."""
    _print_separator()
    print(
        f"{'ASPECT':<25} {'REVIEWS':>8} {'ASPECTS':>8} {'AVG CONF':>10}"
    )
    _print_separator("-")

    for entry in rows:
        aspect = entry["aspect"]
        total_rev = str(entry["total_reviews"]) if entry["total_reviews"] is not None else "-"
        total_asp = entry["total_aspects"]
        avg_conf = entry["avg_confidence"]
        print(
            f"{aspect:<25} {total_rev:>8} {str(total_asp):>8} {avg_conf:>10.4f}"
        )

    _print_separator()


# ── Main E2E flow ────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate the full end-to-end test."""

    # 1. Start the server as a background subprocess.
    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            HOST,
            "--port",
            str(PORT),
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Give the server time to boot.
    time.sleep(4)

    # Delete any old database so we start clean.
    db_path = PROJECT_ROOT / "absa_analytics.db"
    if db_path.exists():
        db_path.unlink()

    try:
        # 2. Verify the server is alive.
        health_resp = httpx.get(f"{BASE_URL}/health", timeout=10)
        health_resp.raise_for_status()
        print("[PASS] Server is healthy.\n")

        # 3. Submit each test review to the /analyze endpoint.
        for idx, review in enumerate(TEST_REVIEWS, start=1):
            resp = httpx.post(
                ANALYZE_URL,
                json=review,
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
            cached_mark = " (cached)" if body["cached"] else ""
            print(
                f"  [{idx}/{len(TEST_REVIEWS)}] "
                f"{review['tenant_id']:>15} -> "
                f"{len(body['result']['aspects'])} aspect(s){cached_mark}"
            )

        print("\n[PASS] All reviews submitted successfully.\n")

        # 4. Query sector summaries via the API.
        for sector in (TenantSector.FINTECH, TenantSector.ECOMMERCE):
            print(f"\n  Sector: {sector.value.upper()}")
            resp = httpx.get(
                f"{BASE_URL}/api/v1/summary/{sector.value}",
                timeout=10,
            )
            resp.raise_for_status()
            _print_table(resp.json())

        print("[PASS] E2E pipeline completed successfully.")

    finally:
        # 5. Clean up the server process.
        server_proc.terminate()
        server_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
