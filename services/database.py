"""
DuckDB adapter for analytical persistence.

DuckDB is an embedded OLAP database — perfect for local analytics without
a separate server.  We store every extracted review in a normalised schema:

    reviews  ──1:N──►  aspect_metrics

This relational design means we can run fast aggregations per sector
(e.g. average confidence, most-mentioned aspects) using standard SQL.

Why two tables instead of one?
  - A single review can mention *many* aspects (battery_life, screen, price).
  - Storing them in a separate table avoids repeating review-level columns
    and makes it easy to query across all aspects independently.
"""

import duckdb
from datetime import datetime, timezone

from domain.base_schemas import ExtractedReview
from domain.tenant_schemas import TenantSector


# Path where the DuckDB database file will be created.
_DB_PATH = "absa_analytics.db"


class AnalyticsDatabase:
    """Persistent storage for extracted reviews and aspect-level metrics.

    Uses DuckDB for zero-configuration, serverless analytics.
    """

    def __init__(self, read_only: bool = False) -> None:
        """Connect to the DuckDB database.

        Args:
            read_only: If True, open in read-only mode.  This allows
                multiple processes to query the same file concurrently.
        """
        self._conn = duckdb.connect(_DB_PATH, read_only=read_only)

        # Initialise tables only when we intend to write.
        if not read_only:
            self._create_tables()

    # ── Schema initialisation ──────────────────────────────────────────

    def _create_tables(self) -> None:
        """Create the relational schema if it does not already exist.

        ``IF NOT EXISTS`` makes this operation **idempotent** — safe to call
        every time the application starts without dropping existing data.
        """
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                review_id   VARCHAR PRIMARY KEY,   -- unique per review
                text        VARCHAR,               -- raw review text
                sector      VARCHAR,               -- fintech / ecommerce
                timestamp   TIMESTAMP              -- when this was stored
            )
            """
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aspect_metrics (
                review_id   VARCHAR,               -- FK → reviews.review_id
                aspect      VARCHAR,               -- e.g. "shipping_speed"
                sentiment   VARCHAR,               -- positive / negative / neutral
                confidence  DOUBLE,                -- 0.0 – 1.0

                FOREIGN KEY (review_id) REFERENCES reviews(review_id)
            )
            """
        )

    # ── Write operations ──────────────────────────────────────────────

    def save_extracted_review(
        self,
        review: ExtractedReview,
        sector: TenantSector,
    ) -> None:
        """Persist an extracted review and all its aspect-level metrics.

        Args:
            review: The ExtractedReview returned by the LLM (or cache).
            sector: The tenant sector that was used during extraction.

        This method wraps both INSERTs in a single transaction so the
        write is atomic — either both rows appear or neither does (ACID).
        """
        now = datetime.now(timezone.utc)

        # Insert the review-level row (ignore if PK already exists).
        self._conn.execute(
            """
            INSERT OR IGNORE INTO reviews (review_id, text, sector, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            [review.review_id, review.text, sector.value, now],
        )

        # Insert each aspect as a separate row in aspect_metrics.
        for asp in review.aspects:
            self._conn.execute(
                """
                INSERT INTO aspect_metrics (review_id, aspect, sentiment, confidence)
                VALUES (?, ?, ?, ?)
                """,
                [review.review_id, asp.aspect, asp.sentiment, asp.confidence],
            )

    # ── Read / OLAP operations ─────────────────────────────────────────

    def get_sector_summary(self, sector: TenantSector) -> list[dict[str, object]]:
        """Return an analytical summary for the given sector.

        The query aggregates all reviews for a sector and calculates:
          - Total reviews processed.
          - Total aspect-sentiment pairs extracted.
          - Average confidence across all aspects.
          - Per-aspect breakdown (count, avg confidence).

        Returns:
            A list of dictionaries — one row per aspect in the sector,
            plus an overall "TOTAL" row at the top.
        """
        # ── Overall sector statistics ──────────────────────────────────
        row = self._conn.execute(
            """
            SELECT
                COUNT(DISTINCT r.review_id)  AS total_reviews,
                COUNT(*)                     AS total_aspects,
                ROUND(AVG(a.confidence), 4)  AS avg_confidence
            FROM reviews r
            JOIN aspect_metrics a  ON r.review_id = a.review_id
            WHERE r.sector = ?
            """,
            [sector.value],
        ).fetchone()

        # fetchone() returns a tuple or None; unpack safely.
        total_reviews: int = row[0] if row else 0
        total_aspects: int = row[1] if row else 0
        avg_confidence: float = row[2] if row else 0.0

        # ── Per-aspect breakdown ───────────────────────────────────────
        per_aspect = self._conn.execute(
            """
            SELECT
                a.aspect,
                COUNT(*)                  AS mention_count,
                ROUND(AVG(a.confidence), 4) AS avg_confidence
            FROM reviews r
            JOIN aspect_metrics a  ON r.review_id = a.review_id
            WHERE r.sector = ?
            GROUP BY a.aspect
            ORDER BY mention_count DESC
            """,
            [sector.value],
        ).fetchall()

        # ── Format as a list of dicts for easy display ─────────────────
        rows: list[dict[str, object]] = []

        # Add the overall rollup row.
        rows.append(
            {
                "aspect": "TOTAL",
                "total_reviews": total_reviews,
                "total_aspects": total_aspects,
                "avg_confidence": avg_confidence,
            }
        )

        # Add per-aspect detail rows.
        for aspect_name, count, avg_conf in per_aspect:
            rows.append(
                {
                    "aspect": aspect_name,
                    "total_reviews": None,  # not applicable per-aspect
                    "total_aspects": count,
                    "avg_confidence": avg_conf,
                }
            )

        return rows
