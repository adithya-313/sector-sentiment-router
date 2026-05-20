"""
Multi-Tenant ABSA Engine — Executive Dashboard.

Single-command monolithic UI that calls phi3 directly (no backend server).
Business-friendly layout with sentiment metrics, stacked charts, and
live review breakdowns.
"""

import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb  # noqa: E402
import plotly.graph_objects as go  # type: ignore[import-untyped]  # noqa: E402
import streamlit as st  # noqa: E402

from domain.base_schemas import ExtractedReview  # noqa: E402
from domain.tenant_schemas import TenantSector  # noqa: E402
from services.cache import LocalSemanticCache  # noqa: E402
from services.extractor import AbsaExtractor  # noqa: E402

# ── Page configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="ABSA Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS injection ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Force Apple System Font */
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    /* Apply Dark Gradient Background */
    .stApp {
        background: linear-gradient(145deg, #0f172a 0%, #000000 100%) !important;
    }
    /* Hide Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Colour palette ───────────────────────────────────────────────────
SENTIMENT_COLORS: dict[str, str] = {
    "positive": "#00CC96",
    "negative": "#EF553B",
    "neutral": "#FECB52",
}

# ── Cached service instances ─────────────────────────────────────────

@st.cache_resource
def _get_extractor() -> AbsaExtractor:
    return AbsaExtractor()


@st.cache_resource
def _get_cache() -> LocalSemanticCache:
    return LocalSemanticCache()


@st.cache_resource
def _get_duckdb_path() -> str:
    return str(PROJECT_ROOT / "absa_analytics.db")


# ── Database helpers — read path ─────────────────────────────────────

def _query_db(sql: str, params: Optional[list[Any]] = None) -> list[Any]:
    """Run a read-only query and return all rows, closing the connection."""
    db_path = _get_duckdb_path()
    conn = duckdb.connect(db_path, read_only=True)
    try:
        return conn.execute(sql, params or []).fetchall()
    except duckdb.CatalogException:
        return []
    finally:
        conn.close()


def _fetchone_db(sql: str, params: Optional[list[Any]] = None) -> Any:
    """Run a read-only query and return one row or None."""
    db_path = _get_duckdb_path()
    conn = duckdb.connect(db_path, read_only=True)
    try:
        return conn.execute(sql, params or []).fetchone()
    except duckdb.CatalogException:
        return None
    finally:
        conn.close()


# ── Database helper — write path ─────────────────────────────────────

def _save_to_database(
    review: ExtractedReview,
    sector: TenantSector,
) -> None:
    """Persist an extracted review using a short-lived connection."""
    db_path = _get_duckdb_path()
    conn = duckdb.connect(db_path)
    try:
        now = datetime.now(timezone.utc)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                review_id VARCHAR PRIMARY KEY, text VARCHAR,
                sector VARCHAR, timestamp TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aspect_metrics (
                review_id VARCHAR, aspect VARCHAR,
                sentiment VARCHAR, confidence DOUBLE,
                FOREIGN KEY (review_id) REFERENCES reviews(review_id)
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO reviews (review_id, text, sector, timestamp)"
            " VALUES (?, ?, ?, ?)",
            [review.review_id, review.text, sector.value, now],
        )
        for asp in review.aspects:
            conn.execute(
                "INSERT INTO aspect_metrics (review_id, aspect, sentiment, confidence)"
                " VALUES (?, ?, ?, ?)",
                [review.review_id, asp.aspect, asp.sentiment, asp.confidence],
            )
    finally:
        conn.close()


# ── Out-of-domain keyword check ──────────────────────────────────────

_HARDWARE_KEYWORDS: set[str] = {
    "battery", "screen", "camera", "speaker", "display",
    "headphone", "charging", "processor", "ram", "storage",
    "keyboard", "touchpad", "microphone", "fingerprint",
}


def _out_of_domain_warning(text: str, sector: str) -> bool:
    """Return True if sector=fintech but text mentions hardware keywords."""
    if sector != "fintech":
        return False
    words = set(text.lower().split())
    return bool(words & _HARDWARE_KEYWORDS)


# ── Layout: two columns ──────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.5], gap="large")

# ═════════════════════════════════════════════════════════════════════
#  LEFT COLUMN — Review Submission
# ═════════════════════════════════════════════════════════════════════

with col_left:
    st.markdown("## Submit a Review")
    st.markdown("Select the sector, paste a customer review, and analyse it.")

    with st.container(border=True):
        sector_option = st.selectbox(
            "Sector / Tenant Type",
            options=["fintech", "ecommerce"],
            format_func=lambda x: x.upper(),
            help="Choose the business domain for aspect relevance.",
        )

        if "review_id" not in st.session_state:
            st.session_state.review_id = str(uuid.uuid4())[:8]
        st.text_input(
            "Review ID (auto-generated)",
            value=st.session_state.review_id,
            disabled=True,
        )

        review_text = st.text_area(
            "Review Text",
            placeholder="Paste or type the customer review here...",
            height=120,
        )

        submitted = st.button(
            "Analyse Review",
            type="primary",
            use_container_width=True,
        )

    # ── Domain boundary warning ───────────────────────────────────
    if review_text and _out_of_domain_warning(review_text, sector_option):
        st.warning(
            "Out-of-Domain Keywords Detected for Fintech Taxonomy. "
            "Categorising under nearest fallback boundaries."
        )

    # ── Processing block ──────────────────────────────────────────
    if submitted:
        if not review_text.strip():
            st.warning("Please enter some review text before submitting.")
        else:
            extractor = _get_extractor()
            cache = _get_cache()
            tenant_sector = TenantSector(sector_option)

            with st.status(
                "Analysing... please wait for the LLM to respond."
            ) as status:
                try:
                    cached = cache.get(review_text, st.session_state.review_id)
                    if cached is not None:
                        extracted = ExtractedReview(**cached)
                        process_time = 0.0  # Cache hit, no LLM inference
                    else:
                        start_time = time.time()
                        extracted = extractor.extract_aspects(
                            text=review_text,
                            tenant_id=st.session_state.review_id,
                            tenant_sector=tenant_sector,
                        )
                        end_time = time.time()
                        process_time = round(end_time - start_time, 2)
                        cache.set(
                            review_text,
                            st.session_state.review_id,
                            extracted.model_dump(),
                        )

                    _save_to_database(extracted, tenant_sector)

                    st.session_state.last_extracted = extracted
                    st.session_state.last_sector = sector_option

                    status.update(label="Analysis complete!", state="complete")
                    st.toast(f"Analysis complete in {process_time}s! ⚡", icon="✅")
                    st.session_state.review_id = str(uuid.uuid4())[:8]

                except Exception as exc:
                    status.update(label="Analysis failed.", state="error")
                    st.error(f"Extraction error: {exc}")

    # ── Live Review Breakdown Cards ───────────────────────────────
    if "last_extracted" in st.session_state and st.session_state.last_extracted:
        st.markdown("---")
        st.markdown("### Current Review Breakdown")
        sector_label: str = st.session_state.last_sector.upper()
        st.caption(f"Sector: {sector_label}")

        for asp in st.session_state.last_extracted.aspects:
            sentiment_upper = asp.sentiment.upper()
            color = SENTIMENT_COLORS.get(asp.sentiment, "#ccc")
            emoji_map: dict[str, str] = {
                "positive": "🟢", "negative": "🔴", "neutral": "🟡",
            }
            emoji = emoji_map.get(asp.sentiment, "⚪")

            with st.container(border=True):
                st.markdown(
                    f"**Aspect:** `{asp.aspect}` "
                    f"| Status: {emoji} **{sentiment_upper}** "
                    f"| Confidence: `{asp.confidence:.0%}`"
                )
                st.markdown(f"> {st.session_state.last_extracted.text}")

    # ── Reset Analytics History ───────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Reset Analytics History", type="secondary", use_container_width=True):
        import duckdb
        try:
            conn = duckdb.connect(str(PROJECT_ROOT / "absa_analytics.db"))
            conn.execute("DELETE FROM aspect_metrics;")
            conn.execute("DELETE FROM reviews;")
            conn.close()
            st.cache_resource.clear()
            st.toast("Analytics history cleared!", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to wipe: {e}")

# ═════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN — Analytics Dashboard
# ═════════════════════════════════════════════════════════════════════

with col_right:
    st.markdown("## Sector Analytics")

    chart_sector = st.selectbox(
        "Filter by sector",
        options=["fintech", "ecommerce"],
        format_func=lambda x: x.upper(),
        key="chart_sector",
    )

    # ── Business Insight Metric Cards ─────────────────────────────
    csi_row = _fetchone_db(
        """
        SELECT
            COUNT(*) FILTER (WHERE a.sentiment = 'positive') AS pos,
            COUNT(*) FILTER (WHERE a.sentiment = 'negative') AS neg,
            COUNT(*) AS total
        FROM reviews r
        JOIN aspect_metrics a ON r.review_id = a.review_id
        WHERE r.sector = ?
        """,
        [chart_sector],
    )

    if csi_row and csi_row[2] and csi_row[2] > 0:
        pos_count = csi_row[0] or 0
        neg_count = csi_row[1] or 0
        csi = round(((pos_count - neg_count) / csi_row[2]) * 100, 1)
    else:
        pos_count = 0
        neg_count = 0
        csi = None

    worst_row = _fetchone_db(
        """
        SELECT a.aspect, COUNT(*) AS cnt
        FROM reviews r
        JOIN aspect_metrics a ON r.review_id = a.review_id
        WHERE r.sector = ? AND a.sentiment = 'negative'
        GROUP BY a.aspect
        ORDER BY cnt DESC
        LIMIT 1
        """,
        [chart_sector],
    )

    tickets_row = _fetchone_db(
        "SELECT COUNT(DISTINCT review_id) FROM reviews WHERE sector = ?",
        [chart_sector],
    )
    total_tickets: int = tickets_row[0] if tickets_row else 0

    # ── Render metric cards (HTML flexbox layout) ──────────────────

    # Determine CSI display values.
    if csi is not None:
        if csi > 0:
            csi_color = "#00CC96"
            csi_indicator = "▲"
        elif csi < 0:
            csi_color = "#EF553B"
            csi_indicator = "▼"
        else:
            csi_color = "#94A3B8"
            csi_indicator = "◆"
        csi_value_html = (
            f'<span style="color:{csi_color}; font-size:1.8rem; font-weight:700;">'
            f"{csi_indicator} {csi:+.1f}%"
            f"</span>"
        )
    else:
        csi_value_html = (
            '<span style="color:#94A3B8; font-size:1.8rem;">—</span>'
        )

    # Determine Critical Fix display.
    if worst_row:
        critical_html = (
            f'<div style="white-space:normal; word-wrap:break-word; '
            f'overflow-wrap:break-word; font-size:1.1rem; font-weight:bold; '
            f'color:#EF553B;">'
            f"⚠️ {worst_row[0]}"
            f"</div>"
            f'<div style="color:#94A3B8; font-size:0.85rem; margin-top:4px;">'
            f"{worst_row[1]} negative mention(s)"
            f"</div>"
        )
    else:
        critical_html = (
            '<div style="white-space:normal; word-wrap:break-word; '
            'overflow-wrap:break-word; font-size:1.1rem; font-weight:bold; '
            'color:#00CC96;">'
            "✅ No critical issues"
            "</div>"
        )

    metrics_html = f"""
    <div style="display:flex; gap:1rem; width:100%; justify-content:space-between; margin-bottom: 2rem;">
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 16px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); padding:1rem; flex:1; min-width:0;">
            <div style="color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">Customer Sentiment Index</div>
            <div style="margin-top:6px;">{csi_value_html}</div>
        </div>
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 16px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); padding:1rem; flex:1; min-width:0;">
            <div style="color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">Critical Fix Alert</div>
            <div style="margin-top:6px;">{critical_html}</div>
        </div>
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 16px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); padding:1rem; flex:1; min-width:0;">
            <div style="color:#94A3B8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">Total Tickets Audited</div>
            <div style="margin-top:6px;">
                <span style="color:#F8FAFC; font-size:1.8rem; font-weight:700;">
                    {total_tickets:,}
                </span>
            </div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    # ── Horizontal Stacked Sentiment Bar Chart ─────────────────────

    chart_data = _query_db(
        """
        SELECT a.aspect, a.sentiment, ROUND(AVG(a.confidence), 4) AS avg_conf
        FROM reviews r
        JOIN aspect_metrics a ON r.review_id = a.review_id
        WHERE r.sector = ?
        GROUP BY a.aspect, a.sentiment
        ORDER BY a.aspect
        """,
        [chart_sector],
    )

    if chart_data:
        aspects_list: list[str] = []
        pivot: dict[str, dict[str, float]] = {}
        for aspect_name, sentiment, avg_conf in chart_data:
            if aspect_name not in pivot:
                pivot[aspect_name] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                aspects_list.append(aspect_name)
            pivot[aspect_name][sentiment] = float(avg_conf) if avg_conf else 0.0

        positive_vals = [pivot[a]["positive"] for a in aspects_list]
        negative_vals = [pivot[a]["negative"] for a in aspects_list]
        neutral_vals = [pivot[a]["neutral"] for a in aspects_list]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=aspects_list,
            x=positive_vals,
            name="Positive",
            marker_color="#00CC96",
            orientation="h",
        ))
        fig.add_trace(go.Bar(
            y=aspects_list,
            x=negative_vals,
            name="Negative",
            marker_color="#EF553B",
            orientation="h",
        ))
        fig.add_trace(go.Bar(
            y=aspects_list,
            x=neutral_vals,
            name="Neutral",
            marker_color="#FECB52",
            orientation="h",
        ))

        fig.update_layout(
            barmode="stack",
            title=f"Sentiment Distribution — {chart_sector.upper()}",
            xaxis_title="AI Confidence Score",
            yaxis_title="Aspect Category",
            height=400,
            legend_title="Sentiment",
            margin=dict(l=180, r=20, t=30, b=40),
            yaxis={"categoryorder": "total ascending"},
            hovermode="y unified",
        )
        fig.update_xaxes(range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)
    else:
        empty_msg = f"""
<div style="background-color: #1A2333; border: 1px dashed #334155; border-radius: 8px; padding: 2rem; text-align: center; color: #94A3B8; margin-top: 1rem;">
    <span style="font-size: 1.5rem;">📊</span><br><br>
    No historical data for the <b>{chart_sector}</b> sector.<br>Submit a review from the left panel to seed the analytics.
</div>
"""
        st.markdown(empty_msg, unsafe_allow_html=True)
