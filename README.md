# 📊 Multi-Tenant ABSA Engine (Aspect-Based Sentiment Analysis)

An enterprise-grade, locally hosted Aspect-Based Sentiment Analysis (ABSA) platform. This engine utilizes local Small Language Models (SLMs) and Structured Output generation to parse messy customer reviews into structured, quantifiable business metrics in real-time.

## 🚀 The Tech Stack
* **AI Engine:** Local SLMs (Phi-3 / Llama-3) running via GPU acceleration.
* **Orchestration:** `Instructor` (Pydantic-enforced strict structured LLM outputs).
* **Data Layer:** `DuckDB` (In-process analytical columnar database for OLAP workloads).
* **Frontend UI:** `Streamlit` (Custom iOS-style Glassmorphism CSS injection).

## 🧠 System Architecture & Topology
1. **Client Request:** User inputs messy text via the UI.
2. **Tenant Context Injection:** The system dynamically routes the prompt based on the selected sector (Fintech vs. E-commerce) to apply strict domain boundaries.
3. **Structured LLM Inference:** The text is processed locally, extracting sentiment polarity (Positive/Negative/Neutral) and categorical aspects (e.g., `payment_gateway`, `product_quality`).
4. **Data Transformation:** The structured JSON payload is committed directly to a local DuckDB ledger.
5. **Analytical Views:** The UI instantly queries DuckDB to render live-updating, stacked horizontal Plotly charts.

*Constraints & Optimization:* The LLM client is cached in VRAM using `@st.cache_resource` to drop inference latency from >10s to <2s. The UI utilizes inline HTML/CSS flexbox for zero-clipping metric cards and glassmorphism styling.

## 🛠️ Quick Start
**1. Clone the repository:**
`git clone <your-repo-url>`
`cd absa_engine`

**2. Install dependencies:**
`pip install -r requirements.txt`

**3. Boot the Analytics Engine:**
`streamlit run dashboard.py`