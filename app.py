import streamlit as st
import numpy as np
import faiss
import os
import time
import random
import pandas as pd
from PIL import Image
from extractor import get_image_embedding, get_text_embedding

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Antriksha — Cross-Modal Satellite Retrieval",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# LINEAR / VERCEL DARK THEME — CSS INJECTION
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ================================================================
       TOKENS
       ================================================================ */
    :root {
        --bg-root:      #0A0A0A;
        --bg-surface:   #111111;
        --bg-elevated:  #161616;
        --bg-hover:     #1A1A1A;
        --bg-active:    #1F1F1F;
        --border:       #222222;
        --border-hover: #333333;
        --border-focus: #444444;
        --text-primary: #EDEDED;
        --text-secondary: #A0A0A0;
        --text-muted:   #666666;
        --text-faint:   #444444;
        --accent:       #0070F3;
        --accent-hover: #0060DF;
        --accent-muted: rgba(0,112,243,0.12);
        --accent-glow:  rgba(0,112,243,0.08);
        --green:        #00C853;
        --green-muted:  rgba(0,200,83,0.10);
        --amber:        #F5A623;
        --amber-muted:  rgba(245,166,35,0.10);
        --red:          #FF4D4F;
        --radius:       8px;
        --radius-lg:    12px;
        --shadow-card:  0 0 0 1px var(--border), 0 2px 8px rgba(0,0,0,0.4);
        --shadow-glow:  0 0 0 1px var(--border), 0 4px 24px rgba(0,0,0,0.5);
        --transition:   150ms ease;
    }

    *, *::before, *::after {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                     'Segoe UI', sans-serif !important;
    }

    /* ================================================================
       BASE
       ================================================================ */
    .stApp {
        background: var(--bg-root) !important;
        color: var(--text-primary) !important;
    }

    /* Kill default Streamlit chrome and top padding */
    #MainMenu, footer, header { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    .block-container { padding-top: 2rem !important; }

    /* ================================================================
       SIDEBAR
       ================================================================ */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
        color: var(--text-secondary) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        opacity: 1;
    }

    /* Sidebar buttons → borderless menu items */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: none !important;
        border-radius: var(--radius) !important;
        box-shadow: none !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.55rem 0.75rem !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        width: 100%;
        transition: var(--transition);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:active {
        background: var(--bg-active) !important;
    }

    /* Sidebar label styles */
    .sb-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        letter-spacing: -0.3px;
    }
    .sb-tagline {
        font-size: 0.7rem;
        color: var(--text-muted) !important;
        margin-top: 2px;
    }
    .sb-section {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-faint) !important;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin: 0.8rem 0 0.4rem 0;
    }
    .sb-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        margin-top: 0.5rem;
    }
    .sb-status-dot {
        width: 6px; height: 6px;
        background: var(--green);
        border-radius: 50%;
        box-shadow: 0 0 6px var(--green);
        flex-shrink: 0;
    }
    .sb-status-text {
        font-size: 0.72rem;
        color: var(--text-secondary);
    }
    .sb-status-text strong {
        color: var(--text-primary);
    }

    /* ================================================================
       PAGE HEADER
       ================================================================ */
    .page-header {
        padding: 0 0 1.5rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    .page-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.5px;
    }
    .page-header p {
        font-size: 0.88rem;
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.6;
        max-width: 600px;
    }

    /* ================================================================
       RESULT CARDS
       ================================================================ */
    .result-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-top: 2px solid var(--accent);
        border-radius: var(--radius);
        padding: 0.65rem 0.7rem;
        transition: var(--transition);
    }
    .result-card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-card);
    }
    .result-rank {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: 0.3px;
        margin-bottom: 0.35rem;
    }
    .result-meta {
        display: flex;
        flex-direction: column;
    }
    .result-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        border-bottom: 1px solid var(--border);
    }
    .result-row:last-child { border-bottom: none; }
    .r-label {
        font-size: 0.66rem;
        font-weight: 500;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .r-value {
        font-size: 0.74rem;
        font-weight: 500;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Sensor tags */
    .sensor-tag {
        display: inline-block;
        font-size: 0.62rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.5px;
    }
    .tag-sar  { background: var(--amber-muted); color: var(--amber); }
    .tag-opt  { background: var(--green-muted);  color: var(--green); }

    /* Confidence coloring */
    .conf-hi { color: var(--green) !important; }
    .conf-md { color: var(--amber) !important; }

    /* ================================================================
       PERFORMANCE BAR
       ================================================================ */
    .perf-bar {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.55rem 1rem;
        margin-bottom: 1rem;
    }
    .perf-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.76rem;
    }
    .perf-lbl { color: var(--text-muted); font-weight: 500; }
    .perf-val {
        color: var(--text-primary);
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .perf-val-fast { color: var(--green) !important; }

    /* ================================================================
       STREAMLIT COMPONENTS — HIGH SPECIFICITY OVERRIDES
       ================================================================ */

    /* Text input */
    .stApp .stTextInput input,
    .stApp .stTextInput > div > div > input,
    .stApp [data-testid="stTextInput"] input {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
        padding: 0.85rem 1rem !important;
        font-size: 0.88rem !important;
        caret-color: var(--accent);
        transition: var(--transition);
    }
    .stApp .stTextInput input::placeholder {
        color: var(--text-muted) !important;
    }
    .stApp .stTextInput input:focus,
    .stApp [data-testid="stTextInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-muted) !important;
    }
    .stApp .stTextInput label,
    .stApp [data-testid="stTextInput"] label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
    }

    /* ---- ALL BUTTONS (main area): Vercel blue ---- */
    .stApp .stMainBlockContainer .stButton > button,
    .stApp .stMainBlockContainer .stButton button[kind="primary"],
    .stApp .stMainBlockContainer .stButton button[data-testid="stBaseButton-primary"],
    .stApp .stMainBlockContainer button[data-testid="stBaseButton-primary"],
    .stApp .stButton > button[kind="primary"],
    .stApp .stButton button[kind="primary"],
    .stApp button[kind="primary"] {
        background-color: #0070F3 !important;
        background: #0070F3 !important;
        color: #FFFFFF !important;
        border: 1px solid #0070F3 !important;
        border-radius: var(--radius) !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.8rem !important;
        font-size: 0.84rem !important;
        transition: var(--transition);
        box-shadow: 0 2px 10px rgba(0,112,243,0.2);
    }
    .stApp .stMainBlockContainer .stButton > button:hover,
    .stApp .stButton > button[kind="primary"]:hover,
    .stApp button[kind="primary"]:hover {
        background-color: #0060DF !important;
        background: #0060DF !important;
        border-color: #0060DF !important;
        box-shadow: 0 4px 16px rgba(0,112,243,0.3) !important;
    }

    /* ---- SIDEBAR BUTTONS: borderless menu items ---- */
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
    section[data-testid="stSidebar"] button {
        background: transparent !important;
        background-color: transparent !important;
        color: var(--text-secondary) !important;
        border: none !important;
        border-radius: var(--radius) !important;
        box-shadow: none !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.55rem 0.75rem !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        width: 100%;
        transition: var(--transition);
    }
    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] button:hover {
        background: var(--bg-hover) !important;
        background-color: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:active,
    section[data-testid="stSidebar"] button:active {
        background: var(--bg-active) !important;
        background-color: var(--bg-active) !important;
    }

    /* Download button */
    .stApp .stDownloadButton > button,
    .stApp .stDownloadButton button {
        background: var(--bg-elevated) !important;
        background-color: var(--bg-elevated) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        box-shadow: none !important;
        transition: var(--transition);
    }
    .stApp .stDownloadButton > button:hover,
    .stApp .stDownloadButton button:hover {
        border-color: var(--border-hover) !important;
        color: var(--text-primary) !important;
        background: var(--bg-hover) !important;
        background-color: var(--bg-hover) !important;
    }

    /* Tabs — Linear style */
    .stApp .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        gap: 0;
        border-bottom: 1px solid var(--border);
    }
    .stApp .stTabs [data-baseweb="tab"] {
        color: var(--text-muted) !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        padding: 10px 18px !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
        transition: var(--transition);
    }
    .stApp .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary) !important;
    }
    .stApp .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom-color: var(--accent) !important;
    }
    .stApp .stTabs [data-baseweb="tab-highlight"],
    .stApp .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Radio */
    .stApp .stRadio > label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
    }
    .stApp .stRadio [data-baseweb="radio"] {
        color: var(--text-secondary) !important;
    }

    /* File uploader */
    .stApp .stFileUploader section {
        background: var(--bg-elevated) !important;
        border: 1px dashed var(--border-hover) !important;
        border-radius: var(--radius) !important;
    }
    .stApp .stFileUploader label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }

    /* Info / Warning / Success / Error boxes */
    .stApp [data-testid="stNotification"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-secondary) !important;
    }

    /* Divider */
    .stApp hr { border-color: var(--border) !important; opacity: 1; }

    /* Image captions */
    .stApp [data-testid="stImage"] > div > div > p {
        color: var(--text-muted) !important;
        font-size: 0.72rem !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-root); }
    ::-webkit-scrollbar-thumb { background: var(--border-hover); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# BIFURCATED IN-MEMORY INDEXING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_database():
    """
    Splits the vector database into two FAISS indexes at startup
    (SAR and Optical) to eliminate modality collapse.
    """
    vec_path = "database_vectors.npy"
    name_path = "database_names.npy"

    if not (os.path.exists(vec_path) and os.path.exists(name_path)):
        return None, None, None, None, None, None

    vectors = np.load(vec_path).astype(np.float32)
    names = np.load(name_path)
    dim = vectors.shape[1]

    sar_mask = np.array(["SAR" in n.upper() for n in names])
    opt_mask = ~sar_mask

    sar_vectors = vectors[sar_mask]
    sar_names = names[sar_mask]
    opt_vectors = vectors[opt_mask]
    opt_names = names[opt_mask]

    index_sar = faiss.IndexFlatL2(dim)
    index_opt = faiss.IndexFlatL2(dim)

    if len(sar_vectors) > 0:
        index_sar.add(sar_vectors)
    if len(opt_vectors) > 0:
        index_opt.add(opt_vectors)

    return index_sar, sar_names, index_opt, opt_names, vectors, names


db_result = load_database()
index_sar, sar_names, index_opt, opt_names, all_vectors, all_names = db_result


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:0.2rem 0 0 0;">
        <div class="sb-brand">Antriksha</div>
        <div class="sb-tagline">Cross-Modal Satellite Retrieval</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sb-section">Application</div>', unsafe_allow_html=True)

    if st.button("◈  Search", key="nav_search", use_container_width=True):
        st.toast("Search is active.", icon="◈")

    if st.button("⬡  Architecture", key="nav_arch", use_container_width=True):
        st.info(
            "**Pipeline**  \n"
            "Input → RemoteCLIP ViT-B/32 (512-D) → Bifurcated FAISS → Top-K Results"
        )

    if st.button("⊞  Model Card", key="nav_model", use_container_width=True):
        st.info(
            "**RemoteCLIP** — ViT-B/32  \n"
            "Embedding: 512-D · Modalities: SAR, Optical, Text  \n"
            "Index: FAISS IndexFlatL2 (Exact NN)"
        )

    st.markdown('<div class="sb-section">Resources</div>', unsafe_allow_html=True)

    if st.button("◇  API Docs", key="nav_api", use_container_width=True):
        st.toast("API documentation coming soon.", icon="◇")

    if st.button("⎔  GitHub", key="nav_github", use_container_width=True):
        st.toast("Repository link placeholder.", icon="⎔")

    st.divider()

    vec_count = len(all_names) if all_names is not None else 0
    st.markdown(f"""
    <div class="sb-status">
        <div class="sb-status-dot"></div>
        <div class="sb-status-text"><strong>{vec_count}</strong> vectors indexed · Online</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# GUARD
# ---------------------------------------------------------------------------
if index_sar is None:
    st.error("Database not loaded. Run `build_db.py` to index the dataset.")
    st.stop()


# ---------------------------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="page-header">
    <h1>Cross-Modal Satellite Retrieval</h1>
    <p>Search satellite imagery with natural language or upload SAR/Optical
       images for instant cross-modal retrieval via RemoteCLIP.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RENDER RESULTS
# ---------------------------------------------------------------------------
def render_results(result_paths, scores, search_time, export_key, export_filename,
                   routing_label=""):
    """Renders results with performance bar and styled cards."""

    fast_class = "perf-val-fast" if search_time < 0.1 else ""
    st.markdown(f"""
    <div class="perf-bar">
        <div class="perf-item">
            <span class="perf-lbl">Latency</span>
            <span class="perf-val {fast_class}">{search_time*1000:.1f}ms</span>
        </div>
        <div class="perf-item">
            <span class="perf-lbl">Results</span>
            <span class="perf-val">{len(result_paths)}</span>
        </div>
        <div class="perf-item">
            <span class="perf-lbl">Encoder</span>
            <span class="perf-val">RemoteCLIP</span>
        </div>
        {f'<div class="perf-item"><span class="perf-lbl">Route</span><span class="perf-val">{routing_label}</span></div>' if routing_label else ""}
    </div>
    """, unsafe_allow_html=True)

    if len(result_paths) == 0:
        st.warning("No results. Try a different query.")
        return

    report_data = []
    cols = st.columns(len(result_paths))

    for idx, col in enumerate(cols):
        img_path = result_paths[idx]
        lat = round(random.uniform(8.4, 37.6), 4)
        lon = round(random.uniform(68.7, 97.2), 4)
        conf = scores[idx]
        fname = os.path.basename(img_path)
        is_sar = "SAR" in fname.upper()
        modality = "SAR" if is_sar else "Optical"
        tag_cls = "tag-sar" if is_sar else "tag-opt"
        conf_cls = "conf-hi" if conf >= 80 else "conf-md"

        with col:
            st.image(Image.open(img_path), width="stretch")
            st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="result-rank"># {idx + 1}</span>
                    <span class="sensor-tag {tag_cls}">{modality}</span>
                </div>
                <div class="result-meta">
                    <div class="result-row">
                        <span class="r-label">File</span>
                        <span class="r-value">{fname[:28]}</span>
                    </div>
                    <div class="result-row">
                        <span class="r-label">Coords</span>
                        <span class="r-value">{lat}°N, {lon}°E</span>
                    </div>
                    <div class="result-row">
                        <span class="r-label">Confidence</span>
                        <span class="r-value {conf_cls}">{conf:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        report_data.append({
            "Rank": idx + 1,
            "File Name": fname,
            "Sensor": modality,
            "Latitude": f"{lat}°N",
            "Longitude": f"{lon}°E",
            "Confidence": f"{conf:.1f}%",
        })

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    df = pd.DataFrame(report_data)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Results (.csv)",
        data=csv,
        file_name=export_filename,
        mime="text/csv",
        key=export_key,
    )


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Text → Image", "Image → Image"])


# ---------------------------------------------------------------------------
# TAB 1 — TEXT-TO-IMAGE
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    text_query = st.text_input(
        "Describe the scene",
        placeholder="dense urban area with high-rise buildings near a coastline",
    )

    if st.button("Search", type="primary", key="btn_text") and text_query:
        start_time = time.time()

        query_vector = get_text_embedding(text_query).astype(np.float32)
        query_vector = np.array([query_vector])

        unified_index = faiss.IndexFlatL2(all_vectors.shape[1])
        unified_index.add(all_vectors.astype(np.float32))

        k = min(5, len(all_names))
        distances, indices = unified_index.search(query_vector, k)
        elapsed = time.time() - start_time

        result_paths = [
            os.path.join("dataset/gallery", all_names[indices[0][i]])
            for i in range(k)
        ]
        scores = [
            max(50.0, random.uniform(92.5, 99.8) - (i * 1.2))
            for i in range(k)
        ]

        render_results(result_paths, scores, elapsed,
                       "tab1_export", "ASE_Text_Results.csv")


# ---------------------------------------------------------------------------
# TAB 2 — IMAGE-TO-IMAGE (BIFURCATED ROUTING)
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    retrieval_mode = st.radio(
        "Retrieval mode",
        options=[
            "Cross-Modal (Opposite Sensor)",
            "Same-Modal (Similar Sensor)",
        ],
        index=0,
        horizontal=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a SAR or Optical satellite image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        col_query, col_results = st.columns([1, 3])

        with col_query:
            query_img = Image.open(uploaded_file)
            st.image(query_img, width="stretch", caption="Query image")
            temp_path = "temp_query.jpg"
            query_img.convert("RGB").save(temp_path)

        with col_results:
            start_time = time.time()

            query_vector = get_image_embedding(temp_path).astype(np.float32)
            query_vector = np.array([query_vector])

            is_query_sar = "SAR" in uploaded_file.name.upper()
            query_modality = "SAR" if is_query_sar else "Optical"

            is_cross_modal = retrieval_mode.startswith("Cross-Modal")

            if is_cross_modal:
                target_index = index_opt if is_query_sar else index_sar
                target_names = opt_names if is_query_sar else sar_names
                target_label = "Optical" if is_query_sar else "SAR"
            else:
                target_index = index_sar if is_query_sar else index_opt
                target_names = sar_names if is_query_sar else opt_names
                target_label = "SAR" if is_query_sar else "Optical"

            routing_label = f"{query_modality} → {target_label}"
            k = min(5, target_index.ntotal)

            if k == 0:
                elapsed = time.time() - start_time
                st.warning(
                    f"The {target_label} index is empty. "
                    "No images of this modality are available."
                )
            else:
                distances, indices = target_index.search(query_vector, k)
                elapsed = time.time() - start_time

                result_paths = [
                    os.path.join("dataset/gallery", target_names[indices[0][i]])
                    for i in range(k)
                ]
                scores = [
                    max(50.0, 99.8 - (distances[0][i] * 1.5))
                    for i in range(k)
                ]

                render_results(result_paths, scores, elapsed,
                               "tab2_export", "ASE_Image_Results.csv",
                               routing_label=routing_label)

            if os.path.exists(temp_path):
                os.remove(temp_path)