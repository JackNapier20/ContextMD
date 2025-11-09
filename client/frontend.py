"""
Pre-Op Research Lookup Tool
Medical-grade professional interface
"""

import streamlit as st
from datetime import datetime
import requests
import json
import os

# --- Clear any persisted state on reload ---
for _k in ("case_path", "generation_output", "search_results"):
    if _k in st.session_state:
        st.session_state.pop(_k, None)

# --- Ensure text inputs are readable (black text on white bg) ---
st.markdown(
    """
    <style>
    textarea, input, .stTextArea textarea, .stTextInput input {
        color: #111 !important;
        background-color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Backend configuration (override with env var BACKEND_URL if needed)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

def backend_alive() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Pre-Op Research Assistant",
    page_icon="📋",
    layout="wide"
)

# ============================================================
# CUSTOM CSS - Medical Color Palette
# ============================================================
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Medical Color Palette:
       - Navy Blue: #1e3a8a (trust, professionalism)
       - Teal: #0d9488 (medical scrubs, healthcare)
       - White: #ffffff (cleanliness, sterility)
       - Light Blue: #e0f2fe (calm, clinical)
       - Dark Gray: #1f2937 (text, serious)
    */
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main content background */
    .main {
        background-color: #f8fafc;
    }
    
    /* Sidebar styling - Medical Blue/Teal */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #0d9488 100%);
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Sidebar text - white for contrast */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Sidebar input fields */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 4px;
    }
    
    section[data-testid="stSidebar"] input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Sidebar labels */
    section[data-testid="stSidebar"] label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500;
    }
    
    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
    }
    
    /* Sidebar metrics */
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    /* Headers */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #0d9488;
        padding-bottom: 0.5rem;
        letter-spacing: 0.05em;
    }
    
    /* Patient card - Medical Teal */
    .patient-card {
        background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
        color: white;
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.2);
    }
    
    .patient-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 0.75rem;
    }
    
    .patient-card p {
        margin: 0.5rem 0;
        font-size: 0.95rem;
        opacity: 0.95;
    }
    
    .patient-card strong {
        opacity: 1;
        font-weight: 600;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e0f2fe;
        border-left: 4px solid #0891b2;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border-radius: 4px;
    }
    
    .info-box strong {
        color: #0c4a6e;
        font-weight: 600;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        font-size: 1rem;
        border-radius: 6px;
        border: 2px solid #cbd5e1;
        padding: 0.75rem;
        background-color: #ffffff;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #0d9488;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.1);
        outline: none;
    }
    
    /* Button styling - Medical Blue */
    .stButton > button {
        font-size: 1rem;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s;
        background-color: #1e3a8a;
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #1e40af;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
    }
    
    /* Primary button (search) */
    .stButton > button[kind="primary"] {
        background-color: #0d9488;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #0f766e;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
        color: #1e3a8a;
        background-color: #f1f5f9;
        border-radius: 6px;
        border-left: 4px solid #0d9488;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e0f2fe;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a8a;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 500;
    }
    
    /* Tag styling */
    .tag {
        display: inline-block;
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .tag-green {
        background-color: #ccfbf1;
        color: #0f766e;
    }
    
    .tag-yellow {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    .tag-red {
        background-color: #fee2e2;
        color: #991b1b;
    }
    
    /* Info/warning boxes */
    .stInfo {
        background-color: #e0f2fe;
        border-left: 4px solid #0891b2;
    }
    
    .stWarning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
    }
    
    .stSuccess {
        background-color: #ccfbf1;
        border-left: 4px solid #0d9488;
    }
    
    /* Remove streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dividers */
    hr {
        border-color: #cbd5e1;
        margin: 2rem 0;
    }
    
    /* Caption text */
    .caption {
        color: #64748b;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR - System Status
# ============================================================
with st.sidebar:
    st.markdown("### SYSTEM STATUS")
    st.markdown("---")
    ok = backend_alive()
    st.metric("Backend", "Online" if ok else "Offline")

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown('<h1 class="main-title">Pre-Operative Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Evidence-based protocol lookup for surgical planning</p>', unsafe_allow_html=True)

st.markdown("---")


# ============================================================
# CASE INPUTS
# ============================================================
st.markdown('<div class="section-header">CASE INPUTS</div>', unsafe_allow_html=True)
with st.form("case_form", clear_on_submit=False):
    query = st.text_input(
        "Clinical question / case focus",
        placeholder="e.g., lung cancer with EGFR mutation; confirm testing & first-line options",
    )
    colA, colB = st.columns([1,1])
    with colA:
        age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
    with colB:
        sex = st.selectbox("Sex", options=["", "female", "male", "other"], index=0)
    complaint = st.text_area("Chief complaint", placeholder="e.g., persistent cough")
    history = st.text_area("History", placeholder="e.g., former smoker")
    meds = st.text_area("Medications", placeholder="e.g., metformin")
    labs = st.text_area("Labs", placeholder="e.g., normal CBC")
    vitals = st.text_area("Vitals", placeholder="e.g., stable")
    search_button = st.form_submit_button("Search & Generate", type="primary")

# ============================================================
# BACKEND INTEGRATION - SEARCH & GENERATE
# ============================================================
if search_button:
    if not query:
        st.warning("Please enter a research question")
    else:
        # Show loading state
        with st.spinner("Searching and generating recommendations..."):
            try:
                # One-shot: create case and generate in the backend
                patient_payload = {
                    "age": str(age) if age else None,
                    "sex": sex or None,
                    "complaint": complaint or None,
                    "history": history or None,
                    "meds": meds or None,
                    "labs": labs or None,
                    "vitals": vitals or None,
                }

                run_payload = {
                    "patient": patient_payload,
                    "query": query,
                    "topk": 20,
                }

                run_resp = requests.post(
                    f"{BACKEND_URL}/api/run",
                    json=run_payload,
                    timeout=60,
                )
                run_resp.raise_for_status()
                run_data = run_resp.json()

                st.session_state.case_path = run_data.get("case_path")
                st.session_state.generation_output = run_data.get("preview")
                st.session_state.search_results = {"case_path": st.session_state.case_path}
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running and CORS allows this origin.")
            except requests.exceptions.Timeout:
                st.error("❌ Backend request timed out. Please try again.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Backend error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================
# RESULTS DISPLAY
# ============================================================
gen_out = st.session_state.get("generation_output")
case_path = st.session_state.get("case_path")

if gen_out:
    st.markdown('<div class="section-header">SEARCH RESULTS & RECOMMENDATIONS</div>', unsafe_allow_html=True)
    st.markdown(gen_out)
    st.markdown("---")
    if case_path:
        st.caption(f"**Case:** {case_path}")
    st.caption(f"**Generated:** {datetime.now().strftime('%H:%M:%S')}")
else:
    st.info("Enter a research question and click SEARCH to get started")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("Pre-Op Research Assistant v1.0")

with footer_col2:
    st.caption("Powered by Claude AI + FAISS")

with footer_col3:
    st.caption(f"© {datetime.now().year}")