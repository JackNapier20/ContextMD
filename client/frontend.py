# app.py
"""
Pre-Op Research Lookup Tool
Medical-grade professional interface
"""

import streamlit as st
from datetime import datetime

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
# SIDEBAR - Patient Info
# ============================================================
with st.sidebar:
    st.markdown("### PATIENT INFORMATION")
    st.markdown("---")
    
    st.text_input("Patient ID", value="12345", disabled=True)
    st.text_input("Name", value="Smith, Jane", disabled=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Age", value="65", disabled=True)
    with col2:
        st.text_input("Sex", value="F", disabled=True)
    
    st.text_input("Weight", value="70 kg", disabled=True)
    
    st.markdown("---")
    st.markdown("### PLANNED PROCEDURE")
    
    st.text_area(
        "Procedure",
        value="Laparoscopic Cholecystectomy",
        height=80,
        disabled=True
    )
    
    st.date_input("Surgery Date", datetime.now())
    st.text_input("Surgeon", value="Dr. Smith", disabled=True)
    
    st.markdown("---")
    st.markdown("### MEDICAL HISTORY")
    
    st.multiselect(
        "Comorbidities",
        ["Hypertension", "Type 2 Diabetes", "GERD", "Hypothyroidism"],
        default=["Hypertension", "Type 2 Diabetes"],
        disabled=True
    )
    
    st.text_input("Allergies", value="Penicillin", disabled=True)
    st.selectbox("ASA Class", ["I", "II", "III", "IV", "V"], index=1, disabled=True)
    
    st.markdown("---")
    st.markdown("### SYSTEM STATUS")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Protocols", "10")
    with col2:
        st.metric("Status", "Ready")

# ============================================================
# MAIN HEADER
# ============================================================
st.markdown('<h1 class="main-title">Pre-Operative Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Evidence-based protocol lookup for surgical planning</p>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# PATIENT OVERVIEW CARD
# ============================================================
st.markdown('<div class="section-header">PATIENT OVERVIEW</div>', unsafe_allow_html=True)

st.markdown("""
<div class="patient-card">
    <h3>Jane Smith, 65F</h3>
    <p><strong>Procedure:</strong> Laparoscopic Cholecystectomy</p>
    <p><strong>ASA Class:</strong> II</p>
    <p><strong>Comorbidities:</strong> Hypertension, Type 2 Diabetes</p>
    <p><strong>Allergies:</strong> Penicillin</p>
    <p><strong>Weight:</strong> 70 kg</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SEARCH SECTION
# ============================================================
st.markdown('<div class="section-header">RESEARCH QUERY</div>', unsafe_allow_html=True)

# Search input
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input(
        "Research Question",
        placeholder="What preoperative tests are required for this patient?",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button("SEARCH", type="primary", use_container_width=True)

# ============================================================
# EXAMPLE QUERIES
# ============================================================
with st.expander("COMMON QUESTIONS", expanded=True):
    examples = [
        "What preoperative tests are required for ASA II patients?",
        "What are the fasting guidelines for elective surgery?",
        "Should beta-blockers be continued perioperatively?",
        "What VTE prophylaxis is recommended?",
        "What are the antibiotic prophylaxis guidelines?",
        "How should diabetes medications be managed preoperatively?",
        "What preoperative optimization is needed?",
        "What consent and patient education is required?"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, example in enumerate(examples):
        with col1 if i % 2 == 0 else col2:
            st.button(example, key=f"ex_{i}", use_container_width=True)

st.markdown("---")

# ============================================================
# RESULTS PLACEHOLDER
# ============================================================
if search_button and query:
    st.markdown('<div class="section-header">SEARCH RESULTS</div>', unsafe_allow_html=True)
    
    # Mock result 1
    with st.expander("NICE NG45 | Routine Preoperative Tests (95% match)", expanded=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown("**Source:** NICE Guidelines")
            st.markdown("**Protocol ID:** NG45")
        
        with col2:
            st.metric("Relevance", "95%")
        
        with col3:
            st.markdown('<span class="tag tag-green">Highly Relevant</span>', unsafe_allow_html=True)
        
        st.markdown("**URL:** https://www.nice.org.uk/guidance/ng45")
        
        st.markdown("---")
        
        st.markdown("**RELEVANT CONTENT**")
        st.info("""
For ASA Class II patients undergoing intermediate surgery:

- Full Blood Count (FBC): Consider based on clinical assessment
- Renal Function: Recommended for patients with diabetes or hypertension  
- ECG: Recommended for patients >65 years or with cardiovascular risk factors
- Chest X-ray: Not routinely recommended unless specific clinical indication
- Blood glucose: Check in diabetic patients

[Placeholder - actual content will come from RAG system]
        """)
    
    # Mock result 2
    with st.expander("WHO Surgical Safety Guidelines (78% match)"):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown("**Source:** WHO")
            st.markdown("**Protocol ID:** Safe Surgery 2009")
        
        with col2:
            st.metric("Relevance", "78%")
        
        with col3:
            st.markdown('<span class="tag tag-yellow">Relevant</span>', unsafe_allow_html=True)
        
        st.markdown("**URL:** https://www.who.int/...")
        
        st.markdown("---")
        
        st.info("[Placeholder content from WHO guidelines]")
    
    # Mock result 3
    with st.expander("AHA Perioperative Guidelines (65% match)"):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown("**Source:** American Heart Association")
        
        with col2:
            st.metric("Relevance", "65%")
        
        with col3:
            st.markdown('<span class="tag">Somewhat Relevant</span>', unsafe_allow_html=True)
        
        st.info("[Placeholder content from AHA]")
    
    st.markdown("---")
    
    # Placeholder for synthesis
    st.markdown('<div class="section-header">SYNTHESIZED RECOMMENDATIONS</div>', unsafe_allow_html=True)
    
    st.info("""
**Key Recommendations for this Patient:**

[AI analysis will appear here once Claude integration is added]

Based on retrieved guidelines:
- Recommendation 1
- Recommendation 2  
- Recommendation 3
    """)
    
    st.markdown("---")
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("**Sources:** NICE, WHO, AHA")
    with col2:
        st.caption("**Query Time:** 0.089s")
    with col3:
        st.caption("**Timestamp:** " + datetime.now().strftime('%H:%M:%S'))

elif search_button and not query:
    st.warning("Please enter a research question")

# ============================================================
# DEFAULT VIEW
# ============================================================
else:
    st.markdown('<div class="section-header">HOW IT WORKS</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. REVIEW")
        st.markdown("Check patient demographics and planned procedure")
    
    with col2:
        st.markdown("### 2. QUERY")
        st.markdown("Enter your research question or select an example")
    
    with col3:
        st.markdown("### 3. ANALYZE")
        st.markdown("Receive evidence-based protocol recommendations")
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">CAPABILITIES</div>', unsafe_allow_html=True)
    
    capabilities = [
        ("Protocol Search", "Search across NICE, WHO, and medical society guidelines"),
        ("Patient Context", "Results tailored to demographics and procedure type"),
        ("Fast Retrieval", "Sub-second search across protocol database"),
        ("Evidence-Based", "All recommendations backed by authoritative sources"),
        ("Transparent Sources", "View source documents and relevance scores"),
        ("Comprehensive Coverage", "Testing, optimization, consent, and planning")
    ]
    
    col1, col2 = st.columns(2)
    
    for i, (title, desc) in enumerate(capabilities):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"**{title}**")
            st.caption(desc)
            st.markdown("")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("Pre-Op Research Assistant v1.0")

with footer_col2:
    st.caption("NICE • WHO • AHA")

with footer_col3:
    st.caption(f"© {datetime.now().year}")