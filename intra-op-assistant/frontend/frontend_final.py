import streamlit as st
from streamlit_mic_recorder import mic_recorder
from google.cloud import speech_v1
from google.cloud import texttospeech_v1
from dotenv import load_dotenv
import wave
import io
import time
import requests
import random

load_dotenv()

st.set_page_config(page_title="Intra-Op Assistant", page_icon=None, layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        * { margin: 0; padding: 0; }
        html, body { height: 100%; }
        body { background: #0f172a !important; color: #e2e8f0; }
        [data-testid="stAppViewContainer"] { padding: 0 !important; background: #0f172a !important; }
        [data-testid="stBottomBlockContainer"] { display: none; }
        [data-testid="stSidebar"] { background: #1a2332 !important; }
        .stButton button { 
            background: #0ea5e9 !important;
            color: white !important;
            font-weight: 600 !important;
            width: 100% !important;
            border: none !important;
            padding: 20px !important;
            font-size: 16px !important;
        }
        .main { padding: 20px; }
        h1 { color: #0ea5e9; margin: 20px 0; font-size: 2em; }
        .spacer { height: 40px; }
        .result-box {
            background: rgba(20, 83, 45, 0.2);
            border-left: 3px solid #10b981;
            padding: 20px;
            margin: 20px 0;
            font-size: 15px;
            line-height: 1.6;
        }
        .result-title { color: #10b981; font-weight: 600; font-size: 1.1em; margin-bottom: 10px; }
        .result-text { color: #cbd5e1; }
        audio { width: 100%; margin: 20px 0; }
        
        .vital-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
            border-left: 3px solid #0ea5e9;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .vital-label { color: #94a3b8; font-size: 12px; font-weight: 600; }
        .vital-value { color: #0ea5e9; font-size: 24px; font-weight: bold; }
        .vital-unit { color: #64748b; font-size: 12px; }
        .patient-info { color: #cbd5e1; font-size: 13px; line-height: 1.6; }
        .patient-header { color: #0ea5e9; font-weight: 600; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

try:
    speech_client = speech_v1.SpeechClient()
    tts_client = texttospeech_v1.TextToSpeechClient()
except Exception as e:
    st.error(f"Google Cloud init failed: {e}")
    st.stop()

def transcribe_audio(audio_bytes):
    try:
        wav_file = wave.open(io.BytesIO(audio_bytes), 'rb')
        sample_rate = wav_file.getframerate()
        wav_file.close()
        
        audio = speech_v1.RecognitionAudio(content=audio_bytes)
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
        )
        response = speech_client.recognize(config=config, audio=audio)
        if response.results:
            return response.results[0].alternatives[0].transcript
        return ""
    except:
        return ""

def generate_speech(text):
    try:
        synthesis_input = texttospeech_v1.SynthesisInput(text=text)
        voice = texttospeech_v1.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-C",
        )
        audio_config = texttospeech_v1.AudioConfig(
            audio_encoding=texttospeech_v1.AudioEncoding.MP3,
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        return response.audio_content
    except:
        return None

def call_intraop_backend(command: str) -> dict:
    try:
        backend_url = "http://localhost:8001/api/assist"
        payload = {"command": command}
        response = requests.post(backend_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except:
        return None

# Initialize patient vitals
if 'vitals' not in st.session_state:
    st.session_state.vitals = {
        "heart_rate": 75,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "oxygen": 98,
        "temp": 37.0,
        "resp_rate": 14,
    }

if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {
        "name": "John Doe",
        "age": 58,
        "sex": "M",
        "id": "ID: 2847563",
        "procedure": "Coronary Artery Bypass",
        "surgeon": "Dr. Smith",
        "time_elapsed": "00:45",
    }

if 'last_vital_update' not in st.session_state:
    st.session_state.last_vital_update = 0

if 'response' not in st.session_state:
    st.session_state.response = None
if 'audio_content' not in st.session_state:
    st.session_state.audio_content = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

# Update vitals every 2 seconds
current_time = time.time()
if current_time - st.session_state.last_vital_update > 2:
    st.session_state.vitals["heart_rate"] = max(60, min(100, st.session_state.vitals["heart_rate"] + random.randint(-3, 3)))
    st.session_state.vitals["bp_systolic"] = max(100, min(140, st.session_state.vitals["bp_systolic"] + random.randint(-2, 2)))
    st.session_state.vitals["bp_diastolic"] = max(60, min(90, st.session_state.vitals["bp_diastolic"] + random.randint(-2, 2)))
    st.session_state.vitals["oxygen"] = max(95, min(100, st.session_state.vitals["oxygen"] + random.randint(-1, 1)))
    st.session_state.vitals["temp"] = round(st.session_state.vitals["temp"] + random.uniform(-0.1, 0.1), 1)
    st.session_state.vitals["resp_rate"] = max(12, min(18, st.session_state.vitals["resp_rate"] + random.randint(-1, 1)))
    st.session_state.last_vital_update = current_time

# Sidebar with patient vitals
with st.sidebar:
    st.markdown("### Patient Info")
    st.markdown(f"""
    <div class='patient-info'>
    <strong>{st.session_state.patient_info['name']}</strong><br>
    Age: {st.session_state.patient_info['age']} | {st.session_state.patient_info['sex']}<br>
    {st.session_state.patient_info['id']}<br>
    <br>
    <strong>Procedure:</strong> {st.session_state.patient_info['procedure']}<br>
    <strong>Surgeon:</strong> {st.session_state.patient_info['surgeon']}<br>
    <strong>Time:</strong> {st.session_state.patient_info['time_elapsed']}
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### Vital Signs")
    
    st.markdown(f"""
    <div class='vital-card'>
        <div class='vital-label'>Heart Rate</div>
        <div class='vital-value'>{st.session_state.vitals['heart_rate']}</div>
        <div class='vital-unit'>bpm</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='vital-card'>
        <div class='vital-label'>Blood Pressure</div>
        <div class='vital-value'>{st.session_state.vitals['bp_systolic']}/{st.session_state.vitals['bp_diastolic']}</div>
        <div class='vital-unit'>mmHg</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='vital-card'>
        <div class='vital-label'>Oxygen Saturation</div>
        <div class='vital-value'>{st.session_state.vitals['oxygen']}</div>
        <div class='vital-unit'>%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='vital-card'>
        <div class='vital-label'>Temperature</div>
        <div class='vital-value'>{st.session_state.vitals['temp']}</div>
        <div class='vital-unit'>°C</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='vital-card'>
        <div class='vital-label'>Respiratory Rate</div>
        <div class='vital-value'>{st.session_state.vitals['resp_rate']}</div>
        <div class='vital-unit'>breaths/min</div>
    </div>
    """, unsafe_allow_html=True)

# Main content area
st.markdown("<h1>Intra-Op Assistant</h1>", unsafe_allow_html=True)

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

# Recording button - full width, no columns
state = mic_recorder(
    start_prompt="Record",
    stop_prompt="Stop",
    just_once=False,
    use_container_width=True,
    format="wav",
    key=f"mic_recorder_{st.session_state.reset_key}",
)

# Processing
if state and not st.session_state.response:
    audio_bytes = state['bytes']
    if len(audio_bytes) > 1000:
        with st.spinner("Processing..."):
            time.sleep(0.5)
            transcript = transcribe_audio(audio_bytes)
            
            if transcript:
                st.session_state.transcript = transcript
                backend_response = call_intraop_backend(transcript)
                
                if backend_response:
                    st.session_state.response = backend_response
                    summary_text = backend_response.get("summary", "")
                    if summary_text:
                        st.session_state.audio_content = generate_speech(summary_text)
                st.rerun()

# Results
if st.session_state.response:
    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    
    # Show what user said
    if st.session_state.transcript:
        st.markdown(f"""
        <div class='result-box' style='border-left: 3px solid #0ea5e9;'>
            <div class='result-title' style='color: #0ea5e9;'>You said:</div>
            <div class='result-text'>{st.session_state.transcript}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show protocols
    protocols = st.session_state.response.get("protocols", [])
    if protocols:
        for protocol in protocols[:2]:
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-title'>{protocol.get('title', 'Protocol')}</div>
                <div class='result-text'>{protocol.get('content', '')[:300]}...</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show summary
    summary = st.session_state.response.get("summary", "")
    if summary:
        st.markdown(f"""
        <div class='result-box' style='border-left: 3px solid #f59e0b;'>
            <div class='result-title' style='color: #f59e0b;'>Clinical Summary:</div>
            <div class='result-text'>{summary}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.audio_content:
        st.audio(st.session_state.audio_content, format="audio/mp3", autoplay=True)
    
    # Button to record again
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Record Again", use_container_width=True, key="record_again_btn"):
            st.session_state.response = None
            st.session_state.audio_content = None
            st.session_state.transcript = None
            st.session_state.reset_key += 1
            st.rerun()