#!/usr/bin/env python3
"""
Intra-Op Assistant Server
Standalone FastAPI server for voice-activated clinical decision support
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import uvicorn
import os
import json
import faiss
import random
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from anthropic import Anthropic

# Load .env from current directory
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()

app = FastAPI(title="Intra-Op Assistant API", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class PatientVitals(BaseModel):
    """Patient vital signs"""
    heart_rate: int = Field(..., description="Heart rate in bpm")
    bp_systolic: int = Field(..., description="Systolic blood pressure")
    bp_diastolic: int = Field(..., description="Diastolic blood pressure")
    oxygen: int = Field(..., description="Oxygen saturation %")
    temp: float = Field(..., description="Temperature in Celsius")
    resp_rate: int = Field(..., description="Respiratory rate")

class PatientInfo(BaseModel):
    """Patient demographic information"""
    name: str = Field(..., description="Patient name")
    age: int = Field(..., description="Patient age")
    sex: str = Field(..., description="Patient sex (M/F)")
    id: str = Field(..., description="Patient ID")
    procedure: str = Field(..., description="Surgical procedure")
    surgeon: str = Field(..., description="Surgeon name")
    time_elapsed: str = Field(..., description="Time elapsed in surgery")
    weight_kg: Optional[int] = Field(default=None, description="Patient weight in kg")
    height_cm: Optional[int] = Field(default=None, description="Patient height in cm")

class IntraOpRequest(BaseModel):
    """Voice command from intra-op assistant"""
    command: str = Field(..., description="Voice transcription")
    context: Optional[str] = Field(default=None, description="Optional context")
    vitals: Optional[PatientVitals] = Field(default=None, description="Current patient vitals")
    patient_info: Optional[PatientInfo] = Field(default=None, description="Patient information")

class ProtocolResult(BaseModel):
    """Retrieved protocol"""
    title: str
    content: str
    source: str
    relevance_score: float

class IntraOpResponse(BaseModel):
    """Response with retrieved protocols"""
    command: str
    protocols: List[ProtocolResult]
    summary: str

# ---------- Globals ----------
_model = None
_index = None
_metadata = None
_chunks = None
_loaded_ok = False

@app.on_event("startup")
def startup():
    """Load all resources on startup"""
    global _model, _index, _metadata, _chunks, _loaded_ok
    
    try:
        # Define paths
        protocol_dir = Path(__file__).parent
        index_path = protocol_dir / "protocol_index.faiss"
        metadata_path = protocol_dir / "protocol_metadata.json"
        chunks_path = protocol_dir / "protocol_chunks.json"
        
        # Check files exist
        if not index_path.exists():
            raise RuntimeError(f"Index not found: {index_path}")
        if not metadata_path.exists():
            raise RuntimeError(f"Metadata not found: {metadata_path}")
        if not chunks_path.exists():
            raise RuntimeError(f"Chunks not found: {chunks_path}")
        
        print("[startup] Loading Sentence Transformer model...")
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print("[startup] Loading FAISS index...")
        _index = faiss.read_index(str(index_path))
        
        print("[startup] Loading metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            _metadata = json.load(f)
        
        print("[startup] Loading chunks...")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            _chunks = json.load(f)
        
        # Check API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("[startup] ⚠️  WARNING: ANTHROPIC_API_KEY not found in environment")
        else:
            print("[startup] ✓ ANTHROPIC_API_KEY loaded")
        
        _loaded_ok = True
        print(f"[startup] ✓ Ready! Index has {_index.ntotal} vectors")
        print(f"[startup] ✓ Metadata entries: {len(_metadata)}")
        print(f"[startup] ✓ Chunks: {len(_chunks)}")
        
    except Exception as e:
        print(f"[startup] ✗ ERROR: {e}")
        _loaded_ok = False
        raise

@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "ok": _loaded_ok,
        "index_vectors": _index.ntotal if _index else 0,
        "metadata_entries": len(_metadata) if _metadata else 0,
        "chunks_total": len(_chunks) if _chunks else 0,
        "api_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }

def _search_protocols(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search for relevant protocols using FAISS"""
    if not _loaded_ok:
        raise HTTPException(500, "Index not loaded")
    
    # Encode query using same model
    query_embedding = _model.encode([query], normalize_embeddings=True)
    
    # Search FAISS
    distances, indices = _index.search(query_embedding, k)
    
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        idx = int(idx)
        if idx < len(_chunks):
            meta = _metadata[idx] if idx < len(_metadata) else {}
            results.append({
                "index": idx,
                "title": meta.get("title", "Protocol"),
                "content": _chunks[idx],
                "source": meta.get("source", "Unknown"),
                "protocol_id": meta.get("protocol_id", ""),
                "score": float(distance)
            })
    
    return results

def _generate_summary(command: str, protocols: List[Dict[str, Any]], vitals: Optional[PatientVitals] = None, patient_info: Optional[PatientInfo] = None) -> str:
    """Generate clinical summary from retrieved protocols with patient context"""
    try:
        # Get API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "Error: ANTHROPIC_API_KEY not set in .env file"
        
        # Create Anthropic client
        client = Anthropic(api_key=api_key)
        
        # Fill in missing vitals and patient info with random data
        if vitals is None:
            vitals = PatientVitals(
                heart_rate=random.randint(60, 100),
                bp_systolic=random.randint(110, 140),
                bp_diastolic=random.randint(70, 90),
                oxygen=random.randint(95, 100),
                temp=round(random.uniform(36.5, 37.5), 1),
                resp_rate=random.randint(12, 18)
            )
        
        if patient_info is None:
            patient_info = PatientInfo(
                name=f"Patient {random.randint(100, 999)}",
                age=random.randint(40, 80),
                sex=random.choice(["M", "F"]),
                id=f"ID: {random.randint(1000000, 9999999)}",
                procedure="General Surgery",
                surgeon="Dr. Unknown",
                time_elapsed="00:00",
                weight_kg=random.randint(60, 100),
                height_cm=random.randint(160, 190)
            )
        
        # Add default values for missing patient info
        if patient_info.weight_kg is None:
            patient_info.weight_kg = random.randint(60, 100)
        if patient_info.height_cm is None:
            patient_info.height_cm = random.randint(160, 190)
        
        # Format protocol context
        protocol_context = "\n".join([
            f"- {p['title']}: {p['content'][:300]}"
            for p in protocols[:3]
        ])
        
        # Format patient context
        patient_context = f"""
PATIENT INFORMATION:
- Name: {patient_info.name}
- Age: {patient_info.age} years old
- Sex: {patient_info.sex}
- Weight: {patient_info.weight_kg} kg
- Height: {patient_info.height_cm} cm
- Procedure: {patient_info.procedure}
- Surgeon: {patient_info.surgeon}
- Time Elapsed: {patient_info.time_elapsed}

CURRENT VITAL SIGNS:
- Heart Rate: {vitals.heart_rate} bpm
- Blood Pressure: {vitals.bp_systolic}/{vitals.bp_diastolic} mmHg
- Oxygen Saturation: {vitals.oxygen}%
- Temperature: {vitals.temp}°C
- Respiratory Rate: {vitals.resp_rate} breaths/min
"""
        
        prompt = f"""You are a clinical decision support assistant for intra-operative care.

{patient_context}

A surgeon asked: "{command}"

Here are relevant clinical protocols:
{protocol_context}

IMPORTANT: You MUST explicitly reference and analyze the patient's current vital signs in your response. Your recommendations must be directly informed by:
1. The specific vital sign values (heart rate, blood pressure, oxygen saturation, temperature, respiratory rate)
2. Whether vitals are within normal ranges or abnormal
3. How vitals may affect the clinical decision for this patient
4. Patient age and weight when relevant to dosing or interventions

Provide a concise, actionable clinical response (2-3 sentences) that:
- Directly addresses the surgeon's request based on the protocols
- Explicitly mentions relevant vital sign findings
- References the patient's specific demographic factors if applicable
- Explains how vital signs influence your recommendations


Be sure to keep it short! The speech to text version should not be more than 30 seconds long! This is critical!
"""
        
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# ---------- Endpoints ----------
@app.post("/api/assist", response_model=IntraOpResponse)
def assist(body: IntraOpRequest):
    """
    Voice-activated clinical decision support
    Takes a voice command and returns relevant protocols with AI-generated summary
    """
    
    if not _loaded_ok:
        raise HTTPException(503, "Protocol index not available")
    
    # Build search query
    search_query = body.command
    if body.context:
        search_query = f"{body.context}. {body.command}"
    
    # Search for relevant protocols
    protocols = _search_protocols(search_query, k=5)
    
    if not protocols:
        raise HTTPException(404, "No relevant protocols found")
    
    # Generate summary with patient context
    summary = _generate_summary(body.command, protocols, body.vitals, body.patient_info)
    
    # Format response
    results = [
        ProtocolResult(
            title=p["title"],
            content=p["content"],
            source=p["source"],
            relevance_score=p["score"]
        )
        for p in protocols
    ]
    
    return IntraOpResponse(
        command=body.command,
        protocols=results,
        summary=summary
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting Intra-Op Assistant Server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)