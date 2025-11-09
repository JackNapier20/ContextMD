# Pre-Op Research Assistant - Setup Guide

## Quick Start (Hackathon Mode)

This guide gets you running end-to-end in demo mode. Once integrated with your actual FAISS index and generation modules, just swap in the real paths.

---

## Prerequisites

```bash
pip install streamlit requests fastapi uvicorn pydantic
```

If using your FAISS index:

```bash
pip install faiss-cpu  # or faiss-gpu
pip install anthropic
```

---

## Directory Structure

```
project/
├── app.py                    # Streamlit frontend
├── server_app.py             # FastAPI backend (place in server/app.py in your repo)
├── data/
│   ├── index.faiss           # Your FAISS index (optional for demo)
│   └── metadata.json         # Paper metadata (optional for demo)
├── cases/                    # Case files saved here (auto-created)
└── server/
    ├── settings.py           # Your settings file
    ├── app.py                # FastAPI app
    └── ...
```

---

## Running in Demo Mode (No Index Needed)

### Terminal 1: Start Backend

```bash
python server_app.py
```

You should see:

```
[startup] Index file not found at ./data/index.faiss - running in demo mode
Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start Frontend

```bash
streamlit run app.py
```

You should see:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8001
```

### Browser

Open `http://localhost:8001` and try:

1. Edit patient info in sidebar (optional)
2. Enter query: "What preoperative tests are required for this patient?"
3. Click SEARCH
4. See demo results + Claude synthesis

---

## Integrating Your Real Components

### Step 1: Add Your FAISS Index

Move/link your FAISS index to `./data/index.faiss`:

```bash
ln -s /path/to/your/index.faiss ./data/index.faiss
```

### Step 2: Add Your Generation Module

Replace the demo `call_haiku()` in `server_app.py` with your real one:

```python
from generation import call_haiku  # Your real implementation
```

### Step 3: Set API Key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or pass it via the frontend (already supports this in code).

### Step 4: Use Real Settings

Instead of the fallback in `server_app.py`, import from your repo:

```python
from server.settings import settings
from server.indexing import search
from server.search import load_index, build_metadata_index, embed_query
```

---

## Key Changes from Original

### Frontend (`app.py`)

✅ **Editable patient fields** - All sidebar inputs can be modified
✅ **Backend integration** - Connects to `/api/cases` and `/api/generate`
✅ **Session state** - Preserves data between reruns
✅ **Real-time results** - Displays markdown from backend
✅ **Error handling** - Shows helpful messages if backend is down
✅ **Removed dead code** - Cut "COMMON QUESTIONS", "HOW IT WORKS", demo sections

### Backend (`server_app.py`)

✅ **Demo mode** - Works without FAISS index (returns fake results)
✅ **Better error handling** - HTTPException with proper messages
✅ **Settings fallback** - Doesn't crash if settings.py missing
✅ **Case auto-creation** - Creates `./cases/` directory if needed
✅ **Optional imports** - Gracefully handles missing modules
✅ **Improved `/api/cases`** - Also performs search (not just saves)

---

## Testing Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Should return:

```json
{
  "ok": false,
  "mode": "demo",
  "metadata_size": 0,
  "nprobe_default": 32
}
```

### Search Endpoint

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "preoperative testing for elderly patients",
    "topk": 10
  }'
```

### Create Case

```bash
curl -X POST http://localhost:8000/api/cases \
  -H "Content-Type: application/json" \
  -d '{
    "patient": {
      "age": "65",
      "sex": "F",
      "complaint": "Cholecystectomy",
      "history": "Hypertension, Diabetes"
    },
    "query": "What tests are needed?"
  }'
```

### Generate Synthesis

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "case_path": "./cases/case_20250109T120000Z.json"
  }'
```

---

## Common Issues

### ❌ "Cannot connect to backend"

- Check backend is running: `curl http://localhost:8000/health`
- Check port 8000 is not in use: `lsof -i :8000`

### ❌ "Index not loaded" but expected real results

- Verify FAISS index at `./data/index.faiss`
- Check metadata loaded: `curl http://localhost:8000/health` → `metadata_size > 0`

### ❌ Streamlit reruns on every keystroke

- This is normal Streamlit behavior
- Session state preserves your work between reruns

### ❌ Generation returns placeholder text

- Missing Claude API key: `export ANTHROPIC_API_KEY="..."`
- Or check if `generation.py` is actually being imported
