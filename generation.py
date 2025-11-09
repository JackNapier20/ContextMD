#!/usr/bin/env python3
# send_to_haiku.py
# Find the latest case JSON in ./Cases, assemble a prompt using patient data + up to 5 abstracts,
# call Claude 3 Haiku, and save a Markdown report next to the case file.

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Project paths and optional .env
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=str(ENV_PATH), override=False)

# If running locally, you can set the key here; CLI flag can still override this.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

try:
    from anthropic import Anthropic
    # Optional: nicer exception handling if available in your SDK version
    try:
        from anthropic import BadRequestError, AuthenticationError, APIStatusError
    except Exception:  # older SDKs may not expose these
        BadRequestError = AuthenticationError = APIStatusError = Exception
except Exception as e:
    raise SystemExit(
        "Missing dependency: anthropic\nInstall with: pip install anthropic"
    ) from e
CASES_DIR = PROJECT_ROOT / "Cases"
MODEL = "claude-3-haiku-20240307"
MAX_REFS = 5
MAX_TOKENS = 1200  # adjust as you like

def find_latest_case() -> Path:
    if not CASES_DIR.exists():
        raise FileNotFoundError(f"Cases folder not found: {CASES_DIR}")
    candidates = sorted(CASES_DIR.glob("case_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No case_*.json files found in {CASES_DIR}")
    return candidates[-1]  # newest by name (timestamp-ordered)

def load_case(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def patient_block(patient: dict) -> str:
    fields = [
        ("Age", patient.get("age")),
        ("Sex", patient.get("sex")),
        ("Chief complaint", patient.get("complaint")),
        ("History", patient.get("history")),
        ("Medications", patient.get("meds")),
        ("Vitals", patient.get("vitals")),
        ("Labs", patient.get("labs")),
        ("Exam", patient.get("exam")),
        ("Notes", patient.get("notes")),
    ]
    lines = ["PATIENT CASE:"]
    for label, val in fields:
        if val:
            lines.append(f"- {label}: {val}")
    return "\n".join(lines) if len(lines) > 1 else "PATIENT CASE: (none provided)"

def select_refs(retrieved: list, max_refs: int = MAX_REFS) -> list:
    # Prefer entries with non-empty abstracts, keep original ranking order
    with_abs = [r for r in retrieved if isinstance(r, dict) and (r.get("abstract") or "").strip()]
    chosen = with_abs[:max_refs]
    # If fewer than max_refs, top-up with items that have empty abstracts (still valuable for titles/PMIDs)
    if len(chosen) < max_refs:
        no_abs = [r for r in retrieved if isinstance(r, dict) and not (r.get("abstract") or "").strip()]
        chosen.extend(no_abs[: max_refs - len(chosen)])
    return chosen

def refs_block(refs: list) -> str:
    lines = ["REFERENCES:"]
    for i, h in enumerate(refs, 1):
        pmid = h.get("pmid", "")
        title = (h.get("title") or "").strip()
        abstract = (h.get("abstract") or "").strip().replace("\n", " ")
        if len(abstract) > 900:
            abstract = abstract[:900] + "..."
        lines.append(f"[{i}] PMID {pmid}: {title}\n{abstract}")
    return "\n\n".join(lines)

def reference_legend(refs: list) -> str:
    lines = ["### Reference legend"]
    for i, h in enumerate(refs, 1):
        pmid = str(h.get("pmid", "")).strip()
        title = (h.get("title") or "").strip()
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        if pmid:
            lines.append(f"[{i}] PMID {pmid}: {title}\n    {url}")
        else:
            lines.append(f"[{i}] (no PMID): {title}")
    return "\n".join(lines)

def build_prompt(patient: dict, refs: list) -> str:
    instructions = (
        "You are an assistant for a doctor who helps dealing with research easy. You are given the patient medical data along with some relevant research paper abstracts.\n"
        "Using these, produce:\n"
        "1) **Summary** — concise, plain-English synthesis tailored to the patient.\n"
        "2) **Insights** — bullets about potentially relevant considerations (tests, risks, confounders, options),\n"
        "   clearly distinguishing evidence-based statements from general clinical considerations.\n"
        "Do **not** provide medical advice or a definitive diagnosis. Cite references using [#], where # corresponds to the numbering in the REFERENCES list below.\n"
        "Optionally include the PMID in the citation like [#; PMID 12345678] if it helps clarity."
    )
    return f"{instructions}\n\n{patient_block(patient)}\n\n{refs_block(refs)}\n\n" \
           "RESPONSE FORMAT:\n" \
           "## Summary\n" \
           "- ...\n\n" \
           "## Insights\n" \
           "- ... [#]\n- ... [#]"

def call_haiku(prompt: str, api_key_override: str = "") -> str:
    # Choose source: CLI override > hardcoded variable > .env
    raw = (api_key_override or ANTHROPIC_API_KEY)
    if raw is None:
        raise RuntimeError(
            f"ANTHROPIC_API_KEY missing. Provide via --api_key, hardcode ANTHROPIC_API_KEY, or set it in {ENV_PATH} as:\n"
            "ANTHROPIC_API_KEY=sk-ant-...  (no quotes)"
        )

    # Strip quotes, whitespace, and common invisible chars (BOM/ZWS)
    key = str(raw).strip().strip('"').strip("'")
    key = key.replace("\ufeff", "").replace("\u200b", "")
    key = re.sub(r"\s+$|^\s+", "", key)

    # Quick sanity check
    if not key or not key.startswith("sk-ant-"):
        masked = (raw[:6] + "…" + raw[-4:]) if raw else "<empty>"
        raise RuntimeError(
            "ANTHROPIC_API_KEY looks malformed. It should start with 'sk-ant-'.\n"
            f"Read from {source}. Current (masked): {masked}"
        )

    # Show a minimal masked debug line so you can confirm which key is being used
    print(f"Using ANTHROPIC_API_KEY from .env (masked): {key[:8]}…{key[-4:]} (len={len(key)})")

    client = Anthropic(api_key=key)

    # Preflight: validate the key with a cheap call
    try:
        _ = client.models.list()
    except Exception as e:
        raise RuntimeError(
            "Anthropic API key validation failed (models.list). If this shows 401/invalid x-api-key, regenerate your key "
            "in the Anthropic console and update your .env."
        ) from e

    # Actual completion call
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    # Messages API returns a list of content blocks; join any text parts
    parts = []
    for block in msg.content:
        if getattr(block, "type", None) == "text" or (isinstance(block, dict) and block.get("type") == "text"):
            parts.append(block.get("text") if isinstance(block, dict) else block.text)
    return "\n".join(parts).strip()

def save_markdown(case_path: Path, text: str) -> Path:
    stem = case_path.stem  # e.g., case_20250101T010203Z
    out_path = case_path.with_name(stem + ".haiku.md")
    out_path.write_text(text, encoding="utf-8")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Generate a Haiku summary for the latest case")
    ap.add_argument("--api_key", type=str, default="", help="Override Anthropic API key (use quotes)")
    args = ap.parse_args()

    case_path = find_latest_case()
    case = load_case(case_path)

    if not ENV_PATH.exists():
        print(f"Warning: {ENV_PATH} not found. If you don't pass --api_key, no key will be loaded from file.")

    patient = case.get("patient", {}) or {}
    retrieved = case.get("retrieved", []) or []
    refs = select_refs(retrieved, MAX_REFS)

    prompt = build_prompt(patient, refs)
    print(f"→ Sending prompt to {MODEL} for: {case_path.name} (refs={len(refs)})")
    output = call_haiku(prompt, api_key_override=args.api_key)

    # Append a legend mapping [#] to PMIDs and PubMed links for clarity
    legend = reference_legend(refs)
    output_with_legend = output + "\n\n---\n" + legend + "\n"

    out_md = save_markdown(case_path, output_with_legend)
    print(f"Saved Haiku report: {out_md}")

    print("\n--- Haiku Output (preview) ---\n")
    print(output_with_legend)

if __name__ == "__main__":
    main()