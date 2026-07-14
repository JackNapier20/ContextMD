#!/usr/bin/env python3
"""
Citation faithfulness evaluation (ALCE-style), judged by Claude.

ContextMD emits claims with bracketed citations like [2]. A medical tool's
citations must be verifiable, so we measure two things per report:

  * Citation precision — of all [#] citations, the fraction where the cited
                         abstract actually supports the attached claim.
  * Citation recall    — of all substantive claim sentences, the fraction that
                         carry at least one citation (how grounded the answer is).

Each [#] is mapped back to its source abstract via the case's `retrieved` order
(the same order generation.py numbers the references). Claude judges support in
one batched call per report.

Requires ANTHROPIC_API_KEY (read from .env or the environment).

Usage:
  python eval/eval_citations.py                    # all cases with a report
  python eval/eval_citations.py --limit 5
  python eval/eval_citations.py --model claude-3-5-sonnet-20241022
"""
from __future__ import annotations

import os
import re
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env", override=False)

from _common import (  # noqa: E402
    find_case_pairs, load_case, extract_answer_text, claim_lines,
    citations_in, ref_context, CASES_DIR,
)

try:
    from anthropic import Anthropic  # noqa: E402
except Exception as e:  # pragma: no cover
    raise SystemExit("Missing dependency: anthropic\npip install anthropic") from e

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_JUDGE = "claude-sonnet-4-5-20250929"
FALLBACK_JUDGE = "claude-haiku-4-5-20251001"

JUDGE_INSTR = (
    "You are a careful biomedical fact-checker. For each item you are given a "
    "CLAIM from a clinical summary and the ABSTRACT it cites. Decide whether the "
    "abstract provides support for the claim.\n"
    "Respond ONLY with a JSON array; one object per item, in order:\n"
    '[{"i": <item number>, "supported": true|false, "reason": "<=15 words}]\n'
    "Mark supported=true only if the abstract substantiates the specific claim."
)


def _extract_json(text: str) -> Any:
    """Pull the first JSON array out of a model response."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in judge output: {text[:200]}")
    return json.loads(m.group(0))


def judge_case(client: Anthropic, model: str, items: List[Dict[str, str]]) -> List[Dict]:
    """items: [{claim, title, abstract}]. Returns verdict dicts aligned by index."""
    if not items:
        return []
    blocks = []
    for i, it in enumerate(items, 1):
        ab = it["abstract"][:1500]
        blocks.append(f"ITEM {i}\nCLAIM: {it['claim']}\nABSTRACT ([{it['cite']}] {it['title']}): {ab}")
    prompt = JUDGE_INSTR + "\n\n" + "\n\n".join(blocks)
    msg = client.messages.create(
        model=model, max_tokens=1500, temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    verdicts = _extract_json(text)
    by_i = {int(v.get("i", idx + 1)): v for idx, v in enumerate(verdicts)}
    out = []
    for idx in range(1, len(items) + 1):
        v = by_i.get(idx, {"supported": False, "reason": "no verdict"})
        out.append({"supported": bool(v.get("supported")), "reason": v.get("reason", "")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Citation precision/recall via Claude")
    ap.add_argument("--cases_dir", type=str, default=str(CASES_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", type=str, default=DEFAULT_JUDGE)
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR))
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (add to .env or export it).")
    client = Anthropic(api_key=key)
    model = args.model

    pairs = find_case_pairs(Path(args.cases_dir))
    if args.limit > 0:
        pairs = pairs[-args.limit:]
    if not pairs:
        raise SystemExit("No case/report pairs found.")

    tot_cites = 0
    tot_supported = 0
    tot_claims = 0
    tot_cited_claims = 0
    per_case = []

    print(f"Judging citations in {len(pairs)} reports with {model}…\n")
    for case_path, md_path in pairs:
        case = load_case(case_path)
        answer = extract_answer_text(md_path.read_text(encoding="utf-8"))
        claims = claim_lines(answer)

        items = []
        cited_claims = 0
        for cl in claims:
            cites = citations_in(cl)
            if cites:
                cited_claims += 1
            for c in cites:
                ref = ref_context(case, c)
                if ref and ref["abstract"]:
                    items.append({"claim": cl, "cite": c,
                                  "title": ref["title"], "abstract": ref["abstract"]})

        try:
            verdicts = judge_case(client, model, items)
        except Exception as e:
            if model != FALLBACK_JUDGE:
                print(f"  ({model} failed: {e}; retrying with {FALLBACK_JUDGE})")
                model = FALLBACK_JUDGE
                verdicts = judge_case(client, model, items)
            else:
                raise

        supported = sum(1 for v in verdicts if v["supported"])
        n_cites = len(items)
        precision = supported / n_cites if n_cites else float("nan")
        recall = cited_claims / len(claims) if claims else float("nan")

        tot_cites += n_cites
        tot_supported += supported
        tot_claims += len(claims)
        tot_cited_claims += cited_claims

        per_case.append({
            "case": case_path.stem,
            "n_claims": len(claims),
            "n_cited_claims": cited_claims,
            "n_citations": n_cites,
            "n_supported": supported,
            "citation_precision": None if n_cites == 0 else round(precision, 4),
            "citation_recall": None if not claims else round(recall, 4),
        })
        p = "n/a" if n_cites == 0 else f"{precision:.2f}"
        r = "n/a" if not claims else f"{recall:.2f}"
        print(f"  {case_path.stem}: precision={p} ({supported}/{n_cites})  recall={r} "
              f"({cited_claims}/{len(claims)})")

    micro_precision = tot_supported / tot_cites if tot_cites else float("nan")
    micro_recall = tot_cited_claims / tot_claims if tot_claims else float("nan")

    print("\n=== CITATION FAITHFULNESS (micro-averaged over all reports) ===")
    print(f"  Citation precision: {micro_precision:.3f}  ({tot_supported}/{tot_cites} citations supported)")
    print(f"  Citation recall:    {micro_recall:.3f}  ({tot_cited_claims}/{tot_claims} claims cited)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) / f"citations_{ts}.json"
    out.write_text(json.dumps({
        "judge_model": model,
        "micro_precision": round(micro_precision, 4) if tot_cites else None,
        "micro_recall": round(micro_recall, 4) if tot_claims else None,
        "totals": {"citations": tot_cites, "supported": tot_supported,
                   "claims": tot_claims, "cited_claims": tot_cited_claims},
        "per_case": per_case,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
