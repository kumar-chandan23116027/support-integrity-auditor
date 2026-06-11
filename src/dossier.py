"""
Stage 3: Evidence Dossier Generation
=====================================
For every ticket classified as a mismatch, produces a
structured JSON dossier with ZERO hallucination — every
field_evidence item is traceable to an actual input column.
"""

import json
import re
from typing import Dict, Any, List

from src.pseudo_labeler import ESCALATION_PHRASES, NEGATION_WORDS, LOW_SEVERITY_PHRASES


PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PRIORITY_LABELS = ["Low", "Medium", "High", "Critical"]


def _find_keywords_in_text(text: str, vocab: List[str]) -> List[str]:
    text_l = text.lower()
    return [p for p in vocab if p in text_l]


def _resolution_time_interpretation(rt_val, inferred_sev: str) -> str:
    try:
        rt = float(rt_val)
    except (TypeError, ValueError):
        return "Resolution time unavailable."
    if rt < 6:
        return f"Resolved in {rt:.1f}h — typically seen for low-severity tickets."
    elif rt < 24:
        return f"Resolved in {rt:.1f}h — moderate resolution window."
    elif rt < 72:
        return f"Resolved in {rt:.1f}h — extended time suggests elevated complexity."
    else:
        return f"Resolved in {rt:.1f}h — very long resolution time, consistent with {inferred_sev} severity."


def _build_feature_evidence(row, text: str) -> List[Dict]:
    evidence = []

    # 1. Escalation keyword evidence
    esc_hits = _find_keywords_in_text(text, ESCALATION_PHRASES)
    if esc_hits:
        evidence.append({
            "signal": "keyword",
            "value": esc_hits[:5],  # cap at 5 for readability
            "weight": round(min(len(esc_hits) / 5, 1.0), 2),
            "source_field": "Ticket Subject + Ticket Description",
        })

    # 2. Negation evidence
    neg_hits = _find_keywords_in_text(text, NEGATION_WORDS)
    if neg_hits:
        evidence.append({
            "signal": "negation_detection",
            "value": neg_hits[:3],
            "weight": round(min(len(neg_hits) / 3, 1.0), 2),
            "source_field": "Ticket Description",
        })

    # 3. Low-severity phrase evidence
    low_hits = _find_keywords_in_text(text, LOW_SEVERITY_PHRASES)
    if low_hits:
        evidence.append({
            "signal": "low_severity_phrase",
            "value": low_hits[:3],
            "weight": round(-min(len(low_hits) / 2, 1.0), 2),
            "source_field": "Ticket Description",
        })

    # 4. Resolution time evidence
    rt_val = row.get("Resolution Time", None)
    if rt_val is not None:
        inferred_sev = str(row.get("inferred_severity", "Unknown"))
        interpretation = _resolution_time_interpretation(rt_val, inferred_sev)
        evidence.append({
            "signal": "resolution_time",
            "value": str(rt_val),
            "interpretation": interpretation,
            "source_field": "Resolution Time",
        })

    # 5. Channel evidence
    channel = str(row.get("Ticket Channel", "")).strip()
    if channel:
        channel_weight = {
            "phone": 0.8, "chat": 0.6, "email": 0.4,
            "social media": 0.9, "web": 0.3,
        }.get(channel.lower(), 0.5)
        evidence.append({
            "signal": "intake_channel",
            "value": channel,
            "weight": channel_weight,
            "interpretation": (
                f"Channel '{channel}' — "
                + ("high-urgency intake method." if channel_weight >= 0.7
                   else "standard intake method.")
            ),
            "source_field": "Ticket Channel",
        })

    return evidence


def _constraint_analysis(row, delta: int, mismatch_type: str) -> str:
    assigned = str(row.get("Ticket Priority", "Unknown"))
    inferred = str(row.get("inferred_severity", "Unknown"))
    text = f"{row.get('Ticket Subject', '')} {row.get('Ticket Description', '')}".strip()

    if mismatch_type == "Hidden Crisis":
        return (
            f"The ticket was assigned priority '{assigned}', but linguistic and temporal "
            f"signals indicate true severity is '{inferred}'. The description contains "
            f"urgency markers that were likely missed during manual triage. "
            f"SLA risk: HIGH — immediate re-prioritisation recommended."
        )
    elif mismatch_type == "False Alarm":
        return (
            f"The ticket was assigned priority '{assigned}', inflated relative to the "
            f"inferred severity of '{inferred}'. Language is non-urgent and resolution "
            f"time is short. This consumes high-priority queue bandwidth unnecessarily."
        )
    return "Ticket appears consistently prioritised."


def generate_dossier(row: dict) -> Dict[str, Any]:
    """
    Generates a hallucination-free Evidence Dossier for one ticket.
    All fields are grounded in actual input data.
    """
    text = f"{row.get('Ticket Subject', '')} {row.get('Ticket Description', '')}".strip()

    assigned_priority = str(row.get("Ticket Priority", "Unknown"))
    inferred_severity = str(row.get("inferred_severity", "Unknown"))
    mismatch_type = str(row.get("mismatch_type", "Unknown"))

    ap_int = {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(
        assigned_priority.lower(), 1)
    inf_int = {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(
        inferred_severity.lower(), 1)
    delta = inf_int - ap_int

    feature_evidence = _build_feature_evidence(row, text)
    constraint_analysis = _constraint_analysis(row, delta, mismatch_type)

    confidence = row.get("mismatch_confidence", row.get("fused_score", 0.5))
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.5

    dossier = {
        "ticket_id": str(row.get("Ticket ID", row.get("ticket_id", "N/A"))),
        "assigned_priority": assigned_priority,
        "inferred_severity": inferred_severity,
        "mismatch_type": mismatch_type,
        "severity_delta": delta,
        "feature_evidence": feature_evidence,
        "constraint_analysis": constraint_analysis,
        "confidence": round(confidence, 4),
    }
    return dossier


def generate_all_dossiers(df, output_path: str = None) -> List[Dict]:
    """Generate dossiers for all mismatch tickets."""
    mismatch_col = "pred_mismatch" if "pred_mismatch" in df.columns else "mismatch_label"
    mismatches = df[df[mismatch_col] == 1]

    print(f"[Stage 3] Generating dossiers for {len(mismatches)} mismatch tickets …")
    dossiers = []
    for _, row in mismatches.iterrows():
        d = generate_dossier(row.to_dict())
        dossiers.append(d)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(dossiers, f, indent=2)
        print(f"[Stage 3] Dossiers saved → {output_path}")

    return dossiers
