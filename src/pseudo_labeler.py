"""
Stage 1: Pseudo-Label Generation (Self-Supervised)
===================================================
Fuses two independent signals to infer "true" severity
without using the Ticket Priority column:
  Signal A — Resolution-Time Regression (severity proxy)
  Signal B — Rule-based NLP Features (keyword density,
              negation detection, escalation phrases)

Outputs: df with columns [inferred_severity, mismatch_label]
"""

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


# ── Escalation / urgency vocabulary ──────────────────────────────────────────
ESCALATION_PHRASES = [
    "urgent", "asap", "immediately", "critical", "outage", "down", "broken",
    "not working", "cannot access", "data loss", "security breach", "hack",
    "vulnerability", "payment fail", "billing error", "refund", "lawsuit",
    "legal", "escalate", "manager", "ceo", "frustrated", "unacceptable",
    "disaster", "emergency", "crash", "corrupted", "deleted", "exposed",
    "breach", "stolen", "compromised", "blocked", "unable to login",
]

NEGATION_WORDS = ["not", "can't", "cannot", "couldn't", "won't", "doesn't",
                  "didn't", "never", "no longer", "stopped", "failed"]

LOW_SEVERITY_PHRASES = [
    "how to", "question", "curious", "wondering", "would like to know",
    "feature request", "suggestion", "nice to have", "minor", "cosmetic",
    "typo", "spelling", "small issue",
]

PRIORITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SEVERITY_LABELS = ["Low", "Medium", "High", "Critical"]


# ── Text helpers ──────────────────────────────────────────────────────────────
def _text(row: pd.Series) -> str:
    return f"{row.get('Ticket Subject', '')} {row.get('Ticket Description', '')}".lower()


def nlp_severity_score(df: pd.DataFrame) -> np.ndarray:
    """
    Rule-based NLP score in [0,1].
    Higher = more severe.
    """
    scores = []
    for _, row in df.iterrows():
        text = _text(row)
        words = text.split()
        n = max(len(words), 1)

        esc_hits = sum(1 for p in ESCALATION_PHRASES if p in text)
        neg_hits = sum(1 for p in NEGATION_WORDS if p in text)
        low_hits = sum(1 for p in LOW_SEVERITY_PHRASES if p in text)

        # keyword density
        esc_density = esc_hits / n
        neg_density = neg_hits / n

        # exclamation / caps ratio (urgency proxies)
        excl = text.count("!")
        caps_ratio = sum(1 for c in row.get("Ticket Description", "") if c.isupper()) / max(
            len(row.get("Ticket Description", "")), 1
        )

        score = (
            0.45 * min(esc_density * 20, 1.0)
            + 0.20 * min(neg_density * 10, 1.0)
            + 0.15 * min(excl / 3, 1.0)
            + 0.10 * min(caps_ratio * 5, 1.0)
            - 0.10 * min(low_hits / 2, 1.0)
        )
        scores.append(np.clip(score, 0.0, 1.0))
    return np.array(scores)


def resolution_time_severity(df: pd.DataFrame) -> np.ndarray:
    """
    Normalise resolution time to [0,1].
    Longer resolution → more severe (indirect signal).
    """
    col = "Resolution Time"
    if col not in df.columns:
        return np.full(len(df), 0.5)

    times = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median()
                                                            if df[col].notna().any() else 24)
    # log-scale normalisation
    log_times = np.log1p(times)
    lo, hi = log_times.min(), log_times.max()
    if hi == lo:
        return np.full(len(df), 0.5)
    return (log_times - lo) / (hi - lo)


def score_to_severity(score: float) -> str:
    if score < 0.25:
        return "Low"
    elif score < 0.50:
        return "Medium"
    elif score < 0.75:
        return "High"
    return "Critical"


def generate_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function.
    Returns df with new columns:
      nlp_score, rt_score, fused_score,
      inferred_severity, mismatch_label (0/1),
      mismatch_type ('Hidden Crisis' | 'False Alarm' | 'Consistent')
    """
    df = df.copy()

    print("[Stage 1] Computing NLP severity scores …")
    nlp_scores = nlp_severity_score(df)

    print("[Stage 1] Computing resolution-time severity scores …")
    rt_scores = resolution_time_severity(df)

    # Fuse: weighted average (NLP heavier because resolution time can lag)
    W_NLP, W_RT = 0.60, 0.40
    fused = W_NLP * nlp_scores + W_RT * rt_scores

    df["nlp_score"] = nlp_scores
    df["rt_score"] = rt_scores
    df["fused_score"] = fused
    df["inferred_severity"] = [score_to_severity(s) for s in fused]

    # Binary mismatch label
    inferred_int = df["inferred_severity"].map(PRIORITY_MAP).fillna(1).astype(int)
    assigned_raw = df["Ticket Priority"].str.lower().str.strip() if "Ticket Priority" in df.columns else pd.Series(["medium"] * len(df))
    assigned_int = assigned_raw.map(PRIORITY_MAP).fillna(1).astype(int)

    df["assigned_int"] = assigned_int
    df["inferred_int"] = inferred_int
    delta = inferred_int - assigned_int

    df["severity_delta"] = delta
    df["mismatch_label"] = (delta != 0).astype(int)

    def classify_mismatch(d):
        if d > 0:
            return "Hidden Crisis"   # ticket is more severe than labelled
        elif d < 0:
            return "False Alarm"     # ticket is less severe than labelled
        return "Consistent"

    df["mismatch_type"] = delta.apply(classify_mismatch)

    n_mismatch = df["mismatch_label"].sum()
    print(f"[Stage 1] Done. {n_mismatch}/{len(df)} tickets flagged as mismatch "
          f"({100*n_mismatch/len(df):.1f}%)")
    print(df["mismatch_type"].value_counts().to_string())

    return df


# ── Signal agreement (for evaluation) ────────────────────────────────────────
def signal_agreement(df: pd.DataFrame) -> float:
    """
    Pairwise agreement between NLP and RT signals
    (both mapped to severity quartiles, then compared).
    """
    nlp_sev = [score_to_severity(s) for s in df["nlp_score"]]
    rt_sev = [score_to_severity(s) for s in df["rt_score"]]
    agree = sum(a == b for a, b in zip(nlp_sev, rt_sev))
    return agree / len(df)
