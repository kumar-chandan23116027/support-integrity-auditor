"""
Data loading and preprocessing utilities.
Handles the Kaggle CRM dataset + synthetic fallback.
"""

import os
import pandas as pd
import numpy as np


REQUIRED_COLS = [
    "Ticket Subject", "Ticket Description", "Ticket Priority",
    "Ticket Channel", "Resolution Time", "Ticket Type",
]

PRIORITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the CRM dataset. Falls back to synthetic data if path not found.
    """
    if path and os.path.exists(path):
        print(f"[Data] Loading from {path} …")
        df = pd.read_csv(path)
        df = _standardise_columns(df)
        print(f"[Data] Loaded {len(df)} rows, {df.shape[1]} columns.")
        return df

    print("[Data] Dataset not found — generating synthetic data for demonstration …")
    return _generate_synthetic_data(n=2000)


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and fill missing fields."""
    # Try to map common variants
    rename_map = {}
    for col in df.columns:
        col_l = col.lower().strip()
        if "subject" in col_l and "ticket" in col_l:
            rename_map[col] = "Ticket Subject"
        elif "description" in col_l:
            rename_map[col] = "Ticket Description"
        elif "priority" in col_l:
            rename_map[col] = "Ticket Priority"
        elif "channel" in col_l:
            rename_map[col] = "Ticket Channel"
        elif "resolution" in col_l and "time" in col_l:
            rename_map[col] = "Resolution Time"
        elif "type" in col_l and "ticket" in col_l:
            rename_map[col] = "Ticket Type"
        elif "id" in col_l and "ticket" in col_l:
            rename_map[col] = "Ticket ID"

    df = df.rename(columns=rename_map)

    # Fill missing required columns
    defaults = {
        "Ticket Subject": "No subject",
        "Ticket Description": "No description",
        "Ticket Priority": "Medium",
        "Ticket Channel": "email",
        "Resolution Time": 24.0,
        "Ticket Type": "General",
        "Ticket ID": range(len(df)),
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Normalise priority casing
    df["Ticket Priority"] = df["Ticket Priority"].str.strip().str.title()
    valid_priorities = ["Low", "Medium", "High", "Critical"]
    df["Ticket Priority"] = df["Ticket Priority"].apply(
        lambda x: x if x in valid_priorities else "Medium"
    )

    # Convert resolution time
    df["Resolution Time"] = pd.to_numeric(df["Resolution Time"], errors="coerce").fillna(24.0)

    return df


def _generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic CRM ticket data."""
    rng = np.random.default_rng(seed)

    subjects = [
        "Cannot log in", "Payment failed", "App crashes on startup",
        "Feature request: dark mode", "Slow loading times", "Data not syncing",
        "Account hacked - urgent", "Billing overcharged",
        "Minor typo on dashboard", "How to export data?",
        "Critical security vulnerability found", "API returning 500 errors",
        "Lost all my data after update", "Question about subscription",
        "Service completely down", "Unable to process transactions",
    ]
    descriptions_high = [
        "I cannot access my account and it's been 3 days. My team cannot work. This is CRITICAL and unacceptable. I need this fixed ASAP or I will cancel my subscription and seek legal action.",
        "Our payment system is completely down. We are losing thousands of dollars per hour. This is an emergency. All transactions are failing. Please escalate immediately.",
        "We discovered a security breach. Customer data may be exposed. This requires immediate attention. Our legal team has been notified.",
        "The system crashed and we lost all data from the past week. This is a disaster for our operations. We need immediate data recovery.",
    ]
    descriptions_low = [
        "Just wondering if there's a way to export my data to CSV. Not urgent at all, just a feature that would be nice to have.",
        "I noticed there's a small typo on the settings page. The word 'recieve' should be 'receive'. Minor cosmetic issue.",
        "Would love to see a dark mode option in a future update. Just a suggestion, nothing urgent.",
        "How do I change my notification preferences? This is not urgent, just curious.",
    ]

    rows = []
    priorities = rng.choice(["Low", "Medium", "High", "Critical"],
                              size=n, p=[0.3, 0.4, 0.2, 0.1])
    channels = rng.choice(["email", "chat", "phone", "social media", "web"],
                            size=n, p=[0.4, 0.25, 0.15, 0.1, 0.1])
    ticket_types = rng.choice(
        ["Technical Issue", "Billing", "Account", "Feature Request", "General"],
        size=n, p=[0.3, 0.2, 0.2, 0.15, 0.15],
    )

    for i in range(n):
        priority = priorities[i]
        # Introduce ~25% mismatch: high severity tickets get low labels and vice versa
        use_high_desc = rng.random() < 0.5

        if priority in ["Low", "Medium"] and rng.random() < 0.20:
            # Hidden crisis: low-label but high-severity description
            desc = rng.choice(descriptions_high)
            subj = rng.choice(subjects[:8])
            rt = rng.uniform(48, 120)
        elif priority in ["High", "Critical"] and rng.random() < 0.20:
            # False alarm: high-label but low-severity description
            desc = rng.choice(descriptions_low)
            subj = rng.choice(subjects[8:])
            rt = rng.uniform(1, 8)
        else:
            if priority in ["High", "Critical"]:
                desc = rng.choice(descriptions_high)
                subj = rng.choice(subjects[:8])
                rt = rng.uniform(24, 96)
            else:
                desc = rng.choice(descriptions_low)
                subj = rng.choice(subjects[8:])
                rt = rng.uniform(2, 24)

        rows.append({
            "Ticket ID": f"TKT-{i+1:05d}",
            "Ticket Subject": subj,
            "Ticket Description": desc,
            "Ticket Priority": priority,
            "Ticket Channel": channels[i],
            "Resolution Time": round(rt, 1),
            "Ticket Type": ticket_types[i],
        })

    df = pd.DataFrame(rows)
    print(f"[Data] Synthetic dataset generated: {len(df)} tickets")
    print(df["Ticket Priority"].value_counts().to_string())
    return df
