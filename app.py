"""
Streamlit App — Support Integrity Auditor (SIA)
================================================
Features:
  - Single ticket form input
  - Batch CSV upload
  - Priority Mismatch Dashboard (distribution, types, signals)
  - Severity delta heatmap across categories × channels
  - Full Evidence Dossier viewer
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src is on path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import _standardise_columns
from src.pseudo_labeler import generate_pseudo_labels, signal_agreement
from src.dossier import generate_dossier, generate_all_dossiers


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px 20px;
    border-left: 4px solid #e63946;
    margin-bottom: 12px;
}
.mismatch-badge {
    background: #e63946;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
}
.consistent-badge {
    background: #2a9d8f;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
}
.dossier-box {
    background: #1a1a2e;
    color: #e0e0e0;
    padding: 20px;
    border-radius: 8px;
    font-family: monospace;
    font-size: 13px;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    return generate_pseudo_labels(df)


def predict_single(subject, description, priority, channel, resolution_time, ticket_type):
    row = {
        "Ticket ID": "SINGLE-001",
        "Ticket Subject": subject,
        "Ticket Description": description,
        "Ticket Priority": priority,
        "Ticket Channel": channel,
        "Resolution Time": resolution_time,
        "Ticket Type": ticket_type,
    }
    df = pd.DataFrame([row])
    df = run_pipeline(df)
    return df.iloc[0].to_dict()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/detective.png", width=60)
    st.title("SIA")
    st.caption("Support Integrity Auditor")
    st.markdown("---")
    mode = st.radio("Mode", ["📋 Single Ticket", "📦 Batch CSV", "📊 Dashboard"])
    st.markdown("---")
    st.markdown("""
**Pipeline:**
1. Pseudo-label generation
2. DeBERTa-v3-small fine-tuning
3. Evidence Dossier generation

**Signals:**
- NLP (keyword + negation)
- Resolution Time regression
""")


# ── Single Ticket Mode ────────────────────────────────────────────────────────
if mode == "📋 Single Ticket":
    st.title("🔍 Single Ticket Audit")
    st.markdown("Enter a support ticket to detect priority mismatch.")

    col1, col2 = st.columns([2, 1])

    with col1:
        subject = st.text_input("Ticket Subject",
                                 value="Cannot access my account - urgent!")
        description = st.text_area("Ticket Description",
            value="I have been unable to log in for 3 days. My entire team is blocked. "
                  "We are losing revenue every hour. This is CRITICAL. "
                  "Please escalate immediately or I will involve legal.",
            height=120)

    with col2:
        priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])
        channel = st.selectbox("Ticket Channel", ["email", "chat", "phone", "social media", "web"])
        resolution_time = st.number_input("Resolution Time (hours)", min_value=0.0,
                                           max_value=720.0, value=48.0, step=1.0)
        ticket_type = st.selectbox("Ticket Type",
                                    ["Technical Issue", "Billing", "Account",
                                     "Feature Request", "General"])

    if st.button("🔎 Audit Ticket", type="primary", use_container_width=True):
        with st.spinner("Analysing ticket …"):
            result = predict_single(subject, description, priority, channel,
                                     resolution_time, ticket_type)

        is_mismatch = result.get("mismatch_label", 0) == 1
        mismatch_type = result.get("mismatch_type", "Consistent")
        inferred = result.get("inferred_severity", "Unknown")
        fused_score = result.get("fused_score", 0.0)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Assigned Priority", priority)
        c2.metric("Inferred Severity", inferred)
        c3.metric("Severity Delta", f"{result.get('severity_delta', 0):+d}")
        c4.metric("Fused Score", f"{fused_score:.3f}")

        if is_mismatch:
            color = "#e63946" if mismatch_type == "Hidden Crisis" else "#f4a261"
            st.markdown(
                f"<div class='metric-card'>"
                f"<b>⚠️ MISMATCH DETECTED</b> &nbsp;"
                f"<span class='mismatch-badge'>{mismatch_type}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='metric-card' style='border-left-color:#2a9d8f'>"
                "<b>✅ CONSISTENT</b> &nbsp;"
                "<span class='consistent-badge'>No Mismatch</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Dossier
        if is_mismatch:
            st.subheader("📄 Evidence Dossier")
            dossier = generate_dossier(result)

            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**🎫 Ticket ID:** `{dossier['ticket_id']}`")
                st.markdown(f"**🏷️ Assigned Priority:** `{dossier['assigned_priority']}`")
                st.markdown(f"**🔍 Inferred Severity:** `{dossier['inferred_severity']}`")
            with d2:
                st.markdown(f"**⚡ Mismatch Type:** `{dossier['mismatch_type']}`")
                st.markdown(f"**📊 Severity Delta:** `{dossier['severity_delta']:+d}`")
                st.markdown(f"**🎯 Confidence:** `{dossier['confidence']:.2%}`")

            st.markdown("**🔎 Constraint Analysis:**")
            st.info(dossier["constraint_analysis"])

            st.markdown("**📌 Feature Evidence:**")
            for ev in dossier["feature_evidence"]:
                signal = ev.get("signal", "").replace("_", " ").title()
                value = ev.get("value", "")
                interp = ev.get("interpretation", "")
                weight = ev.get("weight", "")
                source = ev.get("source_field", "")
                with st.expander(f"🔹 {signal}  —  source: `{source}`"):
                    if isinstance(value, list):
                        st.markdown(f"**Keywords found:** {', '.join(value)}")
                    else:
                        st.markdown(f"**Value:** {value}")
                    if interp:
                        st.markdown(f"**Interpretation:** {interp}")
                    if weight != "":
                        st.markdown(f"**Weight:** {weight}")

            st.download_button(
                "⬇️ Download Dossier (JSON)",
                data=json.dumps(dossier, indent=2),
                file_name=f"dossier_{result.get('Ticket ID', 'ticket')}.json",
                mime="application/json",
            )

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fused_score,
            title={"text": "Inferred Severity Score"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#e63946"},
                "steps": [
                    {"range": [0, 0.25], "color": "#d4edda"},
                    {"range": [0.25, 0.5], "color": "#fff3cd"},
                    {"range": [0.5, 0.75], "color": "#fde8d8"},
                    {"range": [0.75, 1.0], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "red", "width": 4},
                               "thickness": 0.75, "value": 0.5},
            },
        ))
        fig.update_layout(height=250, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)


# ── Batch CSV Mode ────────────────────────────────────────────────────────────
elif mode == "📦 Batch CSV":
    st.title("📦 Batch Ticket Audit")
    st.markdown("Upload a CSV with ticket data to audit all tickets at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df_raw = _standardise_columns(df_raw)
        st.info(f"Loaded {len(df_raw)} tickets.")

        if st.button("🔎 Run Audit", type="primary"):
            with st.spinner(f"Auditing {len(df_raw)} tickets …"):
                df_result = run_pipeline(df_raw)

            n_mismatch = df_result["mismatch_label"].sum()
            sa = signal_agreement(df_result)

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tickets", len(df_result))
            c2.metric("Mismatches", int(n_mismatch),
                      delta=f"{100*n_mismatch/len(df_result):.1f}%")
            c3.metric("Hidden Crises",
                      int((df_result["mismatch_type"] == "Hidden Crisis").sum()))
            c4.metric("Signal Agreement", f"{sa:.2%}")

            # Results table
            st.subheader("Results")
            display_cols = ["Ticket ID", "Ticket Subject", "Ticket Priority",
                            "inferred_severity", "mismatch_type",
                            "severity_delta", "fused_score"]
            display_cols = [c for c in display_cols if c in df_result.columns]

            def color_row(row):
                if row.get("mismatch_type") == "Hidden Crisis":
                    return ["background-color: #fde8d8"] * len(row)
                elif row.get("mismatch_type") == "False Alarm":
                    return ["background-color: #fff3cd"] * len(row)
                return [""] * len(row)

            styled = df_result[display_cols].style.apply(color_row, axis=1)
            st.dataframe(styled, use_container_width=True, height=400)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                csv_out = df_result.to_csv(index=False).encode()
                st.download_button("⬇️ Download Predictions CSV", csv_out,
                                    "predictions.csv", "text/csv")
            with col2:
                dossiers = generate_all_dossiers(df_result)
                st.download_button("⬇️ Download Dossiers JSON",
                                    json.dumps(dossiers, indent=2),
                                    "dossiers.json", "application/json")
    else:
        st.markdown("---")
        st.subheader("Sample CSV format")
        sample = pd.DataFrame([{
            "Ticket Subject": "Cannot login",
            "Ticket Description": "Account locked for 2 days, urgent!",
            "Ticket Priority": "Low",
            "Ticket Channel": "email",
            "Resolution Time": 72,
            "Ticket Type": "Account",
        }])
        st.dataframe(sample, use_container_width=True)
        st.download_button("⬇️ Download Sample CSV",
                            sample.to_csv(index=False),
                            "sample_tickets.csv", "text/csv")


# ── Dashboard Mode ────────────────────────────────────────────────────────────
elif mode == "📊 Dashboard":
    st.title("📊 Priority Mismatch Dashboard")

    # Load data
    if os.path.exists("outputs/pseudo_labeled.csv"):
        df = pd.read_csv("outputs/pseudo_labeled.csv")
        st.success(f"Loaded {len(df)} tickets from pipeline output.")
    else:
        st.info("No pipeline output found. Generating demo data …")
        from src.data_loader import load_dataset
        df = load_dataset()
        df = run_pipeline(df)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    n = len(df)
    n_mismatch = df["mismatch_label"].sum()
    n_hidden = (df["mismatch_type"] == "Hidden Crisis").sum()
    n_false = (df["mismatch_type"] == "False Alarm").sum()
    sa = signal_agreement(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Tickets", n)
    c2.metric("Mismatch Rate", f"{100*n_mismatch/n:.1f}%")
    c3.metric("Hidden Crises 🚨", int(n_hidden))
    c4.metric("False Alarms ⚠️", int(n_false))
    c5.metric("Signal Agreement", f"{sa:.2%}")

    st.markdown("---")

    # ── Row 1 ─────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Mismatch Distribution")
        type_counts = df["mismatch_type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig = px.pie(type_counts, values="Count", names="Type",
                      color="Type",
                      color_discrete_map={
                          "Hidden Crisis": "#e63946",
                          "False Alarm": "#f4a261",
                          "Consistent": "#2a9d8f",
                      },
                      hole=0.45)
        fig.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Assigned vs Inferred Priority")
        cross = pd.crosstab(df["Ticket Priority"], df["inferred_severity"])
        priority_order = ["Low", "Medium", "High", "Critical"]
        cross = cross.reindex(index=[p for p in priority_order if p in cross.index],
                               columns=[p for p in priority_order if p in cross.columns])
        fig = px.imshow(cross, text_auto=True, aspect="auto",
                         color_continuous_scale="RdYlGn_r",
                         labels=dict(x="Inferred Severity", y="Assigned Priority",
                                      color="Count"))
        fig.update_layout(margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("Fused Score Distribution")
        fig = px.histogram(df, x="fused_score", color="mismatch_type",
                            nbins=30,
                            color_discrete_map={
                                "Hidden Crisis": "#e63946",
                                "False Alarm": "#f4a261",
                                "Consistent": "#2a9d8f",
                            },
                            labels={"fused_score": "Fused Severity Score"})
        fig.add_vline(x=0.5, line_dash="dash", line_color="black",
                       annotation_text="Threshold")
        fig.update_layout(margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── Severity Delta Heatmap ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌡️ Severity Delta Heatmap — Ticket Type × Channel")

    if "Ticket Type" in df.columns and "Ticket Channel" in df.columns:
        heatmap_data = df.groupby(["Ticket Type", "Ticket Channel"])["severity_delta"].mean().reset_index()
        pivot = heatmap_data.pivot(index="Ticket Type", columns="Ticket Channel",
                                    values="severity_delta").fillna(0)
        fig = px.imshow(pivot, text_auto=".1f",
                         color_continuous_scale="RdBu_r",
                         color_continuous_midpoint=0,
                         labels=dict(color="Avg Severity Delta"),
                         aspect="auto")
        fig.update_layout(margin=dict(t=20, b=20), height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Positive delta = ticket is more severe than assigned. Negative = ticket is over-assigned.")

    # ── Top Signals ───────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("NLP vs RT Signal Scores")
        sample = df.sample(min(300, len(df)), random_state=42)
        fig = px.scatter(sample, x="nlp_score", y="rt_score",
                          color="mismatch_type",
                          color_discrete_map={
                              "Hidden Crisis": "#e63946",
                              "False Alarm": "#f4a261",
                              "Consistent": "#2a9d8f",
                          },
                          opacity=0.6,
                          labels={"nlp_score": "NLP Score",
                                   "rt_score": "Resolution Time Score"},
                          hover_data=["Ticket Priority", "inferred_severity"])
        fig.update_layout(margin=dict(t=20, b=20), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Mismatch by Assigned Priority")
        priority_mismatch = df.groupby("Ticket Priority")["mismatch_label"].mean().reset_index()
        priority_mismatch.columns = ["Priority", "Mismatch Rate"]
        priority_order = ["Low", "Medium", "High", "Critical"]
        priority_mismatch["Priority"] = pd.Categorical(
            priority_mismatch["Priority"], categories=priority_order, ordered=True)
        priority_mismatch = priority_mismatch.sort_values("Priority")
        fig = px.bar(priority_mismatch, x="Priority", y="Mismatch Rate",
                      color="Mismatch Rate", color_continuous_scale="Reds",
                      labels={"Mismatch Rate": "Mismatch Rate (%)"})
        fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        fig.update_layout(margin=dict(t=20, b=20), height=320,
                           yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # ── Dossier viewer ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📄 Evidence Dossier Viewer")
    mismatch_df = df[df["mismatch_label"] == 1].reset_index(drop=True)
    if len(mismatch_df) > 0:
        idx = st.selectbox("Select mismatch ticket",
                            range(len(mismatch_df)),
                            format_func=lambda i: (
                                f"[{mismatch_df.iloc[i].get('mismatch_type', '')}] "
                                f"{mismatch_df.iloc[i].get('Ticket Subject', '')[:60]}"
                            ))
        dossier = generate_dossier(mismatch_df.iloc[idx].to_dict())

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**🎫 Ticket ID:** `{dossier['ticket_id']}`")
            st.markdown(f"**🏷️ Assigned Priority:** `{dossier['assigned_priority']}`")
            st.markdown(f"**🔍 Inferred Severity:** `{dossier['inferred_severity']}`")
        with d2:
            st.markdown(f"**⚡ Mismatch Type:** `{dossier['mismatch_type']}`")
            st.markdown(f"**📊 Severity Delta:** `{dossier['severity_delta']:+d}`")
            st.markdown(f"**🎯 Confidence:** `{dossier['confidence']:.2%}`")

        st.markdown("**🔎 Constraint Analysis:**")
        st.info(dossier["constraint_analysis"])

        st.markdown("**📌 Feature Evidence:**")
        for ev in dossier["feature_evidence"]:
            signal = ev.get("signal", "").replace("_", " ").title()
            value = ev.get("value", "")
            interp = ev.get("interpretation", "")
            weight = ev.get("weight", "")
            source = ev.get("source_field", "")
            with st.expander(f"🔹 {signal}  —  source: `{source}`"):
                if isinstance(value, list):
                    st.markdown(f"**Keywords found:** {', '.join(value)}")
                else:
                    st.markdown(f"**Value:** {value}")
                if interp:
                    st.markdown(f"**Interpretation:** {interp}")
                if weight != "":
                    st.markdown(f"**Weight:** {weight}")

        st.download_button(
            "⬇️ Download Dossier (JSON)",
            data=json.dumps(dossier, indent=2),
            file_name=f"dossier_{dossier['ticket_id']}.json",
            mime="application/json",
        )
    else:
        st.info("No mismatch tickets found.")
