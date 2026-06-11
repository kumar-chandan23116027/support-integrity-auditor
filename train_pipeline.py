"""
train_pipeline.py
=================
Full SIA pipeline: pseudo-labeling → training → evaluation.
Usage:
    python train_pipeline.py --data data/support_tickets.csv
    python train_pipeline.py   # uses synthetic data
"""

import argparse
import os
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.pseudo_labeler import generate_pseudo_labels, signal_agreement
from src.classifier import train_model
from src.dossier import generate_all_dossiers
from src.evaluation import evaluate, ablation_report


def main(args):
    os.makedirs("outputs", exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_dataset(args.data)

    # ── 2. Pseudo-label generation ────────────────────────────────────────────
    df = generate_pseudo_labels(df)
    df.to_csv("outputs/pseudo_labeled.csv", index=False)
    print(f"[Pipeline] Pseudo-labeled data saved → outputs/pseudo_labeled.csv")

    # Signal agreement metric
    sa = signal_agreement(df)
    print(f"[Pipeline] Signal Agreement (NLP vs RT): {sa:.4f}")

    # Ablation study
    ablation_report(df)

    # ── 3. Train classifier ───────────────────────────────────────────────────
    if args.train:
        model, tokenizer, scaler, le_type, history = train_model(
            df, save_dir="outputs/model"
        )

        # Inference on full dataset for final eval
        from src.classifier import predict_batch
        df_pred = predict_batch(df, model, tokenizer, scaler, le_type)

        # Evaluation
        y_true = df_pred["mismatch_label"].values
        y_pred = df_pred["pred_mismatch"].values
        metrics = evaluate(y_true, y_pred, signal_agreement=sa)

        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # ── 4. Dossier generation ─────────────────────────────────────────────
        dossiers = generate_all_dossiers(df_pred, output_path="outputs/dossiers.json")
        print(f"[Pipeline] Generated {len(dossiers)} dossiers.")

        # Save predictions
        df_pred.to_csv("outputs/predictions.csv", index=False)
    else:
        # Pseudo-label only mode (no GPU training)
        metrics = {}
        dossiers = generate_all_dossiers(df, output_path="outputs/dossiers.json")
        print(f"[Pipeline] Generated {len(dossiers)} dossiers (no classifier trained).")

    print("\n[Pipeline] Complete. Artifacts in outputs/")
    return df, metrics, dossiers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Training Pipeline")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to CSV dataset (Kaggle CRM)")
    parser.add_argument("--train", action="store_true", default=True,
                        help="Run DeBERTa fine-tuning (requires GPU for speed)")
    parser.add_argument("--no-train", dest="train", action="store_false",
                        help="Skip fine-tuning, pseudo-label only")
    args = parser.parse_args()
    main(args)
