"""
predict.py
==========
Inference script: accepts a CSV, outputs predictions + dossiers.
Usage:
    python predict.py --input data/new_tickets.csv --output outputs/results.csv
"""

import argparse
import json
import os
import pandas as pd

from src.data_loader import load_dataset, _standardise_columns
from src.pseudo_labeler import generate_pseudo_labels
from src.dossier import generate_all_dossiers


def predict_with_pseudolabels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast inference path: pseudo-labeling only (no model needed).
    Good for quick demos or when model is not trained yet.
    """
    return generate_pseudo_labels(df)


def predict_with_model(df: pd.DataFrame, model_dir: str = "outputs/model") -> pd.DataFrame:
    """Full inference with fine-tuned DeBERTa model."""
    from src.classifier import load_model, predict_batch
    try:
        model, tokenizer, scaler, le_type = load_model(model_dir)
        df = generate_pseudo_labels(df)
        df = predict_batch(df, model, tokenizer, scaler, le_type)
        return df
    except FileNotFoundError:
        print(f"[Predict] No trained model found at {model_dir}. Using pseudo-labels only.")
        return predict_with_pseudolabels(df)


def main(args):
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Load input
    df = pd.read_csv(args.input)
    df = _standardise_columns(df)

    print(f"[Predict] Processing {len(df)} tickets …")

    # Predict
    if args.use_model:
        df_result = predict_with_model(df, model_dir=args.model_dir)
    else:
        df_result = predict_with_pseudolabels(df)

    # Generate dossiers
    dossier_path = args.output.replace(".csv", "_dossiers.json")
    dossiers = generate_all_dossiers(df_result, output_path=dossier_path)

    # Save predictions CSV
    df_result.to_csv(args.output, index=False)
    print(f"[Predict] Predictions saved → {args.output}")
    print(f"[Predict] Dossiers saved → {dossier_path}")

    # Summary
    mismatch_col = "pred_mismatch" if "pred_mismatch" in df_result.columns else "mismatch_label"
    n_mismatch = df_result[mismatch_col].sum()
    print(f"\n[Summary] {n_mismatch}/{len(df_result)} tickets flagged as mismatch")
    if "mismatch_type" in df_result.columns:
        print(df_result["mismatch_type"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Inference")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="outputs/predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--use-model", action="store_true", default=False,
                        help="Use fine-tuned DeBERTa model (if trained)")
    parser.add_argument("--model-dir", default="outputs/model",
                        help="Path to saved model directory")
    args = parser.parse_args()
    main(args)
