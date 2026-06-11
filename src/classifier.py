"""
Stage 2: Fine-Tuned Classifier (DeBERTa-v3-small)
==================================================
Trains a binary mismatch classifier on pseudo-labeled data.
  - Model: microsoft/deberta-v3-small (fine-tuned, not frozen)
  - Inputs: text (subject + description) + structured metadata
            (channel, resolution_time, ticket_type)
  - Imbalance: weighted cross-entropy loss
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metadata feature columns
CHANNEL_VALUES = ["email", "chat", "phone", "social media", "web"]


# ── Feature helpers ───────────────────────────────────────────────────────────
def encode_metadata(df: pd.DataFrame, scaler: StandardScaler = None,
                     le_channel: LabelEncoder = None, le_type: LabelEncoder = None,
                     fit: bool = True):
    """Returns (meta_array, scaler, le_channel, le_type)."""
    meta = pd.DataFrame()

    # Channel one-hot (known set)
    channel_raw = df.get("Ticket Channel", pd.Series(["email"] * len(df)))
    channel_clean = channel_raw.str.lower().str.strip().fillna("email")

    for v in CHANNEL_VALUES:
        meta[f"ch_{v.replace(' ', '_')}"] = (channel_clean == v).astype(float)

    # Resolution time (normalised)
    rt = pd.to_numeric(df.get("Resolution Time", pd.Series([24.0] * len(df))),
                        errors="coerce").fillna(24.0)
    meta["log_rt"] = np.log1p(rt)

    # Ticket type label-encoded
    tt = df.get("Ticket Type", pd.Series(["General"] * len(df))).fillna("General").str.strip()
    if fit:
        le_type = LabelEncoder()
        meta["ticket_type_enc"] = le_type.fit_transform(tt).astype(float)
    else:
        tt_safe = tt.apply(lambda x: x if x in le_type.classes_ else le_type.classes_[0])
        meta["ticket_type_enc"] = le_type.transform(tt_safe).astype(float)

    arr = meta.values.astype(np.float32)

    if fit:
        scaler = StandardScaler()
        arr = scaler.fit_transform(arr)
    else:
        arr = scaler.transform(arr)

    return arr, scaler, le_channel, le_type


# ── Dataset ───────────────────────────────────────────────────────────────────
class TicketDataset(Dataset):
    def __init__(self, texts, meta_feats, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.meta = meta_feats
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "meta": torch.tensor(self.meta[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────
class SIAClassifier(nn.Module):
    def __init__(self, n_meta_features: int, hidden_size: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)

        # Meta MLP
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Classification head
        self.classifier = nn.Linear(hidden_size + 64, 2)

    def forward(self, input_ids, attention_mask, meta):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])  # [CLS] token
        meta_out = self.meta_proj(meta)
        combined = torch.cat([cls, meta_out], dim=-1)
        return self.classifier(combined)


# ── Training ──────────────────────────────────────────────────────────────────
def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_model(df: pd.DataFrame, save_dir: str = "outputs/model"):
    os.makedirs(save_dir, exist_ok=True)

    print(f"[Stage 2] Training on {DEVICE}")

    # Build text
    texts = (df["Ticket Subject"].fillna("") + " [SEP] " +
             df["Ticket Description"].fillna("")).tolist()
    labels = df["mismatch_label"].values

    # Encode metadata
    meta_arr, scaler, _, le_type = encode_metadata(df, fit=True)
    n_meta = meta_arr.shape[1]

    # Save encoders
    import pickle
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, "le_type.pkl"), "wb") as f:
        pickle.dump(le_type, f)

    # Train / val split (stratified)
    X_tr, X_val, m_tr, m_val, y_tr, y_val = train_test_split(
        texts, meta_arr, labels, test_size=0.15, stratify=labels, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(save_dir)

    tr_ds = TicketDataset(X_tr, m_tr, y_tr, tokenizer)
    val_ds = TicketDataset(X_val, m_val, y_val, tokenizer)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SIAClassifier(n_meta_features=n_meta).to(DEVICE)
    class_weights = compute_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(tr_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=total_steps // 10,
                                                 num_training_steps=total_steps)
    scaler_amp = GradScaler() if DEVICE.type == "cuda" else None

    best_f1 = 0.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in tr_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            meta_b = batch["meta"].to(DEVICE)
            lbl = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            if scaler_amp:
                with autocast():
                    logits = model(ids, mask, meta_b)
                    loss = criterion(logits, lbl)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(ids, mask, meta_b)
                loss = criterion(logits, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                meta_b = batch["meta"].to(DEVICE)
                logits = model(ids, mask, meta_b)
                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_all.extend(preds)
                labels_all.extend(batch["label"].numpy())

        acc = accuracy_score(labels_all, preds_all)
        f1 = f1_score(labels_all, preds_all, average="macro")
        rec = recall_score(labels_all, preds_all, average=None, zero_division=0)
        avg_loss = total_loss / len(tr_loader)

        print(f"  Epoch {epoch}/{EPOCHS} | loss={avg_loss:.4f} | "
              f"acc={acc:.4f} | macro-F1={f1:.4f} | recall={rec}")

        epoch_info = {"epoch": epoch, "loss": avg_loss, "accuracy": acc,
                      "macro_f1": f1, "recall_consistent": float(rec[0]),
                      "recall_mismatch": float(rec[1])}
        history.append(epoch_info)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"  ✓ New best model saved (F1={best_f1:.4f})")

    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final report on val set
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"),
                                      map_location=DEVICE))
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            meta_b = batch["meta"].to(DEVICE)
            logits = model(ids, mask, meta_b)
            preds = logits.argmax(dim=-1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(batch["label"].numpy())

    print("\n[Stage 2] Final Validation Report:")
    print(classification_report(labels_all, preds_all,
                                  target_names=["Consistent", "Mismatch"]))

    # Save meta info
    meta_cfg = {"n_meta_features": n_meta}
    with open(os.path.join(save_dir, "meta_cfg.json"), "w") as f:
        json.dump(meta_cfg, f)

    return model, tokenizer, scaler, le_type, history


# ── Inference ─────────────────────────────────────────────────────────────────
def load_model(save_dir: str = "outputs/model"):
    import pickle
    with open(os.path.join(save_dir, "meta_cfg.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(save_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(save_dir, "le_type.pkl"), "rb") as f:
        le_type = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = SIAClassifier(n_meta_features=cfg["n_meta_features"])
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"),
                                      map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, tokenizer, scaler, le_type


def predict_batch(df: pd.DataFrame, model, tokenizer, scaler, le_type) -> pd.DataFrame:
    texts = (df["Ticket Subject"].fillna("") + " [SEP] " +
             df["Ticket Description"].fillna("")).tolist()
    meta_arr, _, _, _ = encode_metadata(df, scaler=scaler, le_type=le_type, fit=False)

    ds = TicketDataset(texts, meta_arr, [0] * len(texts), tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    all_preds, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            meta_b = batch["meta"].to(DEVICE)
            logits = model(ids, mask, meta_b)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])  # mismatch probability

    result = df.copy()
    result["pred_mismatch"] = all_preds
    result["mismatch_confidence"] = all_probs
    return result
