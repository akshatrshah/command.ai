#!/usr/bin/env python3
"""
Command.ai — Fine-tune DistilBERT on your commands.csv
Run inside your venv: source ~/.command_ai/venv/bin/activate && python3 train.py

CSV format (your format, no changes needed):
  command,description,privilege_level
  privilege_level: 0=LOW(safe), 1=MEDIUM(normal ops), 2=HIGH(destructive)

Output: ~/.command_ai/model/  (loaded automatically by ml_engine.py)
"""

import csv, json, sys
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE  = Path("commands.csv")
MODEL_OUT  = Path.home() / ".command_ai" / "model"
BASE_MODEL = "distilbert-base-uncased"
EPOCHS     = 6        # best eval_loss was at epoch 3 — 6 gives headroom with early stopping
BATCH_SIZE = 16
MAX_LEN    = 96
PATIENCE   = 2        # stop if eval_loss doesn't improve for 2 epochs

LABEL_NAMES = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# ── Parse args ────────────────────────────────────────────────────────────────
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--data" and i + 1 < len(sys.argv) - 1:
        DATA_FILE = Path(sys.argv[i + 2])

if not DATA_FILE.exists():
    print(f"❌  CSV not found: {DATA_FILE}")
    print(f"    Usage: python3 train.py --data path/to/commands.csv")
    sys.exit(1)

# ── Load CSV ──────────────────────────────────────────────────────────────────
texts, labels = [], []
skipped = 0

with open(DATA_FILE, encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) < 3:
            skipped += 1
            continue
        cmd, desc, lvl = row[0].strip(), row[1].strip(), row[2].strip()
        if cmd.lower() == "command":
            continue
        try:
            lvl = int(lvl)
        except ValueError:
            skipped += 1
            continue
        if lvl not in (0, 1, 2):
            skipped += 1
            continue
        texts.append(f"{cmd}: {desc}")
        labels.append(lvl)

if not texts:
    print("❌  No valid rows found in CSV. Check format: command,description,privilege_level")
    sys.exit(1)

dist = Counter(labels)
print(f"✅  Loaded {len(texts)} examples  (skipped {skipped})")
print(f"    Distribution: LOW={dist[0]}  MEDIUM={dist[1]}  HIGH={dist[2]}")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        EarlyStoppingCallback,
        Trainer, TrainingArguments,
    )
except ImportError as e:
    print(f"❌  Missing dependency: {e}")
    print("    Run: pip install transformers torch scikit-learn accelerate")
    sys.exit(1)

# ── Dataset ───────────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.15, random_state=42,
    stratify=labels if min(dist.values()) >= 2 else None
)

class CmdDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc    = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item

train_ds = CmdDataset(train_texts, train_labels)
val_ds   = CmdDataset(val_texts,   val_labels)
print(f"    Train: {len(train_ds)}  Val: {len(val_ds)}")

# ── Model — increase dropout to fight overfitting ─────────────────────────────
from transformers import DistilBertConfig
config = DistilBertConfig.from_pretrained(
    BASE_MODEL,
    num_labels   = 3,
    dropout      = 0.3,   # default is 0.1 — higher = less overfitting
    seq_classif_dropout = 0.3,
)
model = DistilBertForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

# ── Training args ─────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir                  = str(MODEL_OUT / "checkpoints"),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    logging_steps               = 10,
    warmup_steps                = 50,       # replaces deprecated warmup_ratio
    weight_decay                = 0.05,     # slightly stronger regularization
    use_cpu                     = not torch.cuda.is_available(),
    report_to                   = "none",
)

trainer = Trainer(
    model          = model,
    args           = args,
    train_dataset  = train_ds,
    eval_dataset   = val_ds,
    callbacks      = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

print(f"\n🚀  Training on {'GPU' if torch.cuda.is_available() else 'CPU'}...")
print(f"    Early stopping patience: {PATIENCE} epochs\n")
trainer.train()

# ── Save best model ───────────────────────────────────────────────────────────
MODEL_OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MODEL_OUT)
tokenizer.save_pretrained(MODEL_OUT)
json.dump(LABEL_NAMES, open(MODEL_OUT / "label_map.json", "w"))
print(f"\n✅  Model saved to {MODEL_OUT}")

# ── Validation report ─────────────────────────────────────────────────────────
print("\n── Validation Report ────────────────────────────────")
model.eval()
preds = []
for item in val_ds:
    with torch.no_grad():
        logits = model(
            input_ids      = item["input_ids"].unsqueeze(0),
            attention_mask = item["attention_mask"].unsqueeze(0)
        ).logits
    preds.append(logits.argmax().item())

print(classification_report(
    val_labels, preds,
    target_names=["LOW (0)", "MEDIUM (1)", "HIGH (2)"],
    zero_division=0
))