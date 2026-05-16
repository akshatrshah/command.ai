#!/usr/bin/env python3
"""
Command.ai ML Engine - Sidecar for risk assessment and session summarization.

CSV privilege levels map directly to risk:
  0 = LOW  (safe, read-only)
  1 = MEDIUM (normal ops with side effects)
  2 = HIGH (destructive, irreversible)

Loads fine-tuned DistilBERT from ~/.command_ai/model/ for risk assessment.
Uses facebook/bart-large-cnn for session summarization (lazy-loaded on first :summary call).

Protocol (stdin/stdout JSON lines):
  Input:  {"action": "assess"|"summarize", "data": "..."}
  Output: {"result": "...", "confidence": 0.0-1.0}
"""

import sys, json, re, os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DIR      = Path.home() / ".command_ai" / "model"
LABEL_MAP      = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
FALLBACK_MODEL = "typeform/distilbert-base-uncased-mnli"
ZS_LABELS      = ["safe read-only command", "normal operation with side effects", "destructive irreversible command"]
ZS_RISK_MAP    = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# ── Regex pre-screen ──────────────────────────────────────────────────────────
HIGH_RE = [
    re.compile(r, re.I) for r in [
        r"rm\s+-[^\s]*r[^\s]*f", r"\bdd\b.*of=/dev/", r"\bmkfs\b",
        r"git\s+(push\s+--force|push\s+-f)", r"git\s+filter-branch",
        r"git\s+reflog\s+expire.*--expire=now",
        r":\(\)\s*\{.*:\s*\|.*:\s*\}", r"\bvisudo\b",
        r"\bshutdown\b", r"\breboot\b", r"\bhalt\b",
        r"iptables\s+-F", r"/etc/shadow",
        r"git\s+reset\s+--hard\s+HEAD[~^]",
        r"git\s+clean\s+-[^\s]*f",
        r"sudo\s+passwd",
        r"curl\s+https?://[^\s]*\s*\|\s*sh",
        r"wget\s+[^\s]*\s*\|\s*sh",
    ]
]
MEDIUM_RE = [
    re.compile(r, re.I) for r in [
        r"\bsudo\b", r"\bchmod\b", r"\bchown\b",
        r"\bkill\b", r"\bpkill\b",
        r"\bapt\b", r"\byum\b", r"\bdnf\b",
        r"\bsystemctl\b", r"\bcrontab\b",
        r"curl\s+.*\|\s*sh", r"wget\s+.*\|\s*sh",
        r"\bscp\b", r"\brsync\b",
        r"git\s+(push|merge|rebase|cherry-pick|reset\s+--hard\s+HEAD\b)",
        r"git\s+(tag\s+-d|remote\s+remove|stash\s+drop|rm\s+-r)",
        r"git\s+commit\s+--amend",
        r"git\s+push\s+origin\s+--delete",
        r"git\s+submodule",
    ]
]

def rule_based_risk(command: str):
    for p in HIGH_RE:
        if p.search(command):
            return "HIGH"
    for p in MEDIUM_RE:
        if p.search(command):
            return "MEDIUM"
    return None

# ── Risk model loading ────────────────────────────────────────────────────────

def load_finetuned():
    if not MODEL_DIR.exists():
        return None, None
    try:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        import torch
        tok   = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
        model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.eval()
        return tok, model
    except Exception as e:
        print(json.dumps({"warning": f"Could not load fine-tuned model: {e}"}), flush=True)
        return None, None

def load_zeroshot():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=FALLBACK_MODEL, device=-1)

# ── Risk inference ────────────────────────────────────────────────────────────

def assess_finetuned(command: str, tok, model) -> dict:
    import torch
    override = rule_based_risk(command)
    if override:
        return {"result": override, "confidence": 1.0, "source": "rules"}
    inputs = tok(command, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    idx   = probs.argmax().item()
    return {
        "result":     LABEL_MAP[idx],
        "confidence": round(probs[idx].item(), 3),
        "source":     "finetuned"
    }

def assess_zeroshot(command: str, classifier) -> dict:
    override = rule_based_risk(command)
    if override:
        return {"result": override, "confidence": 1.0, "source": "rules"}
    try:
        out   = classifier(command, ZS_LABELS, multi_label=False)
        idx   = ZS_LABELS.index(out["labels"][0])
        level = ZS_RISK_MAP[idx]
        return {"result": level, "confidence": round(out["scores"][0], 3), "source": "zeroshot"}
    except Exception as e:
        return {"result": "MEDIUM", "confidence": 0.5, "source": "fallback", "error": str(e)}

# ── Summarization ─────────────────────────────────────────────────────────────

def parse_session(session_text: str) -> list:
    entries = []
    for part in session_text.split(";"):
        part = part.strip()
        if not part:
            continue
        risk = "LOW"
        risk_match = re.search(r"\[(HIGH|MEDIUM|LOW)\]", part)
        if risk_match:
            risk = risk_match.group(1)
            part = part[:risk_match.start()].strip()
        if ": " in part:
            status, cmd = part.split(": ", 1)
        else:
            status, cmd = "UNKNOWN", part
        entries.append({"status": status.strip(), "cmd": cmd.strip(), "risk": risk})
    return entries

def describe_command(cmd: str) -> str:
    """Convert a raw shell command into a natural language description."""
    c = cmd.strip()
    parts = c.split()
    if not parts:
        return cmd
    base = parts[0]
    rest = parts[1:] if len(parts) > 1 else []
    args = " ".join(rest)

    if base == "mkdir":
        dirs = [p for p in rest if not p.startswith("-")]
        return f"created director{'ies' if len(dirs) > 1 else 'y'} {', '.join(dirs)}" if dirs else "created a directory"
    if base in ("rm", "rmdir"):
        targets = [p for p in rest if not p.startswith("-")]
        flag = " recursively" if any(f in args for f in ["-r", "-rf", "-R"]) else ""
        return f"deleted{flag} {', '.join(targets)}" if targets else "deleted files"
    if base == "cp":
        files = [p for p in rest if not p.startswith("-")]
        return f"copied {files[0]} to {files[1]}" if len(files) >= 2 else "copied files"
    if base == "mv":
        files = [p for p in rest if not p.startswith("-")]
        return f"moved {files[0]} to {files[1]}" if len(files) >= 2 else "moved files"
    if base == "touch":
        targets = [p for p in rest if not p.startswith("-")]
        return f"created file {', '.join(targets)}" if targets else "created a file"
    if base == "cat":
        targets = [p for p in rest if not p.startswith("-")]
        return f"read {', '.join(targets)}" if targets else "read file contents"
    if base == "ls":
        path = next((p for p in rest if not p.startswith("-")), None)
        return f"listed files in {path}" if path else "listed files in current directory"
    if base == "cd":
        return f"changed directory to {rest[0]}" if rest else "changed to home directory"
    if base == "pwd":
        return "checked current directory"
    if base == "echo":
        msg = args.strip('"').strip("'")
        return f"printed '{msg}'"
    if base == "grep":
        non_flag = [p for p in rest if not p.startswith("-")]
        pattern = non_flag[0] if non_flag else "pattern"
        targets = non_flag[1:] if len(non_flag) > 1 else []
        return f"searched for '{pattern}'" + (f" in {', '.join(targets)}" if targets else "")
    if base in ("head", "tail"):
        non_flag = [p for p in rest if not p.startswith("-")]
        direction = "first" if base == "head" else "last"
        n = next((rest[i+1] for i, p in enumerate(rest) if p == "-n" and i+1 < len(rest)), "10")
        return f"showed {direction} {n} lines of {non_flag[0]}" if non_flag else f"showed {direction} lines"
    if base == "wc":
        non_flag = [p for p in rest if not p.startswith("-")]
        what = "lines" if "-l" in args else "words" if "-w" in args else "characters" if "-c" in args else "line count"
        return f"counted {what} in {non_flag[0]}" if non_flag else f"counted {what}"
    if base == "chmod":
        non_flag = [p for p in rest if not p.startswith("-")]
        return f"set permissions {non_flag[0]} on {non_flag[1]}" if len(non_flag) >= 2 else "changed file permissions"
    if base == "chown":
        non_flag = [p for p in rest if not p.startswith("-")]
        return f"changed owner of {non_flag[1]} to {non_flag[0]}" if len(non_flag) >= 2 else "changed file ownership"
    if base == "git" and rest:
        sub = rest[0]
        sub_rest = rest[1:]
        if sub == "push":
            force = " (force)" if "--force" in sub_rest or "-f" in sub_rest else ""
            remote = next((a for a in sub_rest if not a.startswith("-")), "origin")
            branch = next((a for a in sub_rest[1:] if not a.startswith("-")), "current branch")
            return f"pushed{force} to {remote}/{branch}"
        if sub == "pull":
            return f"pulled from {sub_rest[0] if sub_rest else 'origin'}"
        if sub == "commit":
            msg = next((sub_rest[i+1] for i, a in enumerate(sub_rest) if a == "-m" and i+1 < len(sub_rest)), None)
            return f"committed with message '{msg}'" if msg else "committed staged changes"
        if sub == "merge":
            return f"merged branch {sub_rest[0]}" if sub_rest else "merged branch"
        if sub in ("checkout", "switch"):
            new = "-b" in sub_rest
            branch = next((a for a in sub_rest if not a.startswith("-")), "branch")
            return f"{'created and ' if new else ''}switched to branch {branch}"
        if sub == "clone":
            return f"cloned repository {sub_rest[0]}" if sub_rest else "cloned a repository"
        if sub == "add":
            return f"staged {sub_rest[0] if sub_rest else 'changes'} for commit"
        if sub == "stash":
            action = sub_rest[0] if sub_rest else "saved"
            return f"stashed changes ({action})"
        if sub == "log":
            return "viewed commit history"
        if sub == "status":
            return "checked git status"
        if sub == "diff":
            return "viewed uncommitted changes"
        if sub == "reset":
            ref = next((a for a in sub_rest if not a.startswith("-")), "HEAD")
            mode = "hard" if "--hard" in sub_rest else "soft" if "--soft" in sub_rest else ""
            return f"reset {mode+' ' if mode else ''}to {ref}"
        if sub == "rebase":
            return f"rebased onto {sub_rest[0]}" if sub_rest else "rebased"
        if sub == "tag":
            return f"created tag {sub_rest[0]}" if sub_rest and not sub_rest[0].startswith("-") else "managed tags"
        if sub == "fetch":
            return "fetched remote changes"
        if sub == "remote":
            return f"managed remotes ({' '.join(sub_rest[:2])})"
        return f"ran git {sub}"
    if base == "docker" and rest:
        sub = rest[0]
        sub_rest = rest[1:]
        if sub == "build":
            tag = next((sub_rest[i+1] for i, a in enumerate(sub_rest) if a in ("-t","--tag") and i+1 < len(sub_rest)), None)
            return f"built Docker image{' ' + tag if tag else ''}"
        if sub == "run":
            img = next((a for a in sub_rest if not a.startswith("-")), "image")
            return f"ran container from {img}"
        if sub in ("start", "stop", "restart"):
            cid = next((a for a in sub_rest if not a.startswith("-")), "container")
            return f"{sub}ped container {cid}"
        if sub in ("rm", "rmi"):
            target = next((a for a in sub_rest if not a.startswith("-")), "")
            return f"removed {'image' if sub == 'rmi' else 'container'} {target}".strip()
        if sub == "ps":
            return "listed running containers"
        if sub == "logs":
            cid = next((a for a in sub_rest if not a.startswith("-")), "container")
            return f"viewed logs for {cid}"
        if sub == "pull":
            return f"pulled image {sub_rest[0]}" if sub_rest else "pulled Docker image"
        if sub == "push":
            return f"pushed image {sub_rest[0]}" if sub_rest else "pushed Docker image"
        if sub == "exec":
            cid = next((a for a in sub_rest if not a.startswith("-")), "container")
            return f"ran command inside container {cid}"
        return f"ran docker {sub}"
    if base == "kubectl" and rest:
        sub = rest[0]
        targets = [a for a in rest[1:] if not a.startswith("-")]
        if sub == "get":
            return f"listed Kubernetes {targets[0] if targets else 'resources'}"
        if sub == "apply":
            f_arg = next((rest[i+1] for i, a in enumerate(rest) if a == "-f" and i+1 < len(rest)), None)
            return f"applied manifest {f_arg}" if f_arg else "applied Kubernetes manifest"
        if sub == "delete":
            return f"deleted {' '.join(targets[:2])}" if targets else "deleted Kubernetes resource"
        if sub in ("logs", "describe"):
            return f"{sub}d {targets[0]}" if targets else f"ran kubectl {sub}"
        return f"ran kubectl {sub} {' '.join(targets[:2])}".strip()
    if base == "aws" and rest:
        return f"ran AWS CLI: {' '.join(rest[:4])}"
    if base == "terraform" and rest:
        return f"ran terraform {rest[0]}"
    if base == "helm" and rest:
        sub = rest[0]
        name = next((a for a in rest[1:] if not a.startswith("-")), "")
        return f"ran helm {sub} {name}".strip()
    if base in ("pip", "pip3") and rest:
        action = rest[0]
        pkg = next((a for a in rest[1:] if not a.startswith("-")), "")
        return f"{'installed' if action == 'install' else action+'ed'} Python package {pkg}".strip()
    if base in ("npm", "yarn") and rest:
        action = rest[0]
        pkg = next((a for a in rest[1:] if not a.startswith("-")), "")
        return f"ran {base} {action} {pkg}".strip()
    if base in ("apt", "apt-get", "brew", "yum", "dnf") and rest:
        action = rest[0]
        pkg = next((a for a in rest[1:] if not a.startswith("-")), "")
        return f"{'installed' if action == 'install' else action+'ed'} {pkg}".strip()
    if base in ("kill", "pkill", "killall"):
        return f"terminated process {args}"
    if base in ("ps", "top", "htop"):
        return "inspected running processes"
    if base == "ssh":
        host = next((a for a in rest if not a.startswith("-")), "remote host")
        return f"connected via SSH to {host}"
    if base == "curl":
        method = next((rest[i+1] for i, a in enumerate(rest) if a == "-X" and i+1 < len(rest)), "GET")
        url = next((a for a in rest if a.startswith("http")), args)
        return f"made {method} HTTP request to {url}"
    if base == "sudo":
        inner = describe_command(args) if args else "a privileged command"
        return f"ran with sudo: {inner}"
    if base == "systemctl" and rest:
        return f"{rest[0]}ed service {rest[1]}" if len(rest) > 1 else f"ran systemctl {rest[0]}"
    if base == "python3" or base == "python":
        script = next((a for a in rest if not a.startswith("-") and a.endswith(".py")), None)
        return f"ran Python script {script}" if script else "ran Python"
    if base == "java":
        return f"ran Java {'JAR ' + next((rest[i+1] for i, a in enumerate(rest) if a == '-jar' and i+1 < len(rest)), '') if '-jar' in rest else 'program'}"
    if base in ("mvn", "gradle"):
        return f"ran {base} {' '.join(rest[:2])}" if rest else f"ran {base}"
    # Generic fallback — just clean up the command
    short = c[:80] if len(c) > 80 else c
    return short

NOISE_COMMANDS = {
    "ls", "ll", "la", "pwd", "cd", "echo", "cat", "head", "tail",
    "wc", "grep", "find", "which", "whoami", "date", "clear", "history",
    "man", "help", "less", "more", "diff", "sort", "uniq", "cut", "awk",
    "ps", "top", "htop", "df", "du", "free", "uname", "lsblk", "blkid",
    "env", "printenv", "export", "alias", "type", "file", "stat",
    "ping", "curl -I", "curl -s", "ssh -T", "nslookup", "dig", "host",
    "git log", "git status", "git diff", "git show", "git branch",
    "git fetch", "git stash list", "git bisect",
    "docker ps", "docker images", "docker logs", "docker inspect", "docker stats",
    "kubectl get", "kubectl describe", "kubectl logs",
    "terraform plan", "terraform validate", "terraform output",
    "helm list", "helm show", "helm get", "helm history",
    "ansible --version", "kubectl version", "helm version", "docker version",
}

def is_noise(cmd: str) -> bool:
    """Return True if this command has no meaningful developer narrative value."""
    c = cmd.strip().lower()
    parts = c.split()
    if not parts:
        return True
    base = parts[0]

    # Pure navigation / inspection — no side effects
    if base in ("cd", "ls", "ll", "pwd", "echo", "cat", "head", "tail",
                "wc", "grep", "find", "which", "whoami", "date", "clear",
                "history", "man", "less", "more", "diff", "sort", "uniq",
                "cut", "awk", "sed", "tee", "env", "printenv", "export",
                "alias", "type", "file", "stat", "lsblk", "blkid",
                "uname", "free", "vmstat", "iostat", "sar", "uptime",
                "ping", "nslookup", "dig", "host", "traceroute",
                "watch", "screen", "tmux", "htop", "top", "ps"):
        return True

    # Git read-only
    if base == "git" and len(parts) > 1:
        sub = parts[1]
        if sub in ("log", "status", "diff", "show", "branch", "fetch",
                   "stash", "bisect", "describe", "shortlog", "reflog",
                   "remote", "tag", "blame", "archive"):
            # Only noise if no destructive flags
            if not any(f in cmd for f in ["--force", "--hard", "--delete", "-d", "expire"]):
                return True

    # Docker read-only
    if base == "docker" and len(parts) > 1:
        if parts[1] in ("ps", "images", "logs", "inspect", "stats",
                        "info", "version", "search", "manifest", "scan"):
            return True

    # kubectl read-only
    if base == "kubectl" and len(parts) > 1:
        if parts[1] in ("get", "describe", "logs", "top", "explain",
                        "api-resources", "version", "auth", "config"):
            return True

    # terraform read-only
    if base == "terraform" and len(parts) > 1:
        if parts[1] in ("plan", "validate", "output", "show", "version",
                        "providers", "state list", "state pull"):
            return True

    # Version checks, help, lint, test runners (no side effects)
    if any(v in cmd for v in ["--version", "--help", "-h", " -v "]):
        return True

    return False

def summarize(session_text: str) -> dict:
    entries = parse_session(session_text)

    # Only keep commands with developer narrative value
    meaningful = [
        e for e in entries
        if not is_noise(e["cmd"]) or e["status"].startswith("BLOCKED")
    ]

    if not meaningful:
        return {"result": "Session contained only routine commands (file navigation, inspection) — nothing significant to report.", "confidence": 1.0}

    ok_cmds   = [e for e in meaningful if e["status"] == "OK"]
    errors    = [e for e in meaningful if e["status"] == "ERROR"]
    blocked   = [e for e in meaningful if "BLOCKED" in e["status"]]
    cancelled = [e for e in meaningful if e["status"] == "CANCELLED"]
    high      = [e for e in meaningful if e["risk"] == "HIGH"]

    parts = []

    # Narrate successful meaningful actions
    if ok_cmds:
        descriptions = [describe_command(e["cmd"]) for e in ok_cmds]
        if len(descriptions) == 1:
            parts.append(f"The user {descriptions[0]}.")
        elif len(descriptions) <= 5:
            parts.append(f"The user {', '.join(descriptions[:-1])}, and {descriptions[-1]}.")
        else:
            parts.append(
                f"The user ran {len(ok_cmds)} operations: "
                + ", ".join(descriptions[:5])
                + f", and {len(ok_cmds)-5} more."
            )

    # Cancelled actions
    if cancelled:
        descs = [describe_command(e["cmd"]) for e in cancelled]
        parts.append(f"Decided not to proceed with: {', '.join(descs[:3])}.")

    # Errors
    if errors:
        descs = [describe_command(e["cmd"]) for e in errors]
        parts.append(f"{len(errors)} action(s) failed: {', '.join(descs[:3])}.")

    # Blocked
    if blocked:
        priv   = [e for e in blocked if "PRIVILEGE" in e["status"]]
        risk_b = [e for e in blocked if "RISK"      in e["status"]]
        if priv:
            descs = [describe_command(e["cmd"]) for e in priv[:3]]
            parts.append(f"Blocked (insufficient privilege): {', '.join(descs)}.")
        if risk_b:
            descs = [describe_command(e["cmd"]) for e in risk_b[:3]]
            parts.append(f"Blocked (HIGH risk): {', '.join(descs)}.")

    # Verdict
    if len(high) >= 3:
        parts.append("⚠  Multiple high-risk operations attempted this session.")
    elif len(high) > 0:
        parts.append(f"⚠  {len(high)} high-risk operation(s) detected.")
    elif not blocked and not errors:
        parts.append("✓  All operations completed successfully.")

    return {"result": " ".join(parts), "confidence": 1.0}


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    tok, ft_model = load_finetuned()
    zs_classifier = None

    if ft_model:
        print(json.dumps({"status": "ready", "model": "finetuned"}), flush=True)
    else:
        try:
            zs_classifier = load_zeroshot()
            print(json.dumps({"status": "ready", "model": "zeroshot"}), flush=True)
        except ImportError as e:
            for line in sys.stdin:
                print(json.dumps({"error": f"Missing dep: {e}. Run: pip install transformers torch"}), flush=True)
            return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req    = json.loads(line)
            action = req.get("action", "")
            data   = req.get("data", "")

            if action == "assess":
                result = (
                    assess_finetuned(data, tok, ft_model)
                    if ft_model else
                    assess_zeroshot(data, zs_classifier)
                )
            elif action == "summarize":
                result = summarize(data)
            else:
                result = {"error": f"Unknown action: {action}"}

            print(json.dumps(result), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    main()