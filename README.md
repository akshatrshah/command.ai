# Command.ai

> Terminal observability tool with ML-based risk assessment and role-based privilege enforcement.

---

## What it does

- **Spawns an authenticated terminal session** where every command is intercepted before execution
- **Records 900+ developer commands** into structured JSONL logs (`~/.command_ai/logs/`)
- **Assesses risk** using a local T5-small transformer model (CPU-friendly, no GPU required, ~240MB download on first run)
- **Enforces privilege tiers** — blocks or warns before executing commands above your access level
- **Summarizes sessions** with AI at the end of each session

Each terminal gets its own log file (`session_<timestamp>_<pid>.jsonl`) so multiple concurrent terminals never conflict.

---

## Privilege Tiers

| Tier | Role        | Access                                  |
|------|-------------|-----------------------------------------|
| 1    | Root/Admin  | All commands                            |
| 2    | Power User  | All except tier-1-only (sudo, rm -rf…) |
| 3    | Standard    | Safe commands only                      |

Lower number = more privileged (tier 1 is highest).

---

## Risk Levels

| Risk   | Action                           |
|--------|----------------------------------|
| HIGH   | Blocked + logged (no execution)  |
| MEDIUM | Warning shown, confirm to run    |
| LOW    | Execute immediately              |

Risk is assessed by a local **T5-small** model (zero-shot classification) with a fast regex pre-screen for known dangerous patterns (fork bombs, `rm -rf`, disk writes, etc.).

---

## Requirements

### Java
- **JDK 21+** (for compilation)
- **JRE 21+** (for running the JAR)
- Maven 3.6+ (for building)

### Python (for ML engine)
```bash
pip install transformers torch sentencepiece
```
> The Python sidecar is bundled inside the JAR and extracted to `~/.command_ai/ml_engine.py` at first run. If Python/transformers aren't available, the tool falls back to fast rule-based risk assessment automatically.

---

## Build

```bash
# Clone / download project
cd commandai/

# Build the fat JAR (bundles ml_engine.py inside)
mvn clean package

# JAR is at:
ls target/commandai.jar
```

---

## Run

```bash
# Basic — prompts for tier interactively
java -jar target/commandai.jar

# With your commands CSV loaded at startup
java -jar target/commandai.jar --commands commands.csv

# Specify user and tier (skip prompts)
java -jar target/commandai.jar --user alice --tier 2 --commands commands.csv

# Multiple terminals — each gets its own log file (no conflicts)
java -jar target/commandai.jar --user bob --tier 3 &
java -jar target/commandai.jar --user alice --tier 1 &
```

---

## Commands CSV Format

```csv
command,description,privilege_level
ls,List directory contents,3
chmod,Change file permissions,2
sudo,Execute as superuser,1
```

- **command**: The base command name (e.g. `git`, `rm -rf`, `systemctl`)
- **description**: Human-readable description (shown in warnings/logs)
- **privilege_level**: `1` (root), `2` (power user), `3` (standard)

A starter `commands.csv` with 90+ common commands is included. Add yours and reload with:
```
:reload path/to/your_commands.csv
```

---

## Session Log Format

Logs are written to `~/.command_ai/logs/session_<timestamp>_<pid>.jsonl`.

Each line is a JSON object:
```json
{"type":"SESSION_START","session_id":"20240115_143022_12345","user":"alice","tier":2,"pid":12345,"started":"2024-01-15 14:30:22"}
{"type":"CMD","session":"20240115_143022_12345","user":"alice","tier":2,"ts":"2024-01-15 14:30:35","cmd":"ls -la","status":"OK","risk":"LOW","desc":"List directory contents","exit":0,"elapsed":"12ms"}
{"type":"CMD","session":"20240115_143022_12345","user":"alice","tier":2,"ts":"2024-01-15 14:31:02","cmd":"sudo rm -rf /var","status":"BLOCKED_PRIVILEGE","risk":"HIGH","desc":"Requires tier 1, user is tier 2","exit":-1,"elapsed":""}
```

---

## Built-in Controls (type in the terminal)

| Command           | Action                                      |
|-------------------|---------------------------------------------|
| `:help`           | Show help                                   |
| `:status`         | Session info, ML engine status, loaded cmds |
| `:history`        | All commands run this session               |
| `:summary`        | AI-generated session summary                |
| `:reload <path>`  | Hot-reload a new commands CSV               |
| `exit` / `quit`   | End session (prints summary first)          |

---

## Project Structure

```
commandai/
├── pom.xml                                         # Maven build
├── commands.csv                                    # Starter command definitions
├── README.md
└── src/
    └── main/
        ├── java/ai/command/
        │   └── CommandAI.java                      # Entire application (single file)
        └── resources/
            └── ml_engine.py                        # ML sidecar (bundled into JAR)
```
