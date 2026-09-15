# Data

## `commands.csv`

The full command list used for both the app and the risk model — 1,018 commands
covering everything from everyday shell utilities to git, docker, cloud CLIs
(aws/az/gcloud), Kubernetes, Terraform, and more.

| privilege_level | Meaning | Risk | Example |
|---|---|---|---|
| `0` | Safe / read-only | LOW | `ls`, `git status` |
| `1` | Normal ops with side effects | MEDIUM | `chmod`, `git push` |
| `2` | Destructive / irreversible | HIGH | `sudo`, `rm -rf`, `git push --force` |

This single scale is used consistently everywhere: `CommandAI.java` (privilege
enforcement), `ml_engine.py` (fallback rule-based risk), and `ml/train.py`
(fine-tuning labels).

Add your own commands and reload live with:
```
:reload path/to/your_commands.csv
```

> **History note:** this file used to be split across `commands.csv` (a
> 90-command starter list on an old 1–3 scale) and `commands_v2.csv` (a
> 900+ command training set on a 0–2 scale). They've been merged into one
> file on the 0–2 scale, which is what `CommandAI.java` actually reads
> (`userTier`/`privilegeLevel` are compared directly as ints 0–2). If you
> have an older `commands.csv` lying around from before this merge, don't
> mix it in — it uses the old, incompatible scale.
