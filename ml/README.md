# ML Tooling

Scripts for fine-tuning and evaluating the risk-assessment model. These are
developer tools — end users of Command.ai never run these directly; the
packaged app falls back to the rule-based classifier or a zero-shot model if
no fine-tuned model is present.

## `train.py`

Fine-tunes `distilbert-base-uncased` on `../data/commands.csv` to classify
commands as `LOW` / `MEDIUM` / `HIGH` risk (privilege_level `0`/`1`/`2`).

```bash
source ~/.command_ai/venv/bin/activate
python3 train.py --data ../data/commands.csv
```

Saves the resulting model to `~/.command_ai/model/`, where the packaged app's
`ml_engine.py` (in `src/main/resources/`) will pick it up automatically.

## `test_model.py`

Runs a fixed set of test commands through both the rule-based classifier and
the fine-tuned model side by side, so you can see where they agree or
disagree and get an accuracy score for each.

```bash
source ~/.command_ai/venv/bin/activate
python3 test_model.py
```
