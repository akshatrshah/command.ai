#!/usr/bin/env bash
# Command.ai — repo reorganization script
# Run this from the ROOT of your local clone of command.ai
#
#   cd ~/path/to/command.ai
#   bash reorganize.sh
#
# It uses `git mv` so history is preserved for every moved file.
# Safe to re-run: it skips anything that's already been moved.

set -e

echo "== Command.ai reorg =="

# 1. Stop tracking build output
if [ -d target ]; then
  echo "Removing target/ from git tracking (build artifact)..."
  git rm -r --cached target >/dev/null 2>&1 || true
fi

# 2. Create new folders
mkdir -p demo/screenshots data ml

# 3. Data file: commands.csv and commands_v2.csv have been merged into one
#    file (single 0-2 privilege scale) — see data/commands.csv provided
#    alongside this script. Remove the two old root-level CSVs so they don't
#    linger with an incompatible scale.
[ -f commands.csv ]    && git rm commands.csv
[ -f commands_v2.csv ] && git rm commands_v2.csv

# 4. Move ML tooling
[ -f train.py ]      && git mv train.py      ml/train.py
[ -f test_model.py ] && git mv test_model.py ml/test_model.py

# 5. Placeholder files (added below by Claude, just noting expected paths)
touch demo/screenshots/.gitkeep

echo ""
echo "Done. Now:"
echo "  1. Copy in the files from the provided package (overwrite where prompted):"
echo "       README.md, .gitignore, LICENSE"
echo "       data/README.md, data/commands.csv   <- merged, single 0-2 scale"
echo "       ml/README.md"
echo "       demo/README.md"
echo "  2. Drop your dashboard screenshots/GIF into demo/screenshots/"
echo "  3. git add -A && git commit -m 'Reorganize project structure, merge commands CSVs'"
echo "  4. git push"
