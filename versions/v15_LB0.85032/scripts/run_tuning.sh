#!/usr/bin/env bash
# Sequential Optuna studies (each persists to outputs/optuna.db and outputs/params/*.json).
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
mkdir -p outputs/logs
for job in "1 lgb 40" "3 lgb 30" "2 lgb 15"; do
  set -- $job
  echo "=== task $1 $2 ($3 trials) $(date +%T)"
  $PY -u scripts/tune.py --task "$1" --model "$2" --trials "$3" 2>&1 | grep --line-buffered -v -i warning | tee -a "outputs/logs/tune_task$1_$2.log"
done
echo "=== tuning done $(date +%T)"
