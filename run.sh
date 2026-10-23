#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

CONDA_EXEC=$(command -v conda || true)
if [ -z "$CONDA_EXEC" ]; then
    for path in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
        if [ -x "$path" ]; then
            CONDA_EXEC="$path";
            break;
        fi
    done
fi

if [ "$CONDA_DEFAULT_ENV" != "lerobot" ]; then
    if [ -z "$CONDA_EXEC" ]; then
        echo -e "\033[0;31m[ERROR]\033[0m conda not found: run ./setup.sh first"; exit 1
    fi
    source "$("$CONDA_EXEC" info --base)/etc/profile.d/conda.sh"
    conda activate lerobot
fi

python run.py "$@"