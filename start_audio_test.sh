#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:a:h}"
PYTHON_BIN="$(command -v python3 || true)"

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3 not found. Please install python3 first." >&2
  exit 1
fi

mode="${1:-stream}"
seconds="${2:-10}"
model="${3:-auto}"
language="${4:-zh}"
output_dir="${5:-}"
extra_args=()

if [[ -n "$output_dir" ]]; then
  extra_args+=(--output-dir "$output_dir")
fi

if [[ "$mode" == "list" ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/run_audio_test.py" --list-models
fi

case "$mode" in
  stream)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/run_audio_test.py" --mode stream --seconds "$seconds" --model "$model" --language "$language" "${extra_args[@]}"
    ;;
  direct)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/run_audio_test.py" --mode direct --seconds "$seconds" --model "$model" --language "$language" "${extra_args[@]}"
    ;;
  skip-mic)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/run_audio_test.py" --skip-mic --model "$model" --language "$language" "${extra_args[@]}"
    ;;
  *)
    echo "Usage: zsh start_audio_test.sh [stream|direct|skip-mic|list] [seconds] [auto|tiny|base|small] [language] [output_dir]" >&2
    exit 1
    ;;
esac
