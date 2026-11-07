#!/bin/bash
#
# Scheduled routine — a thin wrapper around `python cli.py daily`.
#
#   bash tools/daily_routine.sh morning   # the one email of the day:
#                                         #   trading day  -> full brief
#                                         #   weekend / NYSE holiday -> news-only digest
#   bash tools/daily_routine.sh evening   # weekday evening: full ingest, screens and a
#                                         # database backup; no email. Prepares the
#                                         # next morning's brief so it is fast.
#
# The old mode names still work: quick = morning, full = evening.
#
# Recipient, index and watchlist come from .env (QUANT_EMAIL, QUANT_INDEX,
# QUANT_WATCHLIST) and are read by Python. This script deliberately does not
# `source .env`: values such as SEC_USER_AGENT contain spaces, and sourcing
# would try to execute the second word as a command.
#
# Which email goes out is decided by the exchange calendar, not by this
# script, and a report already emailed today is never sent twice — so it is
# safe for launchd to re-run a job missed while the machine was asleep.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Interpreter: $QUANT_PYTHON, else a project virtualenv, else python3 on PATH.
if [ -n "${QUANT_PYTHON:-}" ]; then
    PYTHON="$QUANT_PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/venv/bin/python" ]; then
    PYTHON="$ROOT/venv/bin/python"
else
    PYTHON="$(command -v python3 || true)"
fi
if [ -z "$PYTHON" ]; then
    echo "python3 not found — set QUANT_PYTHON to your interpreter" >&2
    exit 1
fi

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
MODE="${1:-morning}"
LOGFILE="$LOG_DIR/routine_$(date +%Y%m%d_%H%M)_${MODE}.log"

case "$MODE" in
    morning|quick) ARGS=(daily --mode quick) ;;
    evening|full)  ARGS=(daily --mode full --no-email --backup) ;;
    *) echo "usage: $0 {morning|evening}" >&2; exit 2 ;;
esac

echo "=== $(date '+%Y-%m-%d %H:%M:%S')  routine: $MODE  python: $PYTHON ===" \
    | tee -a "$LOGFILE"
"$PYTHON" cli.py "${ARGS[@]}" 2>&1 | tee -a "$LOGFILE"
rc=${PIPESTATUS[0]}
echo "=== exit $rc ===" | tee -a "$LOGFILE"

# Keep two weeks of logs.
find "$LOG_DIR" -name 'routine_*.log' -mtime +14 -delete 2>/dev/null || true

exit "$rc"
