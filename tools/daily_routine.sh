#!/bin/bash
#
# Daily routine: ingest data → run screens → build and email report.
#
# Called by launchd morning and evening, or manually:
#   bash tools/daily_routine.sh          # full run (ingest + screen + report)
#   bash tools/daily_routine.sh quick    # news-only refresh + report
#
# The script is idempotent: ingest skips sources that are already fresh,
# and re-running a screen on the same day overwrites the prior run_id.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

LOGFILE="$LOG_DIR/routine_$(date +%Y%m%d_%H%M).log"
EMAIL="${QUANT_EMAIL:?Set QUANT_EMAIL in .env or your shell profile}"
INDEX="${QUANT_INDEX:-SPY}"
MODE="${1:-full}"

log() { echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOGFILE"; }

# Markets are closed on weekends — skip unless forced.
DOW=$(date +%u)  # 1=Mon … 7=Sun
if [ "$DOW" -ge 6 ] && [ "${QUANT_FORCE:-}" != "1" ]; then
    echo "$(date '+%H:%M:%S')  weekend — skipping (set QUANT_FORCE=1 to override)" >> "$LOGFILE"
    exit 0
fi

log "=== Quant routine ($MODE) — $(date '+%Y-%m-%d %H:%M') ==="
log "index=$INDEX  email=$EMAIL  python=$PYTHON"

# ── 1. Ingest ────────────────────────────────────────────
if [ "$MODE" = "quick" ]; then
    log "▸ quick refresh: news + prices only"
    "$PYTHON" cli.py ingest --index "$INDEX" --sources news,prices >> "$LOGFILE" 2>&1
else
    log "▸ full ingest (all sources)"
    "$PYTHON" cli.py ingest --index "$INDEX" --full >> "$LOGFILE" 2>&1
fi
log "✓ ingest done"

# ── 2. Run screens for all strategies ────────────────────
log "▸ running screens"
for strat in $("$PYTHON" -c "
from quant import strategies as ST
for k in ST.ALL_STRATEGIES: print(k)
"); do
    log "  ▸ $strat"
    "$PYTHON" cli.py screen --universe "$INDEX" --strategy "$strat" >> "$LOGFILE" 2>&1 || \
        log "  ⚠ $strat failed (continuing)"
done
log "✓ screens done"

# ── 3. Build and email the daily report ──────────────────
log "▸ building report"
"$PYTHON" cli.py report --kind daily --email "$EMAIL" >> "$LOGFILE" 2>&1
log "✓ report emailed to $EMAIL"

# ── 4. Cleanup old logs (keep 14 days) ───────────────────
find "$LOG_DIR" -name 'routine_*.log' -mtime +14 -delete 2>/dev/null || true

log "=== done ==="
