#!/bin/bash
#
# Install / uninstall the daily routines via macOS launchd.
#
# Usage:
#   bash tools/setup_schedule.sh install    # register morning + evening jobs
#   bash tools/setup_schedule.sh uninstall  # remove both jobs
#   bash tools/setup_schedule.sh status     # check if they're loaded
#   bash tools/setup_schedule.sh run        # run the routine right now (test)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
PLIST_DIR="$ROOT/tools/launchd"
LABELS=("com.quant.morning" "com.quant.evening")

case "${1:-status}" in

install)
    mkdir -p "$LAUNCH_DIR" "$ROOT/logs"
    for label in "${LABELS[@]}"; do
        src="$PLIST_DIR/$label.plist"
        dst="$LAUNCH_DIR/$label.plist"
        if [ ! -f "$src" ]; then
            echo "❌  $src not found"; exit 1
        fi
        cp "$src" "$dst"
        launchctl unload "$dst" 2>/dev/null || true
        launchctl load "$dst"
        echo "✅  $label loaded"
    done
    echo ""
    echo "Schedule:"
    echo "  🌅 Morning (7:00 AM) — quick refresh (news + prices) + report"
    echo "  🌆 Evening (6:30 PM) — full ingest + screens + report"
    echo "  📧 Reports emailed to: \$QUANT_EMAIL (set in .env or shell profile)"
    echo ""
    echo "Logs: $ROOT/logs/"
    echo "To test now: bash tools/setup_schedule.sh run"
    ;;

uninstall)
    for label in "${LABELS[@]}"; do
        dst="$LAUNCH_DIR/$label.plist"
        if [ -f "$dst" ]; then
            launchctl unload "$dst" 2>/dev/null || true
            rm "$dst"
            echo "🗑  $label unloaded and removed"
        else
            echo "⚠️  $label not installed"
        fi
    done
    ;;

status)
    echo "Quant routine schedule:"
    for label in "${LABELS[@]}"; do
        if launchctl list "$label" &>/dev/null; then
            echo "  ✅ $label — active"
        else
            echo "  ❌ $label — not loaded"
        fi
    done
    echo ""
    echo "Recent logs:"
    ls -lt "$ROOT/logs/routine_"*.log 2>/dev/null | head -5 || echo "  (none yet)"
    ;;

run)
    echo "Running routine now (full mode)..."
    bash "$ROOT/tools/daily_routine.sh" full
    ;;

*)
    echo "Usage: $0 {install|uninstall|status|run}"
    exit 1
    ;;
esac
