#!/bin/bash
#
# Install / uninstall the daily routine.
#
#   bash tools/setup_schedule.sh install    # macOS: register both launchd jobs
#   bash tools/setup_schedule.sh uninstall  # macOS: remove both jobs
#   bash tools/setup_schedule.sh status     # macOS: are they loaded? recent logs
#   bash tools/setup_schedule.sh cron       # Linux (or macOS cron): print crontab lines
#   bash tools/setup_schedule.sh run        # run the morning routine now (sends email)
#   bash tools/setup_schedule.sh test       # dry run: which email would go out today
#
# Schedule:
#   07:00 every day       one email — full brief on trading days, a news-only
#                         digest on weekends and NYSE holidays
#   18:30 Mon–Fri         full ingest + screens + DB backup, no email
#
# Set QUANT_EMAIL (and optionally QUANT_WATCHLIST, QUANT_INDEX) in .env first.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
PLIST_DIR="$ROOT/tools/launchd"
LABELS=("com.quant.morning" "com.quant.evening")

python_bin() {
    if [ -n "${QUANT_PYTHON:-}" ]; then echo "$QUANT_PYTHON"
    elif [ -x "$ROOT/.venv/bin/python" ]; then echo "$ROOT/.venv/bin/python"
    elif [ -x "$ROOT/venv/bin/python" ]; then echo "$ROOT/venv/bin/python"
    else command -v python3; fi
}

check_email() {
    if ! (cd "$ROOT" && "$(python_bin)" -c 'import config, sys; sys.exit(0 if config.QUANT_EMAIL else 1)'); then
        echo "⚠️  QUANT_EMAIL is not set in $ROOT/.env — the morning job will"
        echo "    exit with an error until it is. Add e.g. QUANT_EMAIL=you@example.com"
    fi
}

case "${1:-status}" in

install)
    if [ "$(uname)" != "Darwin" ]; then
        echo "launchd is macOS-only. On Linux use:  bash $0 cron"; exit 1
    fi
    PY="$(python_bin)"
    # launchd starts jobs with a minimal PATH; give it the interpreter's
    # directory first so the same python3 that has the dependencies is used.
    JOB_PATH="$(dirname "$PY"):/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    mkdir -p "$LAUNCH_DIR" "$ROOT/logs"
    for label in "${LABELS[@]}"; do
        src="$PLIST_DIR/$label.plist"
        dst="$LAUNCH_DIR/$label.plist"
        [ -f "$src" ] || { echo "❌  $src not found"; exit 1; }
        # Fill in this checkout's real paths. The template's placeholders
        # used to be copied verbatim, so the job pointed at /Users/CHANGEME.
        sed -e "s|__ROOT__|$ROOT|g" -e "s|__PATH__|$JOB_PATH|g" "$src" > "$dst"
        plutil -lint "$dst" >/dev/null
        launchctl unload "$dst" 2>/dev/null || true
        launchctl load "$dst"
        echo "✅  $label loaded"
    done
    echo ""
    echo "Schedule:"
    echo "  🌅 07:00 daily  — full brief on trading days, news-only digest when closed"
    echo "  🌆 18:30 Mon–Fri — full ingest + screens + backup (no email)"
    echo "  🐍 python: $PY"
    check_email
    echo ""
    echo "Logs: $ROOT/logs/"
    echo "Dry run: bash $0 test     Send now: bash $0 run"
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

cron)
    PY="$(python_bin)"
    cat <<EOF
# Add with: crontab -e     (times are this machine's local time)
# 07:00 every day — full brief on trading days, news-only digest when closed
0 7 * * *     cd "$ROOT" && "$PY" cli.py daily --mode quick >> "$ROOT/logs/daily.log" 2>&1
# 18:30 Mon–Fri — full ingest + screens + backup, no email
30 18 * * 1-5 cd "$ROOT" && "$PY" cli.py daily --mode full --no-email --backup >> "$ROOT/logs/daily.log" 2>&1
EOF
    mkdir -p "$ROOT/logs"
    check_email
    ;;

run)
    echo "Running the morning routine now (this sends today's email)..."
    bash "$ROOT/tools/daily_routine.sh" morning
    ;;

test)
    cd "$ROOT" && "$(python_bin)" cli.py daily --dry-run
    ;;

*)
    echo "Usage: $0 {install|uninstall|status|cron|run|test}"
    exit 1
    ;;
esac
