#!/bin/sh
#
# Container entrypoint: make sure a database exists, then serve.
#
# Two things have to happen before gunicorn that `python app.py` would
# otherwise do for itself:
#
#   1. Fetch the seed database. A demo deployment ships a point-in-time
#      snapshot because ingestion is disabled, and the image stays small by
#      fetching it at startup rather than baking it into a layer.
#   2. Create the schema. app.py only calls db.init_db() under __main__, so
#      under a WSGI server it never runs and every query would hit a missing
#      table.

set -eu

DB_PATH="${DB_PATH:-db/quant.sqlite}"

# Only seed the default SQLite file. If DATABASE_URL points somewhere else the
# store is managed elsewhere and downloading a file over it would be wrong.
case "${DATABASE_URL:-sqlite}" in
    sqlite*|'')
        if [ -n "${SEED_DB_URL:-}" ] && [ ! -s "$DB_PATH" ]; then
            echo "entrypoint: fetching seed database from \$SEED_DB_URL"
            # A partial download is worse than none: it produces a file that
            # looks present but fails on read. Stage it, then move.
            #
            # urllib rather than curl so the image needs no apt layer at all.
            # The URL travels in the environment, not in argv, so a signed
            # link with a token in its query string stays out of the process
            # list.
            if SEED_URL="$SEED_DB_URL" OUT="$DB_PATH.part" python -c '
import os, urllib.request
urllib.request.urlretrieve(os.environ["SEED_URL"], os.environ["OUT"])
'; then
                mv "$DB_PATH.part" "$DB_PATH"
                echo "entrypoint: seed database ready ($(wc -c < "$DB_PATH") bytes)"
            else
                rm -f "$DB_PATH.part"
                echo "entrypoint: seed download failed — starting with an empty store" >&2
            fi
        fi
        ;;
esac

python cli.py init

# One worker: the factor cache, the HTTP response cache and SQLite itself are
# all per-process, so a second worker would serve answers from a different
# cache and quietly disagree with the first. Threads keep the UI responsive
# without that risk.
#
# The long timeout is not generosity. Screens take tens of seconds on a cold
# factor cache and backtests take minutes, and both run inside the request;
# gunicorn's 30s default would kill them mid-flight and show a bare 502.
exec gunicorn app:server \
    --bind "0.0.0.0:${PORT:-7860}" \
    --workers 1 \
    --threads 4 \
    --timeout "${GUNICORN_TIMEOUT:-300}" \
    --access-logfile - \
    --error-logfile -
