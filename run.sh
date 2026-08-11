#!/bin/bash
# Quant Research Terminal — first-run setup and launch.
set -e

cd "$(dirname "$0")"

echo "📦  Installing dependencies…"
python3 -m pip install -r requirements.txt --quiet

if [ ! -f .env ]; then
  cp .env.example .env
  echo "📝  Created .env from the example. Set SEC_USER_AGENT to your own"
  echo "    name and email — the SEC refuses requests without one."
fi

echo "🗄️   Preparing the database…"
python3 cli.py init

if [ -z "$(python3 -c 'from core import db; print(db.read_sql("SELECT COUNT(*) n FROM prices").iloc[0]["n"] or "")' 2>/dev/null)" ] \
   || [ "$(python3 -c 'from core import db; print(db.read_sql("SELECT COUNT(*) n FROM prices").iloc[0]["n"])' 2>/dev/null)" = "0" ]; then
  echo ""
  echo "📊  No market data yet. Loading the Dow 30 (a few minutes)…"
  echo "    For the full S&P 500 later:"
  echo "      python3 cli.py ingest --index SPY --full --members-from 2015-01-01"
  echo ""
  python3 cli.py ingest --index DIA --full
fi

echo ""
echo "🚀  Starting the terminal at http://localhost:8050"
python3 app.py
