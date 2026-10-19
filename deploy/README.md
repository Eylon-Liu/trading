# Deploying the terminal

The app is a long-lived WSGI server with a SQLite store and on-disk caches, so
it needs a **container host**, not a serverless one. Vercel in particular cannot
run it: `config.py` creates `db/`, `cache/` and `reports/out/` at import time on
a read-only filesystem, the numeric stack is well over the 250 MB bundle limit,
and screens and backtests run inside the request and exceed the function
timeout. Hugging Face Spaces, Render and Fly.io all work.

These instructions use **Hugging Face Spaces**, which has a free Docker tier
with no card required.

---

## 1. Build the seed database (local, ~10 minutes)

Ingestion is disabled on the demo, so it ships with a snapshot instead. Build it
on your own machine, where the SEC and Yahoo rate limits apply to you rather
than to every visitor:

```bash
python3 cli.py init
python3 cli.py ingest --index DIA --full
```

The Dow 30 is the right universe for a demo: roughly 15–20 MB against the
250–300 MB a full S&P 500 history would take, and every tab has enough data to
be worth looking at.

**Optional — warm the AI cache.** New AI generation is off in demo mode, but
panels generated *before* deployment still display, because the Intel tab
checks the `company_briefs` table before calling a model. With your key in
`.env`, open the Intel tab locally for the handful of tickers you want the demo
to show, and those briefs travel with the database.

Shrink it before uploading — ingestion leaves a lot of free pages behind:

```bash
sqlite3 db/quant.sqlite 'VACUUM;'
```

## 2. Publish the snapshot

Attach `db/quant.sqlite` to a GitHub Release on this repository and copy the
asset's download URL. The container fetches it at startup rather than baking it
into an image layer, so the image stays small and the data can be refreshed by
replacing the asset — no rebuild.

## 3. Create the Space

1. New Space at <https://huggingface.co/new-space>, SDK **Docker**, hardware
   **CPU basic (free)**.
2. Push this repository to the Space remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<you>/<space-name>
   git push space main
   ```

3. Replace the Space's `README.md` with `deploy/space-README.md` from this
   repo — Hugging Face reads its YAML frontmatter to find the port.

## 4. Configure it

In **Settings → Variables and secrets**:

| Name | Kind | Value |
|---|---|---|
| `DEMO_MODE` | Variable | `1` |
| `SEED_DB_URL` | Variable | the Release asset URL from step 2 |
| `SEC_USER_AGENT` | Variable | `Your Name your@email.com` |

Leave `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` **unset**. Demo mode refuses new
generation regardless, but an absent key is a guarantee rather than a promise.

---

## What `DEMO_MODE` actually changes

The app has no authentication, so without this flag every visitor spends the
deployer's resources. It closes exactly four surfaces and leaves every database
read alone:

| Surface | Demo behaviour | Why |
|---|---|---|
| Data → Pull / Sync membership | Refused with an explanation | Hundreds of rate-limited SEC and Yahoo requests under your `SEC_USER_AGENT` |
| Reports → email | Builds and previews; does not send | An open form wired to your SMTP account is an open relay |
| AI panels | Cached analysis only | Each panel is a billable call on your key |
| Backtest start date | Capped (default 3 years) | Walk-forward re-scores at every rebalance; an unbounded window stalls a shared two-core container |

Screening, Compare, Intel, Builder, Methodology and the Data tab's coverage
views are untouched — they only read the database.

---

## Running it yourself

```bash
docker build -t quant-terminal .
docker run -p 7860:7860 -e DEMO_MODE=1 quant-terminal
```

Drop `-e DEMO_MODE=1` for a full-capability local container, and mount a volume
at `/home/app/trading/db` to keep the store across restarts.

## Notes for other hosts

- **Render / Fly.io**: same image. Attach a persistent volume over `db/`,
  `cache/` and `reports/out/`, and pin to a **single instance** — the factor
  cache, the HTTP cache and SQLite are all per-process, so a second replica
  would answer from a different cache.
- **Scheduled ingest**: only makes sense on a host with a durable volume and
  with `DEMO_MODE` off. `tools/daily_routine.sh` hardcodes a macOS interpreter
  path and would need that changed first.
