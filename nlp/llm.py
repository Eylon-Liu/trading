"""
Optional AI synthesis — the prompts.

Everything in this module is an *upgrade path*, never a dependency. The local
Loughran-McDonald sentiment and rule-based extraction in nlp/sentiment.py and
nlp/extract.py do the real work with no key and no network; this layer adds
narrative interpretation on top when an API key is present in .env. Transport
lives in nlp/providers.py and speaks either Gemini or Claude.

Design rules that follow from that:

  * Every entry point returns None when no key is configured. Callers render
    the local output instead — the app must never degrade to a blank panel.
  * The model is given *computed numbers*, never raw price series. It
    interprets what the quant engine produced; it does not do the arithmetic,
    because a language model is the wrong tool for a TTM sum and the whole
    point of the SEC pipeline is that the figures are auditable.
  * Nothing it returns is stored as a factor or fed into a score. It is
    commentary attached to a run, so a hallucination can mislead a reader but
    cannot silently corrupt a backtest.

Neither a Claude Pro nor a Gemini consumer subscription includes API access;
both developer APIs are billed separately.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd

from nlp import providers

log = logging.getLogger(__name__)

MAX_TOKENS = 8000

# Re-exported so callers keep importing one module.
available = providers.available
status = providers.status


# ─────────────────────────────────────────────
# CORE CALL
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a sell-side-quality equity research analyst embedded in a quantitative \
screening tool. You are given figures that a point-in-time factor engine has \
already computed from SEC filings and market data.

Interpret those figures. Do not recompute them, do not introduce numbers that \
are not in the input, and do not draw on remembered facts about these companies \
— the whole value of this pipeline is that every number is auditable back to a \
filing, and a figure you supply from memory breaks that.

Where the data is thin or a metric is missing, say so plainly rather than \
filling the gap. Distinguish what the numbers show from what they merely \
suggest. Write for a professional investor: direct, specific, no hedging \
filler and no restating the input back.

This is research commentary, not investment advice, and the reader knows that \
— do not append disclaimers; the application already displays one."""


def _complete(user_prompt: str, system: str = SYSTEM_PROMPT,
              max_tokens: int = MAX_TOKENS) -> str | None:
    """One call on the configured backend. Text, or None on any failure."""
    return providers.complete(user_prompt, system, max_tokens)


# ─────────────────────────────────────────────
# NEWS SYNTHESIS
# ─────────────────────────────────────────────

def synthesize_news(ticker: str, company: str, articles: pd.DataFrame,
                    direction: dict | None = None) -> str | None:
    """
    Turn a pile of headlines into a thesis-relevant read.

    The local engine already extracted events and scored sentiment; this
    reconciles them into prose and says what actually changed.
    """
    if articles is None or articles.empty:
        return None

    lines = []
    for _i, r in articles.head(20).iterrows():
        when = pd.to_datetime(r['published']).strftime('%Y-%m-%d')
        events = f" [events: {r['events']}]" if r.get('events') else ''
        lines.append(
            f"- {when} ({r.get('source', '?')}) {r['title']}"
            f"{events} [local sentiment: {r.get('sentiment', 0):+.2f}]")

    extracted = ''
    if direction:
        extracted = (
            f"\nRule-based extraction already found:\n"
            f"  guidance: {direction.get('guidance') or 'none detected'}\n"
            f"  analyst actions: {direction.get('analyst_actions') or 'none'}\n"
            f"  event counts: {direction.get('top_events') or {}}\n")

    prompt = f"""Recent coverage of {company} ({ticker}), newest first:

{chr(10).join(lines)}
{extracted}
Write a briefing with exactly these four sections, using `## ` headings:

## What changed
The developments that would alter how someone values this company. Skip
routine coverage and price commentary.

## Forward direction
What management or the filings imply about the next few quarters — guidance,
capital allocation, strategy, leadership. Say "no clear signal" if there isn't one.

## Risks
Concrete risks visible in this coverage, not generic market risk.

## Read
Two or three sentences: is the news flow constructive, deteriorating, or noise?
Note explicitly if the coverage is too thin to judge."""

    return _complete(prompt)


# ─────────────────────────────────────────────
# STRATEGY / SCREEN COMMENTARY
# ─────────────────────────────────────────────

def explain_screen(strategy_name: str, horizon: str, thesis: str,
                   scores: pd.DataFrame, universe_desc: str,
                   as_of: date) -> str | None:
    """Commentary on why a screen produced the names it produced."""
    if scores is None or scores.empty:
        return None

    cols = [c for c in ('rank', 'ticker', 'sector', 'composite', 'signal',
                        'ROIC', 'EARNINGS_YIELD', 'PIOTROSKI_F', 'MOM_12_1',
                        'VOL_1Y', 'DEBT_TO_EQUITY')
            if c in scores.columns or c == 'ticker']
    # Callers hand this frame over in two shapes: the engine returns tickers in
    # an unnamed index, while the UI round-trips through a dcc.Store and sets
    # `ticker` as both index and column. Blindly reset_index()-ing the second
    # raises "cannot insert ticker, already exists".
    table = scores.head(15)
    if 'ticker' in table.columns:
        table = table.reset_index(drop=True)
    else:
        table = table.reset_index()
        if 'index' in table.columns:
            table = table.rename(columns={'index': 'ticker'})
    table = table[[c for c in cols if c in table.columns]]

    sectors = (scores['sector'].value_counts().to_dict()
               if 'sector' in scores else {})

    prompt = f"""Screen results.

Strategy: {strategy_name} ({horizon}-term horizon)
Stated thesis: {thesis}
Universe: {universe_desc}
As of: {as_of}
Sector distribution of the results: {sectors}

Top names (composite score is a weighted blend of sector-relative z-scores):

{table.round(3).to_string(index=False)}

Write three sections with `## ` headings:

## What this screen selected for
The characteristics these names share, read off the factor values above.

## Concentration and risk
Sector or factor concentration visible here, and what single bet the reader
would be taking by owning the whole list.

## What to check before acting
The two or three things these factor values cannot tell you, which a human
would need to verify. Be specific to these names."""

    return _complete(prompt)


def explain_backtest(strategy_name: str, metrics: dict) -> str | None:
    """
    Interpret backtest statistics, with rank IC given its proper weight.

    The prompt deliberately pushes on IC because that is the statistic most
    likely to contradict a flattering equity curve, and the one a reader is
    most likely to skip past.
    """
    if not metrics:
        return None

    shown = {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in metrics.items() if not isinstance(v, (list, dict))}
    mix = metrics.get('exit_mix')

    prompt = f"""Walk-forward backtest of "{strategy_name}".

Metrics:
{json.dumps(shown, indent=2, default=str)}
{f'Exit mix: {mix}' if mix else ''}

Write three short sections with `## ` headings:

## Did it work
Return and risk together, versus the benchmark where present.

## Did the ranking predict
Interpret mean rank IC and its hit rate. An IC near zero means the ranking had
no predictive power and any outperformance came from a handful of positions
rather than from the signal — say so directly if that is what the numbers show,
regardless of how good the returns look.

## Caveats
What this test cannot establish. Consider universe breadth, the number of
rebalances, and survivorship or cost assumptions."""

    return _complete(prompt, max_tokens=4000)


def compare_runs_narrative(strategy_name: str, then: date, now: date,
                           entered: list[str], dropped: list[str],
                           risers: list[tuple], fallers: list[tuple]
                           ) -> str | None:
    """Explain what moved between two stored runs."""
    if not any((entered, dropped, risers, fallers)):
        return None

    prompt = f"""Ranking changes for "{strategy_name}" between {then} and {now}.

Entered the list: {', '.join(entered[:15]) or 'none'}
Dropped out: {', '.join(dropped[:15]) or 'none'}
Biggest rank gains: {risers[:10]}
Biggest rank falls: {fallers[:10]}

In under 200 words: what does this rotation suggest about which factors are
being rewarded, and is this normal churn or a regime shift? Say if the
movement is too small to read anything into."""

    return _complete(prompt, max_tokens=2000)


def policy_impact(documents: pd.DataFrame, sectors: list[str]) -> str | None:
    """Read recent policy activity for sector impact."""
    if documents is None or documents.empty:
        return None

    lines = [
        f"- [{pd.to_datetime(r['published']).date()}] {r['doc_type']}: "
        f"{r['title'][:150]} (agencies: {(r.get('agencies') or '')[:70]}; "
        f"themes: {r.get('themes') or 'none'})"
        for _i, r in documents.head(20).iterrows()
    ]

    prompt = f"""Recent US federal rules and executive actions, mapped to these
sectors: {', '.join(sectors) or 'all'}.

{chr(10).join(lines)}

In under 250 words: which of these plausibly affects listed equities, in which
direction, and over what horizon? Ignore items that are procedural or
immaterial to public markets — most of them will be. Name the sector for each
item you flag."""

    return _complete(prompt, max_tokens=3000)
