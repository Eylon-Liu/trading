# Do the strategies beat SPY?

**Short answer: none of them is proven to.** Measured on the S&P 500 from
January 2020 to November 2026, with holds of three and six months:

- With **six-month holds**, in a walk-forward test with trading costs,
  Buffett Quality, Total Shareholder Yield, the Multi-Strategy Consensus and
  Momentum + Trend finished a little ahead of SPY (0.4 to 1.4% a year from
  the same start dates) and the new Balanced blend finished slightly
  behind. None of those margins is statistically distinguishable from zero.
- With **three-month holds** every one of them did 0.6–2.4% a year worse
  than with six-month holds, and three of the five finished behind SPY.
- Which strategy looked best depended on the period. Value screens won in
  2020–2023 and lagged afterwards; momentum screens did the reverse.
- The result moves by about ±3% a year depending on nothing more than which
  months the portfolio is rebalanced in — as large as any edge measured.

What the measurements *did* establish, and what changed in the code because
of them, is at the end.

---

## How it was measured

**Universe.** S&P 500 members as of each date, not today's list. Two gaps in
the historical list were closed first, because both bias a backtest:

- Companies that changed symbol (FB→META, ANTM→ELV, UTX→RTX, BK→BNY and two
  dozen more) were missing from every date before their rename.
- Companies later removed from the index but still trading had no prices
  stored. Loading them cut the share of the historical index with no data
  from 11–21% a year to 0.5–8.5%.

What is still missing is what the free price source no longer carries:
companies that were acquired, and three banks that failed in 2023 (SIVB,
SBNY, FRC). A strategy that would have held one of those banks looks better
here than it was.

**No look-ahead.** Every factor uses only filings and prices dated on or
before the ranking date, and only that date's index members are ranked. (A
bug that let companies be ranked years before they joined the index was
found and fixed during this work.)

**Two methods, because they answer different questions.**

1. *Monthly cohorts.* At each of 82 month-ends, take a strategy's top 20
   names (at most 30% from one sector), hold them 3 or 6 months, compare
   with SPY over the same window. Many overlapping samples, so it shows the
   average outcome of "buy the list on a random month" — but it ignores
   costs and compounding.
2. *Walk-forward.* One portfolio carried through the whole period:
   rebalanced on calendar half-year or quarter ends, filled at the next
   day's close, 15 basis points per trade each way, marked to market daily.
   One path, so it is what an account would actually have done — and it is
   sensitive to when that path starts and ends.

**Chosen on one period, checked on another.** Design choices were made on
ranking dates up to June 2023 ("first period") and then looked at once on
dates from January 2024 ("second period").

---

## Results

### Walk-forward, with costs (the realistic test)

Annual return minus SPY's, to 9 November 2026, top 20 names. The first two
columns start in January, February and March 2020 (averaged); the third
adds starts in April, May and June 2020 and shows the range over all six.

| Strategy | Rebalanced quarterly | Held six months | Six months, all six starts | Worst drawdown |
|---|---|---|---|---|
| Buffett Quality | +0.8% | **+1.4%** | +3.6% (+0.9 to +7.1) | −50% |
| Total Shareholder Yield | 0.0% | **+0.8%** | +2.5% (+0.1 to +4.8) | −42% |
| Momentum + Trend | −1.6% | **+0.8%** | +0.7% (0.0 to +1.2) | −37% |
| Multi-Strategy Consensus | −1.9% | **+0.4%** | +1.2% (+0.2 to +2.2) | −42% |
| Balanced (new) | −3.2% | **−0.8%** | −0.3% (−2.0 to +0.9) | −39% |

SPY itself returned 14.8% a year from January 2020, with a worst drawdown
of −34%. Every 20-name portfolio fell further than SPY in the March 2020
crash.

The high end of the Buffett and Shareholder Yield ranges comes from starting
in April–June 2020, straight after the crash, when cheap, beaten-down
companies rebounded hardest.

### Monthly cohorts (average outcome, before costs)

Annualised return minus SPY's over the hold; "ahead" is the share of
monthly cohorts that beat SPY.

| Strategy | 6-month hold: first period | second period | whole period | ahead | 3-month hold: whole period |
|---|---|---|---|---|---|
| Balanced (new) | +2.0% | +8.2% | +4.9% | 64% | +2.5% |
| Multi-Strategy Consensus | +0.2% | +9.0% | +4.4% | 62% | +1.7% |
| Buffett Quality | +7.3% | +0.8% | +4.3% | 63% | +3.0% |
| Total Shareholder Yield | +7.6% | −2.6% | +3.2% | 51% | +2.0% |
| Momentum + Trend | −5.4% | +18.1% | +5.8% | 53% | +3.6% |
| 52-Week Breakout | −3.6% | +16.7% | +5.1% | 54% | +2.1% |
| Post-Earnings Drift | −3.4% | +17.7% | +4.6% | 50% | +1.8% |
| Style Factors | −8.0% | +20.6% | +4.4% | 58% | +3.1% |
| Low-Vol Relative Strength | −5.5% | +12.9% | +2.7% | 51% | +2.3% |
| Quality Momentum | −1.3% | +4.1% | +1.7% | 51% | +0.3% |
| Pullback in Uptrend | +1.8% | −4.5% | −0.6% | 49% | −3.0% |
| Quality Mean Reversion | +5.0% | −9.2% | −0.9% | 48% | −0.8% |

The largest t-statistic in this table is 1.2 (Balanced, six months). By the
usual standard of 2, nothing here is established.

The cohort averages are more flattering than the walk-forward for three
reasons: no costs, simple rather than compounded averaging, and the cohort
windows end before October–November 2026, when Balanced lost 5.9% in six
weeks while SPY lost 1.3%.

### By year (6-month cohorts, annualised, minus SPY)

| Strategy | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|
| Balanced (new) | +3.7 | +7.4 | −0.6 | +1.7 | +6.7 | +11.7 | +2.0 |
| Multi-Strategy Consensus | −1.8 | +7.8 | −0.1 | +0.9 | +7.0 | +12.3 | +5.2 |
| Buffett Quality | +27.5 | +9.2 | −4.6 | −6.9 | −2.5 | +3.4 | +2.4 |
| Total Shareholder Yield | +28.7 | +0.3 | +1.3 | −3.7 | +4.5 | −5.9 | −14.4 |
| Momentum + Trend | −6.0 | −6.0 | −3.3 | +9.7 | +10.2 | +26.2 | +17.3 |
| Post-Earnings Drift | +4.9 | −10.3 | −3.1 | −3.6 | +1.7 | +30.6 | +26.9 |
| Quality Mean Reversion | +15.2 | +7.7 | +8.0 | −9.1 | −6.9 | −11.3 | −9.3 |
| *Equal-weight index minus SPY* | *+7.0* | *+0.6* | *+0.7* | *−9.6* | *−5.0* | *−4.6* | *−4.8* |

---

## What the data says

**1. Hold six months, not three.** Eleven of the twelve strategies did
better with six-month holds than three-month holds in the cohorts (the
twelfth was unchanged), and all five tested walk-forward did too (by 0.6 to
2.4% a year from the same start dates). Rotating quarterly costs more and sells names
whose trend or re-rating has not finished.

**2. Value and momentum took turns.** Their excess returns correlate about
−0.3. Anyone who chose "the strategy that works" in 2023 from the previous
three years would have chosen value, just before momentum's run; and the
reverse in late 2026.

**3. Since 2023 the benchmark itself has been hard to beat with an
equal-weighted list.** SPY is weighted by size; the average S&P 500 stock
trailed it by 5–10% a year from 2023 as a few of the largest companies led.
Any 20-name equal-weight shortlist starts that far behind unless it happens
to hold those companies.

**4. Raw momentum concentrates.** At the end of March 2026, thirteen of the
twenty highest momentum scores were semiconductor and memory names. That
cohort returned +77% in three months (several names rose 100–250%) and the
next lost 13%. A result like that is a sector bet.

**5. Tight stops were the most expensive mistake found.** Run as trades with
the previous plan (stop two daily ATRs away, 60-day limit), Balanced
returned 3.6% a year and Momentum + Trend 4.2%, against 16.6% for SPY:
245–252 of roughly 450 trades ended at the stop. With the plan now used
(three ATRs or the two-month low, six-month limit) the same strategies
returned 12.3% and 7.9%, with worst drawdowns of −15% and −18% against
−25% for SPY over the same window. Still behind simply holding the list —
stops lower the drawdown and the return together.

**6. Insider buying showed nothing on large caps.** With two years of Form 4
filings loaded for 452 S&P 500 companies (Dec 2024–Oct 2026), names with
net insider buying did no better than the rest over 3 or 6 months (rank
correlation −2 to −3, positive on 17–38% of dates). The effect is
documented mainly in smaller companies. It stays one vote in the consensus;
it has not earned more.

**7. The short-horizon setups are not six-month strategies.** Pullback and
Mean Reversion were level with or behind SPY over 3–6 month holds, and
behind it every year since 2023.

---

## What changed in the code because of this

| Change | Reason |
|---|---|
| Both horizons rebalance every six months (was quarterly / monthly) | Finding 1 |
| Mid-term now means 3–6 months; trade plan stop 3×ATR or two-month low, target 2R, six-month time limit | Finding 5 |
| A shortlist or backtest portfolio takes at most 30% of its names from one sector | Finding 4 |
| `consensus-test --compare` holds every member strategy the same way | It had been comparing exit rules (stops for some, none for others) as much as selection |
| New `balanced` strategy (value + momentum + quality in one ranking) | Finding 2 — offered as a steadier blend, not as an edge; see its numbers above |
| Ticker renames mapped; ranking restricted to that date's members | Backtest correctness |
| Pullback and Mean Reversion described as entry setups | Finding 7 |

Not changed, deliberately: strategy weights were not tuned to these
results. Seven years is one market regime and a half; fitting weights to it
would produce a better-looking table and a worse strategy.

---

## What this does not show

- **Seven years, mostly rising markets.** One crash (2020), one bear market
  (2022), a long mega-cap rally.
- **Acquired and failed companies are missing** (5–9% of the index in the
  earlier years), including three bank failures.
- **Rebalance timing.** Non-overlapping chains from the cohort data, started
  in each possible month, spread by 3–6% a year for the same strategy.
- **The AI earnings review** is forward-only and has no history to test.
- **Costs are modelled, not measured:** 15 basis points each way, no market
  impact, no taxes. Six-month holds replace roughly 100–165% of the
  portfolio a year; taxes on short-term gains would take more than the
  edges measured here.

## If the goal is to beat SPY

The evidence here supports a modest claim: these screens narrow 500
companies to 20 that have, on average, done about as well as SPY — a little
better when held six months — with larger drawdowns and a different mix of
risks. A reasonable way to use them is as a satellite next to an index
holding rather than instead of one, to prefer the consensus or Buffett
Quality (each ahead in all six six-month walk-forward runs, by a small
margin) over any single momentum screen, and to judge the live track record
on the Portfolio tab, which grades every stored ranking against what
actually happened next.

## Reproduce it

```bash
python tools/research/build_panel.py 2020-01-31 2026-10-31 tools/research/panel.pkl   # ~15 min
python tools/research/evaluate.py                                                    # cohort tables
python cli.py consensus-test --start 2020-01-02 --compare                            # walk-forward
```
