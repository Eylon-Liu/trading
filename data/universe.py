"""
Universe construction — filters or index presets, always reproducible.

The original app called `random.shuffle` and took the first N tickers, so the
same request produced a different universe on every run and cross-sectional
z-scores were not comparable between them. Here a universe is a declarative
spec that resolves deterministically and is snapshotted into the run record,
so re-running a past date reproduces it exactly.

Index membership resolves point-in-time (see data/members.py), so a universe
built for 2021 contains the names that were in the index in 2021.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd

import config
from core import db
from data import members, yahoo

log = logging.getLogger(__name__)


@dataclass
class UniverseSpec:
    """A named, serializable screen definition."""

    preset: str | None = None                   # SPY | QQQ | DIA | IWM | None
    sectors: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    exchanges: list[str] = field(default_factory=list)
    min_market_cap: float | None = config.MIN_MARKET_CAP
    max_market_cap: float | None = None
    min_dollar_adv: float | None = config.MIN_DOLLAR_ADV
    min_price: float | None = config.MIN_PRICE
    dividend_payers_only: bool = False
    profitable_only: bool = False
    exclude_adr: bool = False
    max_names: int | None = None                # cap by market-cap rank, never at random
    label: str = 'Custom screen'

    # ── serialization ────────────────────────────────────────────────
    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> 'UniverseSpec':
        return cls(**json.loads(blob))

    def describe(self) -> str:
        bits = []
        if self.preset:
            bits.append(f'{self.preset} members')
        if self.sectors:
            bits.append('sector in ' + '/'.join(self.sectors))
        if self.countries:
            bits.append('country in ' + '/'.join(self.countries))
        if self.min_market_cap:
            bits.append(f'mcap>{self.min_market_cap/1e9:.1f}B')
        if self.max_market_cap:
            bits.append(f'mcap<{self.max_market_cap/1e9:.1f}B')
        if self.min_dollar_adv:
            bits.append(f'ADV>${self.min_dollar_adv/1e6:.0f}M')
        if self.dividend_payers_only:
            bits.append('dividend payers')
        if self.profitable_only:
            bits.append('profitable')
        if self.exclude_adr:
            bits.append('no ADRs')
        if self.max_names:
            bits.append(f'top {self.max_names} by size')
        return ' AND '.join(bits) or 'all securities'

    # ── resolution ───────────────────────────────────────────────────
    def resolve(self, as_of: date | str | None = None,
                apply_liquidity: bool = True) -> list[str]:
        """
        Resolve to a deterministic ticker list as knowable on `as_of`.

        Order of operations matters: membership first (point-in-time), then
        descriptive filters, then liquidity, which needs price history.
        """
        as_of = pd.to_datetime(as_of or date.today()).date()

        # 1. starting population
        if self.preset:
            candidates = members.members_asof(self.preset, as_of)
            if not candidates:
                candidates = members.latest_members(self.preset)
        else:
            candidates = db.read_sql(
                'SELECT ticker FROM securities ORDER BY ticker')['ticker'].tolist()

        if not candidates:
            return []

        # 2. descriptive filters from the securities table
        meta = self._security_meta(candidates)
        if not meta.empty:
            if self.sectors:
                meta = meta[meta['sector'].isin(self.sectors)]
            if self.countries:
                meta = meta[meta['country'].isin(self.countries)]
            if self.exchanges:
                meta = meta[meta['exchange'].isin(self.exchanges)]
            if self.exclude_adr:
                meta = meta[meta['is_adr'].fillna(0) == 0]
            # Names we have no metadata for are kept only when no descriptive
            # filter was requested — otherwise we would be asserting a fact we
            # do not have.
            if self.sectors or self.countries or self.exchanges or self.exclude_adr:
                candidates = meta['ticker'].tolist()
            else:
                candidates = sorted(set(candidates))

        if not candidates:
            return []

        # 3. size and fundamentals
        prof = yahoo.profile_asof(candidates, as_of)
        if not prof.empty:
            # Filter only on values we actually have. Absence of a snapshot is
            # not evidence of a small company, and treating it as one is
            # catastrophic here: `fillna(0)` meant that the moment
            # profile_snapshots held any rows at all, every name *without* one
            # scored zero market cap and was dropped. Nine stored snapshots
            # cut an S&P 500 screen to nine names.
            keep = pd.Series(True, index=prof.index)

            def _within(col: str, lo=None, hi=None) -> pd.Series:
                if col not in prof.columns:
                    return pd.Series(True, index=prof.index)
                v = pd.to_numeric(prof[col], errors='coerce')
                ok = pd.Series(True, index=prof.index)
                if lo is not None:
                    ok &= v >= lo
                if hi is not None:
                    ok &= v <= hi
                return v.isna() | ok          # unknown passes

            if self.min_market_cap or self.max_market_cap:
                keep &= _within('market_cap', self.min_market_cap,
                                self.max_market_cap)
            if self.dividend_payers_only:
                keep &= _within('dividend_yield', lo=1e-9)
            if self.profitable_only:
                keep &= _within('trailing_eps', lo=1e-9)

            excluded = set(prof.index[~keep])
            candidates = [t for t in candidates if t not in excluded]

        if not candidates:
            return []

        # 4. tradability — a signal you cannot fill is not a signal
        if apply_liquidity and (self.min_dollar_adv or self.min_price):
            candidates = self._liquidity_filter(candidates, as_of)

        # 5. deterministic cap by size rank (the fix for random.shuffle)
        if self.max_names and len(candidates) > self.max_names:
            prof = yahoo.profile_asof(candidates, as_of)
            if not prof.empty and 'market_cap' in prof:
                ranked = (prof['market_cap'].fillna(0)
                          .sort_values(ascending=False).index.tolist())
                ranked = [t for t in ranked if t in set(candidates)]
                tail = sorted(set(candidates) - set(ranked))
                candidates = (ranked + tail)[:self.max_names]
            else:
                candidates = sorted(candidates)[:self.max_names]

        return sorted(candidates)

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _security_meta(tickers: list[str]) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        ph = ','.join(f':t{i}' for i in range(len(tickers)))
        params = {f't{i}': t for i, t in enumerate(tickers)}
        return db.read_sql(
            f'SELECT ticker, sector, industry, country, exchange, is_adr '
            f'FROM securities WHERE ticker IN ({ph})', params)

    def _liquidity_filter(self, tickers: list[str], as_of: date) -> list[str]:
        px = yahoo.latest_prices(tickers, as_of)
        adv = yahoo.dollar_adv(tickers, as_of)
        if px.empty:
            return tickers            # no price history yet; do not over-filter

        keep = []
        for t in tickers:
            if self.min_price:
                p = px.get(t)
                if p is None or pd.isna(p) or p < self.min_price:
                    continue
            if self.min_dollar_adv:
                a = adv.get(t)
                if a is None or pd.isna(a) or a < self.min_dollar_adv:
                    continue
            keep.append(t)
        return keep


# ─────────────────────────────────────────────
# PRESETS
# ─────────────────────────────────────────────

PRESETS: dict[str, UniverseSpec] = {
    'spy': UniverseSpec(preset='SPY', label='S&P 500 members'),
    'qqq': UniverseSpec(preset='QQQ', label='NASDAQ 100 members'),
    'dia': UniverseSpec(preset='DIA', label='Dow Jones 30'),
    'iwm': UniverseSpec(preset='IWM', label='Russell 2000 (sample)'),
    'us_large_cap': UniverseSpec(
        preset='SPY', min_market_cap=10e9, label='US large cap (>$10B)'),
    'us_mid_cap': UniverseSpec(
        preset='SPY', min_market_cap=2e9, max_market_cap=10e9,
        label='US mid cap ($2-10B)'),
    'dividend_quality': UniverseSpec(
        preset='SPY', dividend_payers_only=True, profitable_only=True,
        min_market_cap=5e9, label='Dividend payers, profitable'),
    'tech_growth': UniverseSpec(
        preset='SPY', sectors=['Information Technology', 'Communication Services'],
        min_market_cap=5e9, label='Tech & Comms >$5B'),
    'healthcare': UniverseSpec(
        preset='SPY', sectors=['Health Care'], label='Health Care'),
    'liquid_tradables': UniverseSpec(
        preset='SPY', min_dollar_adv=25e6, min_price=10.0,
        label='Highly liquid (ADV>$25M)'),
}


def resolve_preset(name: str, as_of: date | str | None = None) -> list[str]:
    spec = PRESETS.get(name.lower())
    if spec is None:
        raise KeyError(f'unknown preset {name!r}; have {sorted(PRESETS)}')
    return spec.resolve(as_of)


def parse_spec(expr: str) -> UniverseSpec:
    """
    Build a spec from a compact CLI expression.

        "SPY"
        "sector=Information Technology,mcap>10B,adv>5M"
        "SPY,sector=Health Care,mcap>2B,top=50"

    Suffixes B/M/K are honoured on numeric bounds.
    """
    spec = UniverseSpec(label=expr)
    for partition in [p.strip() for p in expr.split(',') if p.strip()]:
        low = partition.lower()

        if low in PRESETS:
            base = PRESETS[low]
            spec.preset = base.preset
            spec.sectors = list(base.sectors)
            continue
        if partition.upper() in config.INDEX_OPTIONS:
            spec.preset = partition.upper()
            continue

        if '=' in partition:
            key, val = (s.strip() for s in partition.split('=', 1))
            key = key.lower()
            if key == 'sector':
                spec.sectors.append(config.normalize_sector(val) if
                                    config.normalize_sector(val) != 'Unknown' else val)
            elif key == 'country':
                spec.countries.append(val)
            elif key == 'exchange':
                spec.exchanges.append(val)
            elif key in ('top', 'max', 'limit'):
                spec.max_names = int(val)
            elif key in ('dividend', 'dividends'):
                spec.dividend_payers_only = val.lower() in ('1', 'true', 'yes')
            elif key == 'profitable':
                spec.profitable_only = val.lower() in ('1', 'true', 'yes')
            continue

        for op, attr_lo, attr_hi in (('>', 'min', 'max'), ('<', 'max', 'min')):
            if op in partition:
                key, val = (s.strip() for s in partition.split(op, 1))
                key = key.lower()
                num = _parse_number(val)
                if num is None:
                    break
                if key in ('mcap', 'marketcap', 'market_cap'):
                    setattr(spec, f'{attr_lo}_market_cap', num)
                elif key in ('adv', 'volume', 'dollar_adv'):
                    if attr_lo == 'min':
                        spec.min_dollar_adv = num
                elif key == 'price':
                    if attr_lo == 'min':
                        spec.min_price = num
                break

    return spec


def _parse_number(val: str) -> float | None:
    v = val.strip().upper().replace('$', '').replace(',', '')
    mult = 1.0
    if v.endswith('B'):
        mult, v = 1e9, v[:-1]
    elif v.endswith('M'):
        mult, v = 1e6, v[:-1]
    elif v.endswith('K'):
        mult, v = 1e3, v[:-1]
    try:
        return float(v) * mult
    except ValueError:
        return None


def sector_breakdown(tickers: list[str]) -> pd.DataFrame:
    """Sector counts for a resolved universe."""
    if not tickers:
        return pd.DataFrame(columns=['sector', 'count'])
    ph = ','.join(f':t{i}' for i in range(len(tickers)))
    params = {f't{i}': t for i, t in enumerate(tickers)}
    df = db.read_sql(
        f'SELECT COALESCE(sector, \'Unknown\') AS sector, COUNT(*) AS count '
        f'FROM securities WHERE ticker IN ({ph}) GROUP BY sector '
        f'ORDER BY count DESC', params)
    return df
