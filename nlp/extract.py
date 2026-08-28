"""
Event and forward-direction extraction from financial text.

Sentiment tells you the tone; this tells you what actually happened and what
management said about what comes next — which is the part that changes a
thesis. Extraction is rule-based and deterministic, so it costs nothing, runs
offline, and gives the same answer twice.

Note that 8-K item codes already provide this for filings (see
config.EIGHTK_ITEMS), and are strictly more reliable because the SEC assigns
them. This module covers news text, where no such structure exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─────────────────────────────────────────────
# PATTERNS
# ─────────────────────────────────────────────

# Order matters: the first matching direction wins, so "raised" is tested
# before the generic "guidance" mention.
# "price target" belongs to the sell side, not to management — a broker
# trimming its target is a different event from a company cutting guidance.
# A lookbehind is required rather than a lookahead: the disqualifying word
# sits *before* "target".
_SUBJECT = r'(?:guidance|outlook|forecast|estimate|(?<!price )target)'

GUIDANCE_PATTERNS = [
    ('raised', r'\b(rais(?:e|es|ed|ing)|lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|'
               r'increas(?:e|es|ed|ing)|hik(?:e|es|ed))\b[^.]{0,40}\b' + _SUBJECT),
    ('lowered', r'\b(cut(?:s|ting)?|lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|'
                r'slash(?:es|ed)?|trim(?:s|med|ming)?|scal(?:e|es|ed) back)\b'
                r'[^.]{0,40}\b' + _SUBJECT),
    ('withdrawn', r'\b(withdraw(?:s|n|ing)?|suspend(?:s|ed|ing)?|pull(?:s|ed)?)\b'
                  r'[^.]{0,40}\b(guidance|outlook|forecast)'),
    ('reaffirmed', r'\b(reaffirm(?:s|ed|ing)?|maintain(?:s|ed|ing)?|reiterat(?:e|es|ed|ing)|'
                   r'confirm(?:s|ed)?)\b[^.]{0,40}\b(guidance|outlook|forecast|target)'),
    ('initiated', r'\b(initiat(?:e|es|ed)|issu(?:e|es|ed)|provid(?:e|es|ed)|'
                  r'introduc(?:e|es|ed))\b[^.]{0,30}\b(guidance|outlook|forecast)'),
    ('beat', r'\b(beat(?:s)?|top(?:s|ped)?|exceed(?:s|ed)?|surpass(?:es|ed)?)\b'
             r'[^.]{0,40}\b(estimates?|expectations?|consensus|forecasts?)'),
    ('missed', r'\b(miss(?:es|ed)?|fell short|below)\b[^.]{0,40}\b'
               r'(estimates?|expectations?|consensus|forecasts?)'),
]

EVENT_PATTERNS = {
    'earnings': r'\b(earnings|quarterly results|q[1-4]\s|fiscal (?:first|second|third|fourth) quarter|reports? (?:results|earnings))\b',
    'ma': r'\b(acquisition|acquire[sd]?|acquiring|merger|merges?|takeover|'
          r'buyout|to be acquired|divest(?:s|ed|iture)?|spin[- ]?off|'
          r'sells? (?:its |the )?(?:unit|division|business|stake))\b',
    'buyback': r'\b(buyback|repurchase|share repurchase|tender offer)\b',
    'dividend': r'\b(dividend)\b',
    'dividend_raise': r'\b(rais(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|hik(?:e|es|ed))\b[^.]{0,30}\bdividend\b',
    'dividend_cut': r'\b(cut(?:s|ting)?|reduc(?:e|es|ed)|suspend(?:s|ed)?|eliminat(?:e|es|ed))\b[^.]{0,30}\bdividend\b',
    'management': r'\b(chief executive|ceo|cfo|chief financial|president|'
                  r'chairman|board member|steps? down|resign(?:s|ed|ation)?|'
                  r'appoint(?:s|ed|ment)?|nam(?:e|es|ed) as|succeed(?:s|ed)?)\b',
    'layoffs': r'\b(layoff|lay off|job cuts?|workforce reduction|restructur(?:e|es|ing)|'
               r'headcount reduction|redundanc(?:y|ies))\b',
    'legal': r'\b(lawsuit|litigation|settle(?:s|d|ment)?|court|judge|jury|'
             r'class action|allege[sd]?|complaint|subpoena|indictment)\b',
    'regulatory': r'\b(regulator|regulatory|sec (?:probe|investigation|inquiry)|'
                  r'antitrust|ftc|doj|investigat(?:e|es|ion)|fine[sd]?|penalt(?:y|ies)|'
                  r'approval|approved|fda|clearance)\b',
    'analyst': r'\b(upgrade[sd]?|downgrade[sd]?|price target|initiat(?:e|es|ed) coverage|'
               r'analyst|rating|overweight|underweight|outperform|buy rating|sell rating)\b',
    'product': r'\b(launch(?:es|ed)?|unveil(?:s|ed)?|introduc(?:e|es|ed)|'
               r'new product|rollout|releases?)\b',
    'guidance': r'\b(guidance|outlook|forecast)\b',
    'capacity': r'\b(capex|capital expenditure|expansion|new (?:plant|facility|factory)|'
                r'invest(?:s|ed|ment) of)\b',
    'partnership': r'\b(partnership|joint venture|collaborat(?:e|es|ion)|'
                   r'strategic alliance|agreement with)\b',
    'supply': r'\b(supply chain|shortage|inventory|backlog|bottleneck|'
              r'production (?:cut|halt|delay))\b',
}

ANALYST_DIRECTION = [
    ('upgrade', r'\b(upgrade[sd]?|rais(?:e|es|ed) (?:price )?target|'
                r'to (?:buy|overweight|outperform|strong.buy))\b'),
    ('downgrade', r'\b(downgrade[sd]?|'
                  r'(?:cut[s]?|lower(?:s|ed)?|reduc(?:e[sd]?)) (?:price )?target|'
                  r'to (?:sell|underweight|underperform)|'
                  r'(?:downgrade[sd]?|cut[s]?) .{0,30}'
                  r'(?:hold|neutral|equal.?weight|market.?perform|'
                  r'moderate.?buy|sector.?perform))\b'),
]

# Money and percentage figures worth capturing alongside a guidance change.
MONEY_RE = re.compile(
    r'\$\s?([\d,]+(?:\.\d+)?)\s?(billion|million|bn|mm|m|b|trillion)?\b', re.I)
PCT_RE = re.compile(r'([-+]?\d+(?:\.\d+)?)\s?(?:%|percent)\b', re.I)


@dataclass
class Extraction:
    """Structured read of one article."""

    events: list[str] = field(default_factory=list)
    guidance: str | None = None
    analyst_action: str | None = None
    figures: dict = field(default_factory=dict)
    direction_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'events': self.events, 'guidance': self.guidance,
            'analyst_action': self.analyst_action, 'figures': self.figures,
            'direction_notes': self.direction_notes,
        }

    @property
    def is_material(self) -> bool:
        """True when the article says something thesis-relevant."""
        material = {'earnings', 'ma', 'guidance', 'management', 'layoffs',
                    'buyback', 'dividend_raise', 'dividend_cut', 'legal',
                    'regulatory'}
        return bool(self.guidance or self.analyst_action
                    or material.intersection(self.events))


def extract(text: str) -> Extraction:
    """Pull events, guidance direction, analyst actions and figures from text."""
    out = Extraction()
    if not text:
        return out
    low = text.lower()

    for name, pattern in EVENT_PATTERNS.items():
        if re.search(pattern, low, re.I):
            out.events.append(name)

    # A specific dividend direction supersedes the generic tag.
    if 'dividend_raise' in out.events or 'dividend_cut' in out.events:
        out.events = [e for e in out.events if e != 'dividend']

    for direction, pattern in GUIDANCE_PATTERNS:
        if re.search(pattern, low, re.I):
            out.guidance = direction
            break

    for direction, pattern in ANALYST_DIRECTION:
        if re.search(pattern, low, re.I):
            out.analyst_action = direction
            break

    money = MONEY_RE.findall(text)
    if money:
        out.figures['amounts'] = [f'${a}{" " + (u or "")}'.strip()
                                  for a, u in money[:4]]
    pcts = PCT_RE.findall(text)
    if pcts:
        out.figures['percentages'] = [f'{p}%' for p in pcts[:4]]

    out.direction_notes = _direction_notes(out)
    return out


def _direction_notes(ex: Extraction) -> list[str]:
    """Plain-language reading of what the article implies for the future."""
    notes = []
    g = {
        'raised': 'Management raised forward guidance',
        'lowered': 'Management cut forward guidance',
        'withdrawn': 'Guidance withdrawn — visibility has deteriorated',
        'reaffirmed': 'Guidance reaffirmed — outlook unchanged',
        'initiated': 'New guidance issued',
        'beat': 'Results came in ahead of expectations',
        'missed': 'Results came in below expectations',
    }
    if ex.guidance and ex.guidance in g:
        notes.append(g[ex.guidance])

    e = set(ex.events)
    if 'buyback' in e:
        notes.append('Capital returning to shareholders via buyback')
    if 'dividend_raise' in e:
        notes.append('Dividend increased — confidence in cash generation')
    if 'dividend_cut' in e:
        notes.append('Dividend cut — cash flow under pressure')
    if 'ma' in e:
        notes.append('Corporate structure changing (M&A or divestiture)')
    if 'management' in e:
        notes.append('Leadership change — strategy may shift')
    if 'layoffs' in e:
        notes.append('Cost restructuring underway')
    if 'capacity' in e:
        notes.append('Capacity or capex expansion signalled')
    if 'regulatory' in e:
        notes.append('Regulatory development in play')
    if 'legal' in e:
        notes.append('Legal overhang')
    if 'supply' in e:
        notes.append('Supply-chain or inventory factor cited')
    if ex.analyst_action == 'upgrade':
        notes.append('Sell-side upgrade')
    elif ex.analyst_action == 'downgrade':
        notes.append('Sell-side downgrade')
    return notes


def aggregate(extractions: list[Extraction]) -> dict:
    """Roll several articles into one per-ticker view."""
    if not extractions:
        return {}

    from collections import Counter
    events = Counter(e for ex in extractions for e in ex.events)
    guidance = Counter(ex.guidance for ex in extractions if ex.guidance)
    analyst = Counter({'upgrade': 0, 'downgrade': 0})
    analyst.update(ex.analyst_action for ex in extractions if ex.analyst_action)
    notes: list[str] = []
    for ex in extractions:
        for n in ex.direction_notes:
            if n not in notes:
                notes.append(n)

    return {
        'top_events': dict(events.most_common(6)),
        'guidance': dict(guidance),
        'analyst_actions': dict(analyst),
        'direction_notes': notes[:8],
        'material_count': sum(1 for ex in extractions if ex.is_material),
        'n_articles': len(extractions),
    }
