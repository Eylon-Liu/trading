"""
Financial sentiment — Loughran-McDonald first, VADER as a secondary read.

General-purpose sentiment tools misread financial text badly. "Liability",
"cost", "expense", "depreciation" and "capital" all score negative in everyday
lexicons and are entirely neutral in a filing. Loughran & McDonald built their
word lists specifically from 10-Ks for this reason, and their categories
(negative, positive, uncertainty, litigious, constraining) map onto what
actually moves a thesis.

A representative subset of the dictionary ships inline so the module works
offline with no downloads and no API key. VADER is consulted when available
and blended at a lower weight, because it reads headline tone — which the
LM lists, built for filings, deliberately ignore.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LOUGHRAN-MCDONALD WORD LISTS (representative subset)
# ─────────────────────────────────────────────

LM_NEGATIVE = {
    'abandon', 'abandoned', 'abandoning', 'abandonment', 'adverse', 'adversely',
    'adversity', 'against', 'aggravate', 'alleged', 'allegation', 'annul',
    'anomalies', 'anomaly', 'arrears', 'assault', 'attrition', 'bad',
    'bail', 'bankrupt', 'bankruptcy', 'bans', 'barred', 'breach', 'breaches',
    'bribery', 'burden', 'burdened', 'cancel', 'cancellation', 'cancelled',
    'cease', 'ceased', 'challenge', 'challenged', 'challenges', 'closed',
    'closing', 'closure', 'collapse', 'collusion', 'complaint', 'complaints',
    'concern', 'concerns', 'confiscated', 'conflict', 'contraction',
    'convicted', 'corrected', 'correction', 'costly', 'counterfeit', 'crisis',
    'critical', 'criticism', 'curtail', 'curtailed', 'cut', 'cutback',
    'damage', 'damages', 'decline', 'declined', 'declines', 'declining',
    'default', 'defaulted', 'defect', 'defective', 'deficiency', 'deficit',
    'delay', 'delayed', 'delays', 'delinquent', 'demolish', 'denied', 'deny',
    'deteriorate', 'deteriorated', 'deterioration', 'devalue', 'difficult',
    'difficulty', 'diminish', 'diminished', 'disagree', 'disagreement',
    'disappointing', 'disappointment', 'disaster', 'disclaim', 'discontinue',
    'discontinued', 'dismissal', 'dispute', 'disputed', 'disruption',
    'downgrade', 'downgraded', 'downturn', 'drop', 'dropped', 'erosion',
    'error', 'errors', 'failed', 'failure', 'fails', 'fine', 'fined', 'fired',
    'flaw', 'forfeit', 'fraud', 'fraudulent', 'guilty', 'halt', 'halted',
    'harm', 'harmed', 'hazard', 'hurt', 'illegal', 'impair', 'impaired',
    'impairment', 'improper', 'inability', 'inadequate', 'incident',
    'ineffective', 'inefficiency', 'injunction', 'injury', 'insolvency',
    'insufficient', 'interruption', 'investigation', 'lag', 'lagged',
    'lawsuit', 'lawsuits', 'layoff', 'layoffs', 'liquidate', 'liquidation',
    'litigation', 'lose', 'loses', 'losing', 'loss', 'losses', 'lost',
    'malfunction', 'misconduct', 'mislead', 'misleading', 'mismanagement',
    'misstatement', 'negative', 'negatively', 'neglect', 'nonperforming',
    'obsolete', 'omission', 'overrun', 'penalty', 'penalties', 'plummet',
    'poor', 'postpone', 'postponed', 'problem', 'problems', 'prosecution',
    'protest', 'question', 'questioned', 'recall', 'recalled', 'recession',
    'reduction', 'refuse', 'refused', 'reject', 'rejected', 'restate',
    'restated', 'restatement', 'restructuring', 'retaliate', 'sacrifice',
    'sanction', 'sanctions', 'scrutiny', 'seizure', 'sever', 'severe',
    'shortage', 'shortfall', 'shrink', 'shut', 'shutdown', 'slow', 'slowdown',
    'slowed', 'slower', 'sluggish', 'strain', 'stress', 'subpoena', 'sue',
    'sued', 'suffer', 'suffered', 'suspend', 'suspended', 'suspension',
    'terminate', 'terminated', 'termination', 'threat', 'threaten', 'tumble',
    'unable', 'uncollectible', 'uncompetitive', 'underperform',
    'underperformance', 'underutilized', 'undesirable', 'unfavorable',
    'unforeseen', 'unfortunately', 'unlawful', 'unpaid', 'unprofitable',
    'unresolved', 'unsatisfactory', 'unsuccessful', 'violate', 'violated',
    'violation', 'volatile', 'volatility', 'weak', 'weaken', 'weakened',
    'weakness', 'worse', 'worsen', 'worsened', 'writedown', 'writeoff', 'wrong',
}

LM_POSITIVE = {
    'able', 'abundance', 'accomplish', 'accomplished', 'achieve', 'achieved',
    'achievement', 'achievements', 'advance', 'advanced', 'advantage',
    'advantageous', 'advantages', 'alliance', 'attain', 'attained',
    'attractive', 'beautiful', 'beneficial', 'benefit', 'benefited',
    'benefits', 'best', 'better', 'boost', 'boosted', 'breakthrough',
    'brilliant', 'collaborate', 'collaboration', 'compelling', 'complement',
    'conducive', 'confident', 'constructive', 'courteous', 'creative',
    'delight', 'delighted', 'dependable', 'desirable', 'despite', 'diligent',
    'distinction', 'distinctive', 'dream', 'easier', 'easily', 'efficiency',
    'efficient', 'empower', 'enable', 'enabled', 'enables', 'encouraged',
    'encouraging', 'enhance', 'enhanced', 'enhancement', 'enjoy', 'enjoyed',
    'enthusiasm', 'excellence', 'excellent', 'exceptional', 'excited',
    'exclusive', 'exemplary', 'fantastic', 'favorable', 'favorably', 'gain',
    'gained', 'gaining', 'gains', 'good', 'great', 'greater', 'greatest',
    'growth', 'happiness', 'happy', 'highest', 'ideal', 'impress',
    'impressive', 'improve', 'improved', 'improvement', 'improvements',
    'improving', 'incredible', 'influential', 'informative', 'ingenuity',
    'innovate', 'innovation', 'innovative', 'insightful', 'inspiration',
    'leadership', 'leading', 'loyal', 'lucrative', 'meritorious', 'opportunity',
    'opportunities', 'optimistic', 'outperform', 'outperformed', 'outstanding',
    'perfect', 'pleased', 'plentiful', 'popular', 'positive', 'positively',
    'premier', 'premium', 'prestige', 'proactive', 'proficiency', 'profitable',
    'progress', 'prospered', 'prosperity', 'prosperous', 'record', 'rebound',
    'recovery', 'regain', 'resilient', 'revolutionize', 'reward', 'rewarded',
    'satisfaction', 'satisfactory', 'satisfied', 'solid', 'stability',
    'stable', 'strength', 'strengthen', 'strengthened', 'strong', 'stronger',
    'strongest', 'succeed', 'success', 'successful', 'successfully',
    'superior', 'surpass', 'surpassed', 'thrive', 'transform', 'tremendous',
    'unmatched', 'unparalleled', 'unprecedented', 'upside', 'upturn',
    'valuable', 'versatile', 'vibrant', 'win', 'winner', 'winning', 'worthy',
}

LM_UNCERTAINTY = {
    'almost', 'ambiguity', 'ambiguous', 'anticipate', 'anticipated',
    'apparently', 'appear', 'appeared', 'appears', 'approximate',
    'approximately', 'assume', 'assumed', 'assumption', 'assumptions',
    'believe', 'believed', 'believes', 'clarification', 'conceivable',
    'contingency', 'contingent', 'could', 'depend', 'depended', 'depending',
    'depends', 'doubt', 'estimate', 'estimated', 'estimates', 'exposure',
    'fluctuate', 'fluctuation', 'fluctuations', 'hidden', 'imprecise',
    'improbable', 'indefinite', 'indeterminate', 'likelihood', 'may', 'maybe',
    'might', 'nearly', 'occasionally', 'pending', 'perhaps', 'possible',
    'possibly', 'precaution', 'predict', 'preliminary', 'presumably',
    'probable', 'probably', 'random', 'reassess', 'recalculate',
    'reconsider', 'revise', 'risk', 'risks', 'risky', 'roughly', 'seems',
    'seldom', 'somewhat', 'sometimes', 'speculate', 'speculation',
    'sporadic', 'sudden', 'suggest', 'tending', 'tentative', 'uncertain',
    'uncertainty', 'unclear', 'unconfirmed', 'undecided', 'undetermined',
    'unforeseeable', 'unknown', 'unpredictable', 'unproven', 'unsure',
    'unusual', 'vague', 'variability', 'variable', 'varied', 'vary', 'volatile',
}

LM_LITIGIOUS = {
    'adjudication', 'allegation', 'appeal', 'appellate', 'arbitration',
    'attorney', 'claimant', 'complaint', 'counterclaim', 'court', 'damages',
    'defendant', 'deposition', 'docket', 'indemnification', 'indictment',
    'injunction', 'judicial', 'juror', 'jury', 'law', 'lawsuit', 'legal',
    'legislation', 'liable', 'litigate', 'litigation', 'plaintiff',
    'prosecute', 'regulation', 'regulatory', 'settlement', 'statute',
    'subpoena', 'testimony', 'tort', 'tribunal', 'verdict',
}

LM_CONSTRAINING = {
    'commit', 'commitment', 'commitments', 'compel', 'compelled', 'comply',
    'compulsory', 'constrain', 'constrained', 'constraint', 'constraints',
    'covenant', 'covenants', 'encumber', 'encumbered', 'imposed', 'limit',
    'limitation', 'limitations', 'limited', 'mandate', 'mandatory',
    'obligate', 'obligated', 'obligation', 'obligations', 'prohibit',
    'prohibited', 'prohibition', 'require', 'required', 'requirement',
    'restrict', 'restricted', 'restriction', 'restrictions', 'stipulate',
}

# Negations flip the polarity of the next few words.
NEGATORS = {'not', 'no', 'never', 'none', 'neither', 'nor', 'cannot',
            "can't", "won't", "didn't", "doesn't", "isn't", "wasn't",
            'without', 'lacks', 'lacking', 'fails', 'failed', 'unable'}
NEGATION_WINDOW = 3

_TOKEN_RE = re.compile(r"[a-z']+")


@lru_cache(maxsize=1)
def _vader():
    """VADER analyzer if the lexicon is present; None otherwise."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:                              # noqa: BLE001
        try:
            import nltk
            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except Exception:                          # noqa: BLE001
            log.info('VADER unavailable; using Loughran-McDonald alone')
            return None


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or '').lower())


def lm_scores(text: str) -> dict[str, float]:
    """
    Category counts and a normalized polarity for a piece of financial text.

    Negation handling matters: "not profitable" and "profitable" should not
    both register as positive.
    """
    tokens = tokenize(text)
    if not tokens:
        return {'positive': 0, 'negative': 0, 'uncertainty': 0,
                'litigious': 0, 'constraining': 0, 'polarity': 0.0, 'n': 0}

    counts = {'positive': 0, 'negative': 0, 'uncertainty': 0,
              'litigious': 0, 'constraining': 0}
    negate_until = -1

    for i, tok in enumerate(tokens):
        if tok in NEGATORS:
            negate_until = i + NEGATION_WINDOW
            continue
        flipped = i <= negate_until

        if tok in LM_POSITIVE:
            counts['negative' if flipped else 'positive'] += 1
        elif tok in LM_NEGATIVE:
            counts['positive' if flipped else 'negative'] += 1
        if tok in LM_UNCERTAINTY:
            counts['uncertainty'] += 1
        if tok in LM_LITIGIOUS:
            counts['litigious'] += 1
        if tok in LM_CONSTRAINING:
            counts['constraining'] += 1

    tone = counts['positive'] + counts['negative']
    polarity = ((counts['positive'] - counts['negative']) / tone) if tone else 0.0

    return {**counts, 'polarity': round(polarity, 4), 'n': len(tokens)}


# Events whose direction is unambiguous regardless of the words around them.
# A guidance raise is bullish even when phrased in entirely neutral vocabulary
# ("Management raised full-year guidance and announced a buyback" contains no
# LM sentiment words at all, yet is plainly positive).
EVENT_POLARITY = {
    'guidance:raised': 0.6, 'guidance:beat': 0.5, 'guidance:initiated': 0.1,
    'guidance:reaffirmed': 0.1, 'guidance:lowered': -0.6,
    'guidance:missed': -0.5, 'guidance:withdrawn': -0.7,
    'analyst:upgrade': 0.4, 'analyst:downgrade': -0.4,
    'event:buyback': 0.3, 'event:dividend_raise': 0.4,
    'event:dividend_cut': -0.6, 'event:layoffs': -0.3,
    'event:legal': -0.2, 'event:partnership': 0.15,
}


def _event_polarity(text: str) -> float | None:
    """Directional signal implied by extracted events, if any."""
    try:
        from nlp import extract as _extract
    except ImportError:
        return None

    ex = _extract.extract(text)
    signals = []
    if ex.guidance:
        signals.append(EVENT_POLARITY.get(f'guidance:{ex.guidance}'))
    if ex.analyst_action:
        signals.append(EVENT_POLARITY.get(f'analyst:{ex.analyst_action}'))
    for e in ex.events:
        signals.append(EVENT_POLARITY.get(f'event:{e}'))

    signals = [s for s in signals if s is not None]
    if not signals:
        return None
    return float(np.clip(sum(signals), -1.0, 1.0))


def score(text: str, blend_vader: bool = True) -> dict:
    """
    Combined sentiment read.

    Returns polarity in [-1, 1] plus the component signals. Three inputs:
    Loughran-McDonald word counts (the primary read, because it is built for
    financial language), event direction (which catches bullish or bearish
    facts stated in neutral vocabulary), and VADER as a minority read of
    headline tone when its lexicon is available.
    """
    lm = lm_scores(text)
    polarity = lm['polarity']
    vader_compound = None

    if blend_vader:
        an = _vader()
        if an is not None and text:
            try:
                vader_compound = float(an.polarity_scores(text)['compound'])
                polarity = 0.7 * polarity + 0.3 * vader_compound
            except Exception:                      # noqa: BLE001
                pass

    event_pol = _event_polarity(text)
    if event_pol is not None:
        # Events dominate when the lexicon is silent, and still carry weight
        # when it is not.
        polarity = (0.75 * event_pol + 0.25 * polarity) if abs(polarity) < 0.15 \
            else (0.5 * event_pol + 0.5 * polarity)

    # Heavy uncertainty language damps conviction either way.
    if lm['n'] > 0:
        uncertainty_ratio = lm['uncertainty'] / max(lm['n'], 1)
        if uncertainty_ratio > 0.05:
            polarity *= max(0.5, 1.0 - uncertainty_ratio * 4)

    return {
        'polarity': round(float(polarity), 4),
        'label': label_for(polarity),
        'lm_positive': lm['positive'], 'lm_negative': lm['negative'],
        'lm_uncertainty': lm['uncertainty'], 'lm_litigious': lm['litigious'],
        'lm_constraining': lm['constraining'],
        'vader': round(vader_compound, 4) if vader_compound is not None else None,
        'tokens': lm['n'],
    }


def label_for(polarity: float) -> str:
    if polarity >= 0.35:
        return '🟢 Positive'
    if polarity >= 0.10:
        return '🟢 Mildly positive'
    if polarity > -0.10:
        return '⚪ Neutral'
    if polarity > -0.35:
        return '🟠 Mildly negative'
    return '🔴 Negative'
