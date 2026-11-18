"""
Extractive summarization — TextRank over sentences.

Picks the sentences that best represent a body of text rather than generating
new prose, which means it never invents a fact. That property matters more
here than fluency: a hallucinated earnings figure in a research tool is worse
than a slightly clunky sentence.

nlp/llm.py optionally layers genuine synthesis on top when an API key is
configured.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'all', 'also', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does',
    'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had',
    'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'him', 'his', 'how',
    'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more',
    'most', 'my', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only',
    'or', 'other', 'our', 'out', 'over', 'own', 'said', 'same', 'she', 'should',
    'so', 'some', 'such', 'than', 'that', 'the', 'their', 'them', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
    'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you',
    'your', 'from', 'has', 'had',
}

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')
_WORD_RE = re.compile(r"[a-z][a-z'-]+")


def sentences(text: str) -> list[str]:
    """Split into sentences, dropping fragments too short to carry meaning."""
    if not text:
        return []
    clean = re.sub(r'\s+', ' ', text).strip()
    return [s.strip() for s in _SENT_SPLIT.split(clean)
            if len(s.strip()) > 25]


def _words(sentence: str) -> list[str]:
    return [w for w in _WORD_RE.findall(sentence.lower()) if w not in STOPWORDS]


def textrank(text: str, top_n: int = 3) -> list[str]:
    """
    Rank sentences by PageRank over a similarity graph.

    Sentences are linked by shared vocabulary; the ones the graph converges on
    are those most connected to the rest of the document.
    """
    sents = sentences(text)
    if len(sents) <= top_n:
        return sents

    tokenized = [_words(s) for s in sents]
    n = len(sents)
    sim = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            a, b = set(tokenized[i]), set(tokenized[j])
            if not a or not b:
                continue
            overlap = len(a & b)
            if overlap:
                denom = np.log(len(a) + 1) + np.log(len(b) + 1)
                if denom > 0:
                    sim[i, j] = sim[j, i] = overlap / denom

    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    scores = np.ones(n) / n
    damping = 0.85
    for _ in range(30):
        prev = scores.copy()
        scores = (1 - damping) / n + damping * (transition.T @ scores)
        if np.abs(scores - prev).sum() < 1e-6:
            break

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_n]
    return [sents[i] for i in sorted(ranked)]     # keep original order


def summarize(text: str, max_sentences: int = 3, max_chars: int = 400) -> str:
    """Short extractive summary."""
    if not text:
        return ''
    picked = textrank(text, top_n=max_sentences)
    if not picked:
        return text[:max_chars]
    out = ' '.join(picked)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(' ', 1)[0] + '…'
    return out


def keywords(text: str, top_n: int = 8) -> list[str]:
    """Most frequent meaningful terms."""
    words = _words(text or '')
    if not words:
        return []
    return [w for w, _c in Counter(words).most_common(top_n)]


def digest(items: list[dict], max_items: int = 5) -> str:
    """
    One-paragraph rollup across several articles.

    Used for the per-ticker news card and the daily report.
    """
    if not items:
        return 'No recent coverage.'
    lines = []
    for it in items[:max_items]:
        title = (it.get('title') or '').strip()
        if not title:
            continue
        label = it.get('sentiment_label', '')
        lines.append(f'{label} {title}'.strip())
    return ' | '.join(lines) if lines else 'No recent coverage.'
