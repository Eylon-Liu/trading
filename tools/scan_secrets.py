#!/usr/bin/env python3
"""
Secret scanner. Run standalone, or as a pre-commit hook.

    python tools/scan_secrets.py            # scan tracked + staged files
    python tools/scan_secrets.py --all      # scan the whole working tree

Exits non-zero when something that looks like a live credential is found, so
as a pre-commit hook it blocks the commit.

This exists because .gitignore alone is a single point of failure: one
`git add -f`, one renamed file, one key pasted into a notebook, and the
credential is in history permanently. A scanner is the second line.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Patterns are deliberately anchored on vendor-specific prefixes rather than
# generic entropy: entropy alone flags every hash and base64 blob in a repo,
# and a scanner people learn to ignore protects nothing.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ('Anthropic API key', re.compile(r'sk-ant-[A-Za-z0-9_\-]{20,}')),
    ('OpenAI API key', re.compile(r'sk-(?:proj-)?[A-Za-z0-9_\-]{32,}')),
    ('Google API key (AIza)', re.compile(r'AIza[0-9A-Za-z_\-]{35}')),
    ('Google API key (AQ.)', re.compile(r'\bAQ\.[A-Za-z0-9_\-]{20,}')),
    ('AWS access key', re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b')),
    ('GitHub token', re.compile(r'\bgh[pousr]_[A-Za-z0-9]{36,}')),
    ('Slack token', re.compile(r'\bxox[abprs]-[A-Za-z0-9\-]{10,}')),
    # Google App Passwords are 16 lowercase letters, usually pasted in four
    # groups of four. The generic rule below misses them because it requires
    # no whitespace inside the value.
    ('Google App Password',
     re.compile(r'(?i)\b(?:SMTP_PASS|smtp_password|app[_-]?password)\b\s*[:=]\s*'
                r'["\']?(?:[a-z]{4}[ -]?){4}["\']?')),
    # Finnhub keys are 40 lowercase alphanumerics, often two 20-char halves.
    # The generic rule below misses a bare unquoted assignment.
    ('Finnhub API key',
     re.compile(r'(?i)\bfinnhub[_-]?(?:api[_-]?)?key\b\s*[:=]\s*["\']?[a-z0-9]{30,}')),
    ('SMTP credentials in code',
     re.compile(r'(?i)\.login\(\s*["\'][^"\']+["\']\s*,\s*["\'][^"\']{8,}["\']')),
    ('Private key block', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
    ('Generic assigned secret',
     re.compile(r'(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b'
                r'\s*[:=]\s*["\']([^"\'\s]{16,})["\']')),
]

# Files that are supposed to contain secrets, or to describe them.
SKIP_NAMES = {'.env', '.env.local'}
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv',
             'cache', 'db', '.pytest_cache', '.ruff_cache'}
SKIP_SUFFIXES = {'.pkl', '.sqlite', '.db', '.png', '.jpg', '.jpeg', '.gif',
                 '.pdf', '.zip', '.gz', '.parquet', '.ico', '.woff', '.woff2'}

# Placeholder values that are documentation, not credentials.
ALLOW = re.compile(
    r'(?i)(your[_-]?(api[_-]?)?key|xxx+|\.\.\.|<[^>]+>|example|placeholder'
    r'|changeme|dummy|fake|test[_-]?key|sk-ant-api03-REPLACE)')


def _tracked_files() -> list[Path]:
    """Files git knows about, plus anything staged."""
    try:
        tracked = subprocess.run(
            ['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True,
            check=True).stdout.split()
        staged = subprocess.run(
            ['git', 'diff', '--cached', '--name-only'], cwd=ROOT,
            capture_output=True, text=True, check=True).stdout.split()
        return [ROOT / f for f in set(tracked) | set(staged)]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _all_files() -> list[Path]:
    out = []
    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def scan(paths: list[Path]) -> list[tuple[Path, int, str, str]]:
    findings = []
    for path in paths:
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # The scanner itself contains every pattern by definition.
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(errors='ignore')
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            if ALLOW.search(line):
                continue
            for label, pattern in PATTERNS:
                m = pattern.search(line)
                if m:
                    hit = m.group(0)
                    masked = hit[:6] + '…' + hit[-4:] if len(hit) > 14 else '…'
                    rel = path.relative_to(ROOT)
                    findings.append((rel, lineno, label, masked))
                    break
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all', action='store_true',
                    help='scan the whole working tree, not just tracked files')
    args = ap.parse_args()

    paths = _all_files() if args.all else (_tracked_files() or _all_files())
    findings = scan(paths)

    if not findings:
        print(f'✅ no secrets found ({len(paths)} files scanned)')
        return 0

    print('🚨 possible secrets found — commit blocked\n')
    for rel, lineno, label, masked in findings:
        print(f'  {rel}:{lineno}  {label}: {masked}')
    print('\nMove the value into .env (gitignored) and reference it via '
          'os.environ. If this is a false positive, add a placeholder marker '
          'or extend ALLOW in tools/scan_secrets.py.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
