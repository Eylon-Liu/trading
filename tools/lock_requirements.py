#!/usr/bin/env python3
"""Write requirements.lock.txt from the versions installed right now.

Run it after upgrading dependencies *and* seeing the test suite pass:

    python -m pytest -q && python tools/lock_requirements.py
"""
from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = f"""# Exact versions the test suite last passed with (Python {sys.version_info.major}.{sys.version_info.minor}).
# requirements.txt gives minimums; install from this file instead when a
# newer release of a dependency breaks something:
#     pip install -r requirements.lock.txt
# Regenerate after upgrading and re-running the tests:
#     python tools/lock_requirements.py
"""


def main() -> int:
    names = []
    for line in (ROOT / 'requirements.txt').read_text().splitlines():
        line = line.split('#')[0].strip()
        if line:
            names.append(re.split(r'[<>=!~\[]', line)[0].strip())
    lines = [HEADER]
    for name in names + ['pytest']:
        try:
            lines.append(f'{name}=={version(name)}')
        except PackageNotFoundError:
            lines.append(f'# {name}: not installed')
    (ROOT / 'requirements.lock.txt').write_text('\n'.join(lines) + '\n')
    print(f'wrote requirements.lock.txt ({len(names) + 1} packages)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
