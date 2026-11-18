"""
Email delivery for generated reports.

Evolves the original `send_report`: same SMTP approach, but it sends an
already-rendered report rather than rebuilding one, so what lands in the inbox
is byte-identical to what the Reports tab shows.
"""

from __future__ import annotations

import logging
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from pathlib import Path

import config
from core import db

log = logging.getLogger(__name__)


class SMTPNotConfigured(RuntimeError):
    """Raised when credentials are absent, with instructions rather than a stack trace."""


def _check_config() -> tuple[str, str]:
    if not config.SMTP_USER or not config.SMTP_PASS:
        raise SMTPNotConfigured(
            'SMTP credentials are not set. Copy .env.example to .env and fill in '
            'SMTP_USER and SMTP_PASS. Gmail requires an App Password rather than '
            'your account password — see '
            'https://support.google.com/accounts/answer/185833'
        )
    return config.SMTP_USER, config.SMTP_PASS


def recipients(value: str | list[str] | None) -> list[str]:
    """'a@x.com, b@y.com' -> ['a@x.com', 'b@y.com']."""
    if not value:
        return []
    if isinstance(value, str):
        value = value.replace(';', ',').split(',')
    return [v.strip() for v in value if v and v.strip()]


def html_to_text(html: str) -> str:
    """
    Readable plain-text rendering of a report.

    Sent as the text/plain alternative: mail clients that block HTML show
    it, and spam filters score an HTML-only message as suspicious — a
    scheduled email that lands in spam is the same as one never sent.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        for tag in soup(['style', 'script', 'title']):
            tag.decompose()
        for br in soup.find_all(['br', 'tr', 'div', 'p', 'h1', 'h2', 'li']):
            br.insert_after('\n')
        for cell in soup.find_all(['td', 'th']):
            cell.insert_after('  ')
        text = soup.get_text()
    except Exception:                                  # noqa: BLE001
        import re
        text = re.sub(r'<[^>]+>', ' ', html)
    lines = [' '.join(line.split()) for line in text.splitlines()]
    out, blank = [], False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank:
            out.append('')
            blank = True
    return '\n'.join(out).strip() + '\n'


def send_report(html: str, recipient: str | list[str], subject: str | None = None,
                report_id: str | None = None,
                attachment_path: str | Path | None = None) -> None:
    """
    Send a rendered report. Raises SMTPNotConfigured with guidance if unset.

    `recipient` may hold several addresses separated by commas. When
    `report_id` is given, the report's history row is stamped as emailed.
    """
    user, password = _check_config()
    to = recipients(recipient)
    if not to:
        raise ValueError('no recipient address given')
    subject = subject or f'Quant Research Brief — {date.today()}'

    body = MIMEMultipart('alternative')
    body.attach(MIMEText(html_to_text(html), 'plain', 'utf-8'))
    body.attach(MIMEText(html, 'html', 'utf-8'))

    p = Path(attachment_path) if attachment_path else None
    if p is not None and p.exists():
        # Attachments belong beside the alternatives, not among them: inside
        # multipart/alternative a client shows exactly one part.
        from email.mime.application import MIMEApplication
        msg = MIMEMultipart('mixed')
        msg.attach(body)
        part = MIMEApplication(p.read_bytes(), Name=p.name)
        part['Content-Disposition'] = f'attachment; filename="{p.name}"'
        msg.attach(part)
    else:
        msg = body

    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = ', '.join(to)
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=user.split('@')[-1] or None)

    with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.login(user, password)
        server.sendmail(user, to, msg.as_string())

    log.info('report emailed to %s', ', '.join(to))

    if report_id:
        db.upsert(db.reports, [{'report_id': report_id,
                                'emailed_at': config.utc_now()}])


def send_file(path: str | Path, recipient: str, subject: str | None = None) -> None:
    """Send a previously generated report file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'report not found: {p}')
    send_report(p.read_text(encoding='utf-8'), recipient,
                subject or f'Quant Research Brief — {p.stem}')
