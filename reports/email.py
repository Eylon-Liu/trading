"""
Email delivery for generated reports.

Evolves the original `send_report`: same SMTP approach, but it sends an
already-rendered report rather than rebuilding one, so what lands in the inbox
is byte-identical to what the Reports tab shows.
"""

from __future__ import annotations

import logging
import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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


def send_report(html: str, recipient: str, subject: str | None = None,
                report_id: str | None = None,
                attachment_path: str | Path | None = None) -> None:
    """Send a rendered report. Raises SMTPNotConfigured with guidance if unset."""
    user, password = _check_config()
    subject = subject or f'Quant Research Brief — {date.today()}'

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = recipient
    msg.attach(MIMEText(html, 'html', 'utf-8'))

    if attachment_path:
        p = Path(attachment_path)
        if p.exists():
            from email.mime.application import MIMEApplication
            part = MIMEApplication(p.read_bytes(), Name=p.name)
            part['Content-Disposition'] = f'attachment; filename="{p.name}"'
            msg.attach(part)

    with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.login(user, password)
        server.sendmail(user, recipient, msg.as_string())

    log.info('report emailed to %s', recipient)

    if report_id:
        db.upsert(db.reports, [{'report_id': report_id,
                                'emailed_at': datetime.utcnow()}])


def send_file(path: str | Path, recipient: str, subject: str | None = None) -> None:
    """Send a previously generated report file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'report not found: {p}')
    send_report(p.read_text(encoding='utf-8'), recipient,
                subject or f'Quant Research Brief — {p.stem}')
