#!/usr/bin/env python3
"""
Standalone Email Notification Test Script
----------------------------------------

Purpose:
- Sends an email using Gmail SMTP
- Triggers instant phone notifications via Gmail / Google Voice app

Requirements:
- Gmail account
- Gmail App Password
- Python 3.8+

Secrets are read ONLY from environment variables.
"""

import os
import sys
import smtplib
from email.message import EmailMessage
from datetime import datetime

# =========================
# REQUIRED ENV VARS
# =========================

REQUIRED_VARS = [
    "GMAIL_FROM",
    "GMAIL_APP_PASSWORD",
    "EMAIL_TO",
]

missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    print(f"❌ Missing environment variables: {missing}")
    sys.exit(1)

GMAIL_FROM = os.environ["GMAIL_FROM"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
EMAIL_TO = os.environ["EMAIL_TO"]

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465


def send_email(subject: str, body: str):
    msg = EmailMessage()
    msg["From"] = GMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(GMAIL_FROM, GMAIL_APP_PASSWORD)
        server.send_message(msg)


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = "[PRNG][TEST] Email Notification OK"
    body = f"""
PRNG Email Notification Test

Status: SUCCESS
Timestamp: {ts}

If you received this:
- Gmail SMTP works
- App password works
- Phone notifications are functional
"""

    print("📧 Sending test email...")
    send_email(subject, body)
    print("✅ Email sent successfully")
    print("📱 Check your phone (Gmail / Google Voice notification)")


if __name__ == "__main__":
    main()
