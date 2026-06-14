"""Telegram delivery via the raw Bot API (requests), with retry.

Reads ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` from the environment (set as
GitHub Actions secrets for the scheduled run). If they're missing it runs in
*dry-run* mode: messages are logged, not sent, so local runs never fail.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from alerting.formatter import format_alert, format_run_header
from common.logging_config import get_logger
from common.types import Signal, SignalAction

log = get_logger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        parse_mode: str = "Markdown",
        disable_web_page_preview: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.parse_mode = parse_mode
        self.disable_web_page_preview = disable_web_page_preview
        self.max_retries = max_retries

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_message(self, text: str) -> bool:
        if not self.enabled:
            log.warning("Telegram not configured — DRY RUN. Message:\n%s\n%s", text, "-" * 40)
            return False

        import requests

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": self.disable_web_page_preview,
        }
        url = _API.format(token=self.token)
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code == 200 and resp.json().get("ok"):
                    return True
                # 429 -> respect retry_after
                if resp.status_code == 429:
                    wait = resp.json().get("parameters", {}).get("retry_after", 2 * attempt)
                    time.sleep(float(wait))
                    continue
                log.warning("Telegram send failed (%s): %s", resp.status_code, resp.text[:200])
            except Exception as exc:
                log.warning("Telegram send error (attempt %s): %s", attempt, exc)
            time.sleep(1.5 * attempt)
        return False

    def send_signals(self, signals: list[Signal], settings=None, header_as_of=None) -> int:
        """Send a header + each signal. Returns the count successfully sent."""
        if settings is None:
            from config.loader import get_settings

            settings = get_settings()
        if not settings.alerts.telegram.enabled:
            log.info("Telegram alerts disabled in settings")
            return 0

        buys = [s for s in signals if s.action == SignalAction.BUY]
        buys.sort(key=lambda s: s.composite, reverse=True)
        exits = [s for s in signals if s.action == SignalAction.EXIT]
        cap = settings.alerts.telegram.max_signals_per_run
        buys = buys[:cap]

        sent = 0
        if header_as_of is not None:
            regime = signals[0].details.get("regime") if signals else None
            self.send_message(format_run_header(header_as_of, len(buys), len(exits), regime))

        for sig in exits + buys:  # exits first — acting on them is time-sensitive
            if self.send_message(format_alert(sig, settings.alerts.top_reasons)):
                sent += 1
            time.sleep(0.4)  # stay well under Telegram's rate limit
        return sent


def get_notifier(settings=None) -> TelegramNotifier:
    if settings is None:
        from config.loader import get_settings

        settings = get_settings()
    tg = settings.alerts.telegram
    return TelegramNotifier(
        parse_mode=tg.parse_mode,
        disable_web_page_preview=tg.disable_web_page_preview,
    )
