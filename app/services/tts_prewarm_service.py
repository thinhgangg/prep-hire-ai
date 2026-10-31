from __future__ import annotations

import json
import threading
import urllib.request

from app.core.config import settings


class TtsPrewarmService:
    """Best-effort notification that never delays or fails an interview request."""

    @staticmethod
    def prewarm(text: str) -> None:
        normalized = " ".join((text or "").split())
        if not settings.TTS_PREWARM_ENABLED or not normalized:
            return

        def notify() -> None:
            payload = json.dumps({"text": normalized}, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                settings.TTS_PREWARM_URL,
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=settings.TTS_PREWARM_NOTIFY_TIMEOUT_SECONDS,
                ) as response:
                    response.read(1)
            except Exception as exc:
                # TTS is optional. A failed prewarm must never block question delivery.
                if settings.DEBUG:
                    print(f"[TTS PREWARM] skipped: {exc}")

        threading.Thread(target=notify, name="tts-prewarm", daemon=True).start()


tts_prewarm_service = TtsPrewarmService()

