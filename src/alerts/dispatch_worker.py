"""
Asynchronous Task Queue & Dispatch Workers for Project Vigilance.
Decouples real-time video pipeline from external network calls via Asyncio Worker Pool / Celery interface.
Dispatches to Webhook Tactical C2, Twilio Telephony SMS/Voice, and Local Edge GPIO Alarm Relay.
"""

import asyncio
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import requests

try:
    from twilio.rest import Client as TwilioClient
    HAS_TWILIO = True
except ImportError:
    HAS_TWILIO = False

from src.alerts.offline_cache import OfflineCacheManager
from src.alerts.schemas import IncidentPayload, IncidentType
from src.utils.logger import setup_logger

logger = setup_logger("AlertDispatcher")


class TacticalC2WebhookDispatcher:
    """Dispatches JSON alert payloads to central Command & Control (C2) servers over HTTP/HTTPS."""

    def __init__(self, webhook_url: str = "https://httpbin.org/post", timeout_sec: float = 5.0, max_retries: int = 3) -> None:
        self.webhook_url = webhook_url
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def dispatch(self, payload: IncidentPayload) -> bool:
        """Sends HTTP POST request with exponential backoff retries."""
        headers = {"Content-Type": "application/json", "User-Agent": "ProjectVigilance-Edge/2.0"}
        data = payload.model_dump()

        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                logger.info(f"Posting alert [{payload.incident_id}] to Tactical C2 Webhook: {self.webhook_url} (Attempt {attempt})...")
                resp = requests.post(self.webhook_url, json=data, headers=headers, timeout=self.timeout_sec)
                if resp.status_code in [200, 201, 202]:
                    logger.info(f"[SUCCESS] Tactical C2 Webhook dispatch OK for [{payload.incident_id}] (HTTP {resp.status_code})")
                    return True
                else:
                    logger.warning(f"Tactical C2 returned HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                logger.warning(f"Tactical C2 Webhook POST failed (Attempt {attempt}): {e}")

            time.sleep(1.0 * (2 ** (attempt - 1)))

        return False


class TwilioEmergencyDispatcher:
    """Dispatches automated emergency SMS & Text-to-Speech (TTS) Voice Calls via Twilio SDK."""

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        emergency_contacts: Optional[List[str]] = None
    ) -> None:
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "AC_MOCK_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN", "MOCK_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER", "+18005550199")
        self.emergency_contacts = emergency_contacts or ["+18005550100"]

        if HAS_TWILIO and self.account_sid.startswith("AC") and not "MOCK" in self.account_sid:
            self.client = TwilioClient(self.account_sid, self.auth_token)
        else:
            self.client = None
            logger.info("Twilio SDK in Mock Mode (API keys missing or mock credentials configured).")

    def send_sms(self, payload: IncidentPayload) -> bool:
        """Sends concise tactical alert SMS message."""
        sms_text = (
            f"[VIGILANCE ALERT] [{payload.event_type.value.upper()}]\n"
            f"Location: {payload.location_tag} ({payload.camera_id})\n"
            f"Time: {payload.timestamp}\n"
            f"Confidence: {payload.confidence_score*100:.0f}%\n"
            f"Media: {payload.clip_path or 'Local Outpost Storage'}"
        )

        for number in self.emergency_contacts:
            if self.client is not None:
                try:
                    message = self.client.messages.create(
                        body=sms_text,
                        from_=self.from_number,
                        to=number
                    )
                    logger.info(f"[SMS] Sent to {number}: SID = {message.sid}")
                except Exception as e:
                    logger.error(f"Failed to send Twilio SMS to {number}: {e}")
            else:
                logger.info(f"[SMS] [MOCK to {number}]:\n{sms_text}")
        return True

    def trigger_voice_call(self, payload: IncidentPayload) -> bool:
        """Triggers automated Text-to-Speech (TTS) emergency voice call."""
        tts_script = (
            f"<Response><Say voice='alice'>"
            f"Emergency Alert! Project Vigilance has detected a verified {payload.event_type.value} incident "
            f"at location {payload.location_tag}. Immediate response required."
            f"</Say></Response>"
        )

        for number in self.emergency_contacts:
            if self.client is not None:
                try:
                    call = self.client.calls.create(
                        twiml=tts_script,
                        from_=self.from_number,
                        to=number
                    )
                    logger.info(f"[VOICE CALL] Triggered to {number}: SID = {call.sid}")
                except Exception as e:
                    logger.error(f"Failed to trigger Twilio Voice call to {number}: {e}")
            else:
                logger.info(f"[VOICE CALL] [MOCK to {number}]: Reading TTS message for {payload.event_type.value}")
        return True


class LocalEdgeAlarmRelay:
    """Triggers local hardware siren/strobe light relay on edge outpost hardware."""

    def __init__(self, gpio_pin: int = 18) -> None:
        self.gpio_pin = gpio_pin

    def trigger_siren(self, duration_sec: float = 3.0) -> None:
        """Simulates or triggers GPIO siren relay pulse."""
        logger.info(f"[LOCAL ALARM] SIREN PULSED on GPIO Pin {self.gpio_pin} for {duration_sec}s!")


class AlertDispatcher:
    """Master Asynchronous Dispatch Engine managing Webhook, Telephony, Edge Alarm, and Offline SQLite Cache."""

    def __init__(
        self,
        c2_webhook_url: str = "https://httpbin.org/post",
        enable_twilio: bool = False,
        enable_local_alarm: bool = True,
        cache_manager: Optional[OfflineCacheManager] = None
    ) -> None:
        self.c2_dispatcher = TacticalC2WebhookDispatcher(webhook_url=c2_webhook_url)
        self.twilio_dispatcher = TwilioEmergencyDispatcher()
        self.local_alarm = LocalEdgeAlarmRelay()
        self.enable_twilio = enable_twilio
        self.enable_local_alarm = enable_local_alarm
        self.cache_manager = cache_manager or OfflineCacheManager()

        # Async queue setup
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None

    def dispatch_incident_sync(self, payload: IncidentPayload) -> None:
        """Synchronous dispatch wrapper for non-async pipeline callers."""
        logger.info(f"Processing dispatch for verified incident [{payload.incident_id}]...")

        # 1. Trigger local edge alarm siren immediately
        if self.enable_local_alarm:
            self.local_alarm.trigger_siren(duration_sec=2.0)

        # 2. Attempt Webhook C2 dispatch
        success = self.c2_dispatcher.dispatch(payload)

        # 3. Trigger Twilio Telephony SMS & Voice if enabled
        if self.enable_twilio:
            self.twilio_dispatcher.send_sms(payload)
            self.twilio_dispatcher.trigger_voice_call(payload)

        # 4. If WAN dispatch failed, persist to local SQLite offline cache
        if not success:
            logger.warning(f"WAN dispatch failed for [{payload.incident_id}]. Serializing payload to offline SQLite cache.")
            self.cache_manager.enqueue_alert(payload)
