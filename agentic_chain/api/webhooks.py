"""
Webhook manager for sending notifications.
"""

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class Webhook:
    """Represents a registered webhook."""
    
    def __init__(
        self,
        webhook_id: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
    ):
        self.webhook_id = webhook_id
        self.url = url
        self.events = events
        self.secret = secret
        self.created_at = datetime.now(timezone.utc)


class WebhookManager:
    """Manages webhook registrations and notifications."""
    
    def __init__(self):
        self._webhooks: Dict[str, Webhook] = {}
    
    def register(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
    ) -> Webhook:
        """Register a new webhook."""
        webhook_id = str(uuid4())
        webhook = Webhook(
            webhook_id=webhook_id,
            url=url,
            events=events,
            secret=secret,
        )
        self._webhooks[webhook_id] = webhook
        logger.info(f"Registered webhook {webhook_id} for events {events}")
        return webhook
    
    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook."""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            logger.info(f"Deleted webhook {webhook_id}")
            return True
        return False
    
    def list_webhooks(self) -> List[Webhook]:
        """List all registered webhooks."""
        return list(self._webhooks.values())
    
    def _compute_signature(self, payload: bytes, secret: str) -> str:
        """Compute HMAC-SHA256 signature for payload."""
        return hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256
        ).hexdigest()
    
    async def send_notification(
        self,
        event: str,
        data: Dict[str, Any],
        webhook_url: Optional[str] = None,
    ) -> int:
        """
        Send notification to registered webhooks.
        
        Args:
            event: Event type (e.g., 'job.completed')
            data: Event payload
            webhook_url: Optional specific URL to notify (for job-specific webhooks)
            
        Returns:
            Number of successful notifications sent
        """
        payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        
        # Collect target webhooks
        targets: List[tuple] = []  # (url, secret)
        
        # Add job-specific webhook if provided
        if webhook_url:
            targets.append((webhook_url, None))
        
        # Add registered webhooks for this event
        for webhook in self._webhooks.values():
            if event in webhook.events:
                targets.append((webhook.url, webhook.secret))
        
        if not targets:
            return 0
        
        success_count = 0
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url, secret in targets:
                try:
                    headers = {"Content-Type": "application/json"}
                    
                    if secret:
                        signature = self._compute_signature(payload_bytes, secret)
                        headers["X-Webhook-Signature"] = f"sha256={signature}"
                    
                    response = await client.post(
                        url,
                        content=payload_bytes,
                        headers=headers,
                    )
                    
                    if response.status_code < 400:
                        success_count += 1
                        logger.info(f"Webhook notification sent to {url}")
                    else:
                        logger.warning(
                            f"Webhook notification failed for {url}: "
                            f"status={response.status_code}"
                        )
                except Exception as e:
                    logger.error(f"Failed to send webhook to {url}: {e}")
        
        return success_count


# Global webhook manager instance
webhook_manager = WebhookManager()
