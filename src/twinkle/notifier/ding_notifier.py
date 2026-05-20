import base64
import hashlib
import hmac
import json
import time
import urllib.parse
from typing import Optional

from .base import Notifier


class DingNotifier(Notifier):
    """Send notifications to a DingTalk custom robot webhook.

    Args:
        ding_url: The full webhook URL, e.g.
            ``https://oapi.dingtalk.com/robot/send?access_token=xxx``.
        secret: Optional signing secret. If provided, ``timestamp``/``sign``
            query parameters are appended to each request as required by
            DingTalk's signed-robot mode.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        ding_url: str,
        secret: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        super().__init__()
        if not ding_url:
            raise ValueError('ding_url must be a non-empty DingTalk webhook URL')
        self.ding_url = ding_url
        self.secret = secret
        self.timeout = timeout

    def _sign(self) -> dict:
        """Build ``timestamp``/``sign`` query params for signed webhooks."""
        if not self.secret:
            return {}
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f'{timestamp}\n{self.secret}'
        hmac_code = hmac.new(
            self.secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return {'timestamp': timestamp, 'sign': sign}

    def _build_url(self) -> str:
        extra = self._sign()
        if not extra:
            return self.ding_url
        sep = '&' if '?' in self.ding_url else '?'
        query = '&'.join(f'{k}={v}' for k, v in extra.items())
        return f'{self.ding_url}{sep}{query}'

    def __call__(self, message: str) -> dict:
        """Send ``message`` as a plain-text DingTalk notification.

        Returns the parsed JSON response from DingTalk. Raises on HTTP
        failure or on a non-zero ``errcode`` in the response body.
        """
        import requests

        payload = {
            'msgtype': 'text',
            'text': {
                'content': str(message)
            },
        }
        resp = requests.post(
            self._build_url(),
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get('errcode', 0) != 0:
            raise RuntimeError(f'DingTalk notify failed: errcode={result.get("errcode")}, '
                               f'errmsg={result.get("errmsg")}')
        return result
