"""HMAC authentication for the TradingView webhook.

Why HMAC and not just an API token
----------------------------------
TradingView's free webhook has a few weaknesses for unauthenticated payloads:

1. **No client TLS auth** — TV connects to your endpoint over plain HTTPS,
   anyone who learns the URL can spam it.
2. **Static URL** — there's no per-request rotation; URL leaks once = leaks
   forever.
3. **No request signing built-in** — Pro+ subscribers can set custom headers,
   but those are still secrets-in-transit.

HMAC gives us **payload integrity + authenticity** with one shared secret.
The receiver computes the expected MAC over the raw request body and
constant-time compares to the header.  An attacker who learns only the URL
can't forge a valid alert without the secret.

Constant-time compare
---------------------
Use ``hmac.compare_digest`` — never ``==``.  String equality leaks length
and content via timing.  This is exactly the kind of bug that lets an
attacker brute-force one byte at a time.

Replay protection
-----------------
HMAC alone doesn't prevent replay.  We pair it with the dedup cache
(``dedupe.py``) which drops alerts whose ``SignalId`` was seen in the
last N minutes.

Pine Script integration
-----------------------
Pine v5 cannot compute HMAC at alert-emit time, so the secret is **not
in the JSON body**.  Instead we put it in a custom HTTP header set in
the TradingView alert dialog ("X-Webhook-Secret").  Combined with HTTPS
this is sufficient — the body+header are encrypted in transit, and the
server validates the header.  When Pine v6 adds proper secrets, switch
to true HMAC of the body.
"""

from __future__ import annotations

import hmac
from hashlib import sha256


def constant_time_eq(a: str, b: str) -> bool:
    """Wrapper around ``hmac.compare_digest`` for string-typed comparison.

    Exists as its own function purely so the security-critical path is
    visible in greps and unit-tested separately.
    """
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def hmac_sign(secret: str, body: bytes) -> str:
    """Compute hex-encoded HMAC-SHA256 of ``body`` keyed with ``secret``.

    Used both for **producing** signatures (test fixtures, future Pine v6
    integration) and for the **expected value** during verification.
    """
    if not secret:
        raise ValueError("secret must be non-empty")
    digest = hmac.new(secret.encode("utf-8"), body, sha256).hexdigest()
    return digest


def verify_hmac(secret: str, body: bytes, provided_signature: str) -> bool:
    """Constant-time verify ``provided_signature`` over ``body`` with ``secret``.

    Returns ``False`` on any mismatch / empty input — callers should map
    that to a 401 response, not crash.  Specifically guards against:

      * empty signature header
      * wrong-length signature
      * malicious string-comparison timing leak
    """
    if not provided_signature:
        return False
    expected = hmac_sign(secret, body)
    return constant_time_eq(expected, provided_signature)


def verify_shared_secret(expected_secret: str, provided_secret: str) -> bool:
    """Plain shared-secret check (constant-time).

    Used as the Pine v5 fallback path until v6 supports HMAC body signing.
    Slightly weaker than HMAC because the secret is sent verbatim, but
    over HTTPS this is acceptable for retail-volume traffic.
    """
    if not expected_secret or not provided_secret:
        return False
    return constant_time_eq(expected_secret, provided_secret)
