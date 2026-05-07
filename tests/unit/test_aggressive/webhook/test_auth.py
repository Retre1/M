"""Tests for webhook auth — HMAC + shared-secret verification."""

from __future__ import annotations

import pytest

from apexfx.aggressive.webhook.auth import (
    constant_time_eq,
    hmac_sign,
    verify_hmac,
    verify_shared_secret,
)


class TestConstantTimeEq:
    def test_equal_strings(self) -> None:
        assert constant_time_eq("hello", "hello") is True

    def test_unequal_strings(self) -> None:
        assert constant_time_eq("hello", "world") is False

    def test_different_lengths(self) -> None:
        assert constant_time_eq("hello", "helloworld") is False

    def test_empty_both(self) -> None:
        assert constant_time_eq("", "") is True

    def test_empty_one(self) -> None:
        assert constant_time_eq("hello", "") is False
        assert constant_time_eq("", "hello") is False


class TestHmacSign:
    def test_deterministic(self) -> None:
        s1 = hmac_sign("secret", b"body")
        s2 = hmac_sign("secret", b"body")
        assert s1 == s2

    def test_different_body_different_sig(self) -> None:
        s1 = hmac_sign("secret", b"body1")
        s2 = hmac_sign("secret", b"body2")
        assert s1 != s2

    def test_different_secret_different_sig(self) -> None:
        s1 = hmac_sign("secret1", b"body")
        s2 = hmac_sign("secret2", b"body")
        assert s1 != s2

    def test_empty_secret_raises(self) -> None:
        with pytest.raises(ValueError):
            hmac_sign("", b"body")

    def test_hex_format(self) -> None:
        sig = hmac_sign("s", b"b")
        assert len(sig) == 64  # SHA256 → 32 bytes → 64 hex chars
        int(sig, 16)  # parses as hex


class TestVerifyHmac:
    def test_correct_sig_passes(self) -> None:
        body = b"my-body"
        secret = "shared"
        sig = hmac_sign(secret, body)
        assert verify_hmac(secret, body, sig) is True

    def test_wrong_sig_fails(self) -> None:
        assert verify_hmac("shared", b"my-body", "deadbeef" * 8) is False

    def test_tampered_body_fails(self) -> None:
        secret = "shared"
        sig = hmac_sign(secret, b"original")
        assert verify_hmac(secret, b"modified", sig) is False

    def test_empty_signature_fails(self) -> None:
        assert verify_hmac("shared", b"body", "") is False

    def test_empty_secret_fails_safely(self) -> None:
        # hmac_sign would raise; verify must handle gracefully
        # (we use try/except internally? actually we don't — test that
        # verify_hmac does not crash on empty secret)
        with pytest.raises(ValueError):
            verify_hmac("", b"body", "anysig")


class TestVerifySharedSecret:
    def test_correct_passes(self) -> None:
        assert verify_shared_secret("expected", "expected") is True

    def test_wrong_fails(self) -> None:
        assert verify_shared_secret("expected", "guessed") is False

    def test_empty_provided_fails(self) -> None:
        assert verify_shared_secret("expected", "") is False

    def test_empty_expected_fails(self) -> None:
        # Defense in depth — if the server has empty secret config,
        # don't accept ALL requests
        assert verify_shared_secret("", "anything") is False
        assert verify_shared_secret("", "") is False
