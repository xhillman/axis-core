"""Tests for shared sensitive-data redaction utilities."""

from __future__ import annotations

from axis_core.redaction import REDACTED_VALUE, is_sensitive_key, redact_sensitive_data


class TestSensitiveKeyDetection:
    """Tests for key-name based sensitivity detection."""

    def test_matches_hyphenated_secret_keys(self) -> None:
        """Hyphenated key variants should be treated as sensitive."""
        assert is_sensitive_key("x-api-key")
        assert is_sensitive_key("private-key")
        assert is_sensitive_key("access-key")


class TestRedactSensitiveData:
    """Tests for recursive redaction behavior."""

    def test_redacts_values_for_hyphenated_keys(self) -> None:
        payload = {
            "x-api-key": "sk-secret-123",
            "nested": {"private-key": "value"},
            "safe-value": "ok",
        }

        redacted = redact_sensitive_data(payload)

        assert redacted["x-api-key"] == REDACTED_VALUE
        assert redacted["nested"]["private-key"] == REDACTED_VALUE
        assert redacted["safe-value"] == "ok"

    def test_redacts_sensitive_key_value_pairs_in_error_strings(self) -> None:
        message = (
            "tool failed: x-api-key=sk-secret-123 "
            "authorization: Bearer tok_abc123 "
            "password='super-secret'"
        )

        redacted = redact_sensitive_data(message)

        assert "sk-secret-123" not in redacted
        assert "tok_abc123" not in redacted
        assert "super-secret" not in redacted
        assert redacted.count(REDACTED_VALUE) >= 3
