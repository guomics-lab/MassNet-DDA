"""Redact credentials before writing configuration or provenance artifacts.

This module deliberately has no model/runtime dependencies. Redaction creates a
copy: callers must keep the original configuration for authenticated services.
Free-form text is not a secret store; only credential keys and URLs are covered.
"""

from collections.abc import Mapping
import re
from urllib.parse import urlsplit, urlunsplit


REDACTED = "[REDACTED]"
_SECRET_PARTS = (
    "token", "password", "passwd", "secret", "credential", "apikey",
    "authorization", "privatekey", "accesskey",
)
# An apostrophe is legal in URL user information; stopping at it could disclose
# the rest of a password. Exclude only whitespace and non-URL quoting delimiters.
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"<>]+")


def redact_url(value):
    """Remove user information, the entire query, and fragment from a URL.

    Query parameter names do not reliably identify credentials (signed URLs are
    one example), so no query parameters are retained in exported provenance.
    Malformed URLs are suppressed rather than returned with possible secrets.
    """
    try:
        parsed = urlsplit(value)
        if not parsed.scheme:
            return value
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except (ValueError, UnicodeError):
        return REDACTED


def redact_config(value):
    """Return a recursive credential-redacted copy of JSON-like configuration.

    Sensitive keys are matched case-insensitively with punctuation removed, so
    ``api_key``, ``api-key``, and ``apiKey`` receive the same treatment. Ordinary
    values (including peptide settings and numerical parameters) are preserved.
    """
    if isinstance(value, Mapping):
        return {
            key: REDACTED if any(
                part in re.sub(r"[^a-z0-9]", "", str(key).lower())
                for part in _SECRET_PARTS
            ) else redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_config(item) for item in value)
    if isinstance(value, str):
        return _URL.sub(lambda match: redact_url(match.group(0)), value)
    return value
