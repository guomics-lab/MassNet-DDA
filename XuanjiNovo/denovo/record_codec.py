"""Versioned, data-only spectrum records for LMDB indexes.

The wire format is UTF-8 JSON prefixed with a version marker.  It deliberately
does not support pickle: rebuild old indexes from trusted peak files instead.
Limits bound per-record decoding work, not total database or model memory.
"""

import json
import math
from numbers import Integral, Real

import numpy as np


FORMAT_KEY = b"record_format"
FORMAT_VERSION = b"massnet-json-v1"
RECORD_PREFIX = b"MASSNET-JSON-1\n"
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_PEAKS = 250_000
MAX_ANNOTATION_CHARS = 65_536
MAX_CHARGE = 100
FIELDS = {"precursor_mz", "precursor_charge", "mz_array", "intensity_array", "pep"}
REBUILD_MESSAGE = (
    "Unsupported or legacy index record format. Rebuild the index from trusted "
    "peak files; loading or converting pickle indexes is not supported."
)


def _number(value, field, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must contain real numbers")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must contain finite numbers") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must contain finite numbers")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _array(values, field, *, positive=False, nonnegative=False):
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "iuf":
            raise ValueError(f"{field} must be a one-dimensional numeric array")
    elif not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be a one-dimensional numeric array")
    if len(values) > MAX_PEAKS:
        raise ValueError(f"{field} exceeds the {MAX_PEAKS} peak limit")
    return [_number(value, field, positive=positive, nonnegative=nonnegative)
            for value in values]


def validate_record(record):
    """Return a normalized record, accepting NumPy numbers from peak parsers."""
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("Spectrum record must contain exactly the supported fields")
    charge = record["precursor_charge"]
    if isinstance(charge, (bool, np.bool_)) or not isinstance(charge, Integral):
        raise ValueError("precursor_charge must be an integer")
    if not 1 <= charge <= MAX_CHARGE:
        raise ValueError(f"precursor_charge must be between 1 and {MAX_CHARGE}")
    annotation = record["pep"]
    # Preserve missing annotations and titles; sequence interpretation is not
    # the storage codec's responsibility.
    if annotation is not None and not isinstance(annotation, str):
        raise ValueError("pep must be a string or null")
    if annotation is not None and len(annotation) > MAX_ANNOTATION_CHARS:
        raise ValueError("pep exceeds the annotation length limit")
    mz = _array(record["mz_array"], "mz_array", positive=True)
    intensity = _array(record["intensity_array"], "intensity_array", nonnegative=True)
    if len(mz) != len(intensity):
        raise ValueError("Peak m/z and intensity arrays must have equal lengths")
    return {
        "precursor_mz": _number(record["precursor_mz"], "precursor_mz", positive=True),
        "precursor_charge": int(charge),
        "mz_array": mz,
        "intensity_array": intensity,
        "pep": annotation,
    }


def encode_record(record):
    """Encode validated numeric/string data without executable object reducers."""
    record = validate_record(record)
    encoded = RECORD_PREFIX + json.dumps(
        record, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("Spectrum record exceeds the encoded byte limit")
    return encoded


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate field in spectrum record")
        result[key] = value
    return result


def decode_record(encoded):
    """Reject legacy/object data before decoding a bounded JSON record."""
    if not isinstance(encoded, (bytes, memoryview)):
        raise ValueError("Missing or invalid spectrum record")
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("Spectrum record exceeds the encoded byte limit")
    # LMDB can expose a read-only view; check its length before copying it.
    encoded = bytes(encoded)
    if not encoded.startswith(RECORD_PREFIX):
        raise ValueError(REBUILD_MESSAGE)
    try:
        record = json.loads(encoded[len(RECORD_PREFIX):].decode("utf-8"),
                            object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid spectrum JSON record") from exc
    return validate_record(record)
