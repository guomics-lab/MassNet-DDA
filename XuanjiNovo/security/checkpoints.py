"""Explicit trust and immutable staging for legacy object-based checkpoints.

This is not a safe unpickler. A trusted digest approves an exact executable
artifact; it does not make a checkpoint from an unknown publisher safe.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import tempfile


# The model hashes already published by this repository in download_data.py.
OFFICIAL_CHECKPOINT_SHA256 = frozenset({
    "59469325807a2d1b666daad6b3b5c05a5d3b66a489d43b0486485a55a3bde3e6",
    "d909f907aeaa9b215d343b95c6694d85809265ccaabc9f90b1e1b58942cbdcec",
})
MAX_CHECKPOINT_BYTES = 16 * 1024**3


@contextmanager
def verified_checkpoint(filename, trusted_sha256=None, *, max_bytes=MAX_CHECKPOINT_BYTES):
    """Copy, hash and approve an artifact before any object deserialization.

    Custom digests must be supplied by the caller from a trusted source, never
    taken from the checkpoint or a neighboring untrusted manifest. Loading uses
    the private copy so replacing the source after verification cannot change
    the bytes consumed by the framework. Temporary disk capacity is required.
    """
    if trusted_sha256 is not None:
        if not isinstance(trusted_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", trusted_sha256):
            raise ValueError("trusted checkpoint SHA-256 must contain 64 hexadecimal characters")
        approved = {trusted_sha256.lower()}
    else:
        approved = OFFICIAL_CHECKPOINT_SHA256
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("checkpoint byte limit must be a positive integer")
    source = Path(filename)
    if not source.is_file():
        raise FileNotFoundError("checkpoint must be an existing regular file")
    if source.stat().st_size > max_bytes:
        raise ValueError("checkpoint exceeds the byte limit")
    with tempfile.TemporaryDirectory(prefix="xuanjinovo-checkpoint-") as directory:
        staged = Path(directory) / "verified.ckpt"
        digest = hashlib.sha256()
        total = 0
        with source.open("rb") as reader, staged.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("checkpoint exceeds the byte limit")
                digest.update(chunk)
                writer.write(chunk)
        if digest.hexdigest() not in approved:
            raise ValueError(
                "Checkpoint is not approved. Use a verified official release, or pass "
                "--trusted-checkpoint-sha256 from a trusted publisher for your own model. "
                "Legacy checkpoints can execute code; hashing an untrusted file does not make it safe."
            )
        yield str(staged)


def load_verified_checkpoint(model_class, filename, trusted_sha256=None, **kwargs):
    """Load only the verified private copy, without a preliminary model instance."""
    with verified_checkpoint(filename, trusted_sha256) as staged:
        return model_class.load_from_checkpoint(staged, **kwargs)
