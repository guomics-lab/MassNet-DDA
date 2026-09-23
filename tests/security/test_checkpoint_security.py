"""Checkpoint trust-boundary tests; no torch import or checkpoint execution."""

import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from XuanjiNovo.security.checkpoints import (
    OFFICIAL_CHECKPOINT_SHA256, load_verified_checkpoint, verified_checkpoint,
)


class CheckpointSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "model.ckpt"
        self.source.write_bytes(b"synthetic data only; never unpickled")
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_unknown_artifact_rejected_before_loader(self):
        model = mock.Mock()
        with self.assertRaisesRegex(ValueError, "not approved"):
            load_verified_checkpoint(model, self.source)
        model.load_from_checkpoint.assert_not_called()

    def test_source_replacement_does_not_change_loaded_bytes(self):
        original = self.source.read_bytes()
        paths = []
        def inspect_only(path, **kwargs):
            paths.append(Path(path))
            self.source.write_bytes(b"replacement")
            self.assertEqual(Path(path).read_bytes(), original)
            self.assertNotEqual(Path(path), self.source)
            self.assertEqual(kwargs, {"option": 7})
            return "loaded"
        model = mock.Mock()
        model.load_from_checkpoint.side_effect = inspect_only
        self.assertEqual(load_verified_checkpoint(model, self.source, self.digest, option=7), "loaded")
        self.assertFalse(paths[0].exists())

    def test_digest_mismatch_and_invalid_digest(self):
        for digest in ("0" * 64, "bad", True):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                with verified_checkpoint(self.source, digest):
                    self.fail("must not yield unapproved bytes")

    def test_uppercase_digest_and_cleanup_on_loader_failure(self):
        staged_path = None
        with self.assertRaisesRegex(RuntimeError, "loader failed"):
            with verified_checkpoint(self.source, self.digest.upper()) as staged:
                staged_path = Path(staged)
                raise RuntimeError("loader failed")
        self.assertFalse(staged_path.exists())

    def test_resource_limit(self):
        with self.assertRaisesRegex(ValueError, "byte limit"):
            with verified_checkpoint(self.source, self.digest, max_bytes=1):
                self.fail("must reject before loading")

    def test_official_hashes_match_existing_release_manifest(self):
        root = Path(__file__).resolve().parents[2]
        module = ast.parse((root / "download_data.py").read_text(encoding="utf-8"))
        assignment = next(n for n in module.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "DOWNLOADS" for t in n.targets))
        data = ast.literal_eval(assignment.value)
        expected = {v["sha256"] for k, v in data.items() if k.endswith(".ckpt")}
        self.assertEqual(OFFICIAL_CHECKPOINT_SHA256, expected)


if __name__ == "__main__":
    unittest.main()
