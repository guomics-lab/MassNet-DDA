"""Artifact security regressions; run with Python's standard-library unittest.

The helper modules are imported normally by file. Production metadata/config
writers are compiled from their original AST with hardware/serializer stand-ins,
so the tests exercise real export methods without importing PyTorch or Pydantic.
No build script is executed and no external archive is downloaded.
"""

import ast
import copy
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, Optional
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


redaction = load_module("artifact_redaction", "XuanjiNovo/security/redaction.py")
safe_archive = load_module("safe_archive", "ctcdecode-master/safe_archive.py")


def source_definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError("A tested production definition disappeared")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


class RedactionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "neptune_api_token": "PRIVATE_TOKEN",
            "nested": [{"api-Key": "PRIVATE_KEY", "Password": "PRIVATE_PASSWORD"},
                       {"clientSecret": "PRIVATE_SECRET", "credentials": "PRIVATE_CREDENTIAL"}],
            "remote": "https://USER:PRIVATE_URL_PASSWORD@example.org/repo.git?key=PRIVATE_QUERY#PRIVATE_FRAGMENT",
            "description": "Source: ssh://USER:PRIVATE_SSH_PASSWORD@example.org/repo?signature=PRIVATE_SIGNATURE",
            "train_batch_size": 400,
            "residues": {"C+57.021": 160.030649},
        }

    def assert_no_credentials(self, data):
        text = json.dumps(data)
        self.assertNotIn("PRIVATE_", text)
        self.assertNotIn("USER", text)

    def test_recursive_redaction_does_not_mutate_runtime_config(self):
        original = copy.deepcopy(self.config)
        result = redaction.redact_config(self.config)
        self.assert_no_credentials(result)
        self.assertEqual(self.config, original)
        self.assertEqual(result["train_batch_size"], 400)
        self.assertEqual(result["residues"], original["residues"])
        self.assertEqual(result["remote"], "https://example.org/repo.git")

    def test_url_query_is_removed_without_assuming_parameter_names(self):
        self.assertEqual(redaction.redact_url("https://a:b@[::1]:443/data?x=PRIVATE#PRIVATE"),
                         "https://[::1]:443/data")
        self.assertEqual(redaction.redact_url("https://a:b@[invalid/path"), "[REDACTED]")
        self.assertEqual(redaction.redact_url("git@example.org:project/repo.git"),
                         "git@example.org:project/repo.git")

    def test_apostrophes_in_url_credentials_do_not_disclose_password_suffixes(self):
        for value in (
            "https://USER:PRIVATE_START'PRIVATE_SUFFIX@example.org/repo.git?PRIVATE_QUERY",
            "Source: ssh://USER:PRIVATE_START'PRIVATE_SUFFIX@example.org/repo.git",
        ):
            with self.subTest(value=value):
                result = redaction.redact_config({"remote": value})
                self.assert_no_credentials(result)
                self.assertIn("example.org/repo.git", result["remote"])

    def test_real_metadata_writer_and_git_provenance_do_not_export_credentials(self):
        remote = SimpleNamespace(urls=[self.config["remote"]])
        repo = SimpleNamespace(head=SimpleNamespace(object=SimpleNamespace(hexsha="abc")),
                               active_branch=SimpleNamespace(name="main"),
                               is_dirty=lambda: False, remotes=[remote])
        fake_git = SimpleNamespace(Repo=lambda **kwargs: repo,
                                   InvalidGitRepositoryError=RuntimeError, NoSuchPathError=ValueError)
        namespace = source_definitions("XuanjiNovo/utils/metadata.py",
                                       {"get_git_info", "record_metadata"}, {
            "git": fake_git, "redact_url": redaction.redact_url,
            "redact_config": redaction.redact_config, "os": os, "json": json,
            "datetime": datetime, "sys": sys, "logger": logging.getLogger("artifact-test"),
            "torch": SimpleNamespace(__version__="stand-in"),
            "get_hardware_info": lambda: {"gpus": []},
            "get_software_info": lambda: {"cuda_version": "Not available"},
        })
        with tempfile.TemporaryDirectory() as directory:
            paths = namespace["record_metadata"](directory, self.config)
            self.assertTrue(all(paths))
            metadata = json.loads(Path(paths[0]).read_text())
            self.assert_no_credentials(metadata)
            self.assertEqual(metadata["git_info"]["git_remote_url"], "https://example.org/repo.git")
            self.assertEqual(self.config["neptune_api_token"], "PRIVATE_TOKEN")

    def test_real_config_logger_passes_only_redacted_values_to_serializers(self):
        # The YAML stand-in captures the actual value provided by log_config.
        yaml_inputs = []

        def dump_yaml(value, stream, **kwargs):
            yaml_inputs.append(copy.deepcopy(value))
            json.dump(value, stream)

        namespace = source_definitions("XuanjiNovo/config/logger.py", {"ConfigLogger"}, {
            "redact_config": redaction.redact_config, "os": os, "json": json,
            "datetime": datetime, "Dict": Dict, "Optional": Optional,
            "XuanjiNovoConfig": object, "yaml": SimpleNamespace(dump=dump_yaml),
        })
        config = SimpleNamespace(
            dict=lambda: self.config, get_device_config=lambda: {"gpu_count": 0},
            model=SimpleNamespace(dim_model=400, n_head=8, n_layers=9),
            training=SimpleNamespace(train_batch_size=400, learning_rate=0.00035, max_epochs=150),
            decoding=SimpleNamespace(PMC_enable=True, n_beams=5, refine_iters=3),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = namespace["ConfigLogger"](directory).log_config(config, "regression")
            for path in paths.values():
                self.assert_no_credentials(Path(path).read_text())
            self.assertEqual(len(yaml_inputs), 1)
            self.assert_no_credentials(yaml_inputs[0])

    def test_real_configuration_save_method_redacts_nested_credentials(self):
        source = ast.parse((ROOT / "XuanjiNovo/config/model.py").read_text())
        config_class = next(node for node in source.body
                            if isinstance(node, ast.ClassDef) and node.name == "XuanjiNovoConfig")
        method = next(node for node in config_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "save_to_file")
        namespace = {"json": json, "redact_config": redaction.redact_config}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "config/model.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "config.json"
            namespace["save_to_file"](SimpleNamespace(json=lambda: json.dumps(self.config)), output)
            self.assert_no_credentials(json.loads(output.read_text()))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.archive = self.directory / "source.tar.gz"
        self.destination = self.directory / "third_party"

    def write_archive(self, entries):
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, kind, content in entries:
                member = tarfile.TarInfo(name)
                member.type = kind
                if kind == tarfile.REGTYPE:
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
                else:
                    member.linkname = "../../outside"
                    archive.addfile(member)

    def extract(self, **kwargs):
        return safe_archive.extract_source_archive(self.archive, self.destination, "source", **kwargs)

    def assert_rejected_without_publishing(self, entries, **kwargs):
        self.write_archive(entries)
        with self.assertRaises(ValueError):
            self.extract(**kwargs)
        self.assertFalse((self.destination / "source").exists())
        self.assertFalse((self.directory / "outside").exists())
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_normal_source_tree_extracts_and_preserves_contents(self):
        self.write_archive([("./source/", tarfile.DIRTYPE, b""),
                            ("source/include/file.h", tarfile.REGTYPE, b"safe source")])
        target = self.extract()
        self.assertEqual((target / "include/file.h").read_bytes(), b"safe source")
        self.assertEqual(sorted(path.name for path in self.destination.iterdir()), ["source"])

    def test_traversal_absolute_and_platform_paths_are_rejected_before_any_write(self):
        for path in ("../outside", "/outside", "C:/outside", "source/../../outside",
                     "source/..\\outside", "source/file:stream", "source/NUL", "source/trailing."):
            with self.subTest(path=path):
                self.assert_rejected_without_publishing([
                    ("source/valid.h", tarfile.REGTYPE, b"valid"), (path, tarfile.REGTYPE, b"bad")])

    def test_links_devices_fifo_and_sparse_files_are_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE,
                     tarfile.BLKTYPE, tarfile.FIFOTYPE, tarfile.GNUTYPE_SPARSE):
            with self.subTest(kind=kind):
                self.assert_rejected_without_publishing([("source/evil", kind, b"")])

    def test_windows_device_aliases_are_rejected_without_opening_devices(self):
        for name in ("CON", "con.txt", "NUL .txt", "AUX.dat", "PRN", "COM1", "LPT9.log",
                     "COM\u00b9", "COM\u00b2.txt", "COM\u00b3", "LPT\u00b9", "LPT\u00b2.txt", "LPT\u00b3",
                     "CONIN$", "CONOUT$.log", "trailing ", "trailing.", "trailing . ",
                     "invalid?", "invalid*", "invalid|", "control\x01name"):
            with self.subTest(name=ascii(name)):
                self.assert_rejected_without_publishing([
                    ("source/valid.h", tarfile.REGTYPE, b"valid"),
                    ("source/" + name, tarfile.REGTYPE, b"must never reach a device")])

    def test_unexpected_root_cannot_replace_sibling_dependency(self):
        self.assert_rejected_without_publishing([("kenlm/setup.py", tarfile.REGTYPE, b"bad")])

    def test_duplicate_and_file_directory_conflicts_are_rejected(self):
        for paths in (("source/a", "source/a"), ("source/a", "source/a/file")):
            with self.subTest(paths=paths):
                self.assert_rejected_without_publishing([(path, tarfile.REGTYPE, b"x") for path in paths])

    def test_member_count_and_size_limits_are_enforced(self):
        entries = [("source/a", tarfile.REGTYPE, b"123"), ("source/b", tarfile.REGTYPE, b"456")]
        for limits in ({"max_members": 1}, {"max_file_size": 2}, {"max_total_size": 5}):
            with self.subTest(limits=limits):
                self.assert_rejected_without_publishing(entries, **limits)

    def test_existing_source_tree_is_never_overwritten(self):
        self.write_archive([("source/a", tarfile.REGTYPE, b"replacement")])
        (self.destination / "source").mkdir(parents=True)
        original = self.destination / "source/a"
        original.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.extract()
        self.assertEqual(original.read_bytes(), b"original")

    def test_setup_download_extract_uses_safe_extraction_for_cached_archives(self):
        # Compile this one function; setup() and its downloads/builds never run.
        namespace = source_definitions("ctcdecode-master/setup.py", {"download_extract"}, {
            "os": os, "extract_source_archive": safe_archive.extract_source_archive,
            "urllib": SimpleNamespace(request=SimpleNamespace(
                urlretrieve=lambda *args: self.fail("Unexpected network request"))),
        })
        self.write_archive([("source/../../outside", tarfile.REGTYPE, b"bad")])
        with self.assertRaises(ValueError):
            namespace["download_extract"]("https://example.invalid/source.tar.gz", str(self.archive))
        self.assertFalse((self.directory.parent / "outside").exists())


if __name__ == "__main__":
    unittest.main()
