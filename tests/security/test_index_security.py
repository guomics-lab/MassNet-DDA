"""CPU-only index checks: python -m unittest discover -s tests/security.

These tests use real LMDB transactions. The test subclass changes only the map
size to avoid reserving the production 4 TB mapping for a two-record fixture.
"""

import copy
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lmdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from XuanjiNovo.denovo.db_index import DB_Index
from XuanjiNovo.denovo import record_codec as codec


CALLBACKS = []


def harmless_marker():
    CALLBACKS.append("called")
    return fixture_record()


class LegacyMarker:
    def __reduce__(self):
        return harmless_marker, ()


class SmallIndex(DB_Index):
    def _init_db(self):
        self.env = lmdb.open(str(self.db_path), subdir=False,
                             readonly=not self.lock, lock=self.lock,
                             map_size=1024 * 1024)


def fixture_record():
    return {"precursor_mz": 350.1234, "precursor_charge": 2,
            "mz_array": [100., 200.], "intensity_array": [5., 10.], "pep": "PEPTIDE"}


def parser_records(records):
    return SimpleNamespace(
        precursor_mz=[record["precursor_mz"] for record in records],
        precursor_charge=[record["precursor_charge"] for record in records],
        mz_arrays=[record["mz_array"] for record in records],
        intensity_arrays=[record["intensity_array"] for record in records],
        annotations=[record["pep"] for record in records],
    )


class RecordCodecTests(unittest.TestCase):
    def test_numpy_numeric_record_roundtrip(self):
        record = fixture_record()
        record.update(precursor_mz=np.float32(350.125), precursor_charge=np.int64(2),
                      mz_array=np.array([100., 200.], dtype=np.float64),
                      intensity_array=np.array([5., 10.], dtype=np.float32))
        decoded = codec.decode_record(codec.encode_record(record))
        self.assertEqual(decoded["precursor_mz"], float(record["precursor_mz"]))
        self.assertEqual(decoded["precursor_charge"], 2)
        self.assertEqual(decoded["mz_array"], record["mz_array"].tolist())
        self.assertEqual(decoded["intensity_array"], record["intensity_array"].tolist())

    def test_annotations_and_empty_peak_lists_preserved(self):
        for annotation in (None, "", "scan 10 标注", "+42.011PEPTIDE"):
            record = fixture_record()
            record.update(pep=annotation, mz_array=[], intensity_array=[])
            self.assertEqual(codec.decode_record(codec.encode_record(record)), record)

    def test_invalid_fields_and_numeric_values_rejected(self):
        invalid = [
            ("precursor_mz", float("nan")), ("precursor_mz", float("inf")),
            ("precursor_mz", 0), ("precursor_mz", True),
            ("precursor_charge", 0), ("precursor_charge", -2),
            ("precursor_charge", codec.MAX_CHARGE + 1),
            ("precursor_charge", 1.5), ("precursor_charge", True),
            ("precursor_charge", np.bool_(True)),
            ("mz_array", [100., float("nan")]),
            ("mz_array", [[100.], [200.]]),
            ("mz_array", np.array([[100., 200.]])),
            ("mz_array", np.array([100., 200.], dtype=object)),
            ("intensity_array", [1., -1.]), ("intensity_array", [float("inf"), 1.]),
            ("intensity_array", [1.]), ("pep", ["PEPTIDE"]),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=repr(value)):
                record = fixture_record()
                record[key] = value
                with self.assertRaises(ValueError):
                    codec.encode_record(record)
        for record in ({}, {**fixture_record(), "unexpected": 1}):
            with self.assertRaises(ValueError):
                codec.encode_record(record)

    def test_small_budgets_reject_before_large_allocations(self):
        record = fixture_record()
        encoded = codec.encode_record(record)
        with patch.object(codec, "MAX_PEAKS", 1):
            with self.assertRaisesRegex(ValueError, "peak limit"):
                codec.encode_record(record)
            with self.assertRaisesRegex(ValueError, "peak limit"):
                codec.decode_record(encoded)
        with patch.object(codec, "MAX_RECORD_BYTES", len(encoded) - 1):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                codec.decode_record(encoded)
            with self.assertRaisesRegex(ValueError, "byte limit"):
                codec.decode_record(memoryview(encoded))
            with self.assertRaisesRegex(ValueError, "byte limit"):
                codec.encode_record(record)
        with patch.object(codec, "MAX_ANNOTATION_CHARS", 2):
            with self.assertRaisesRegex(ValueError, "annotation"):
                codec.encode_record(record)

    def test_untrusted_json_uses_same_validation(self):
        for key, value in (("precursor_mz", float("nan")), ("precursor_charge", 101),
                           ("mz_array", [100.]), ("pep", {})):
            record = fixture_record()
            record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                codec.decode_record(codec.RECORD_PREFIX + json.dumps(record).encode())
        with self.assertRaises(ValueError):
            codec.decode_record(codec.RECORD_PREFIX + b'{"pep":"A","pep":"B"}')
        for invalid in (None, b"invalid", codec.RECORD_PREFIX + b"\xff", codec.RECORD_PREFIX + b"[]"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                codec.decode_record(invalid)

    def test_pickle_record_never_invokes_callback(self):
        CALLBACKS.clear()
        with self.assertRaisesRegex(ValueError, "Rebuild"):
            codec.decode_record(pickle.dumps(LegacyMarker()))
        self.assertEqual(CALLBACKS, [])


class RealLmdbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "index_without_extension"

    def make_index(self):
        index = SmallIndex(self.path)
        self.addCleanup(index.env.close)
        return index

    def test_new_index_roundtrip_and_reopen_without_extension(self):
        index = self.make_index()
        record = fixture_record()
        index.write_to_db(parser_records([record]), 1)
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], (record["mz_array"], record["intensity_array"],
                                    record["precursor_mz"], record["precursor_charge"], record["pep"]))
        with index.env.begin() as txn:
            self.assertEqual(txn.get(codec.FORMAT_KEY), codec.FORMAT_VERSION)
            self.assertTrue(txn.get(b"0").startswith(codec.RECORD_PREFIX))
        index.env.close()
        reopened = SmallIndex(self.path, lock=False)
        self.addCleanup(reopened.env.close)
        self.assertEqual(reopened[0][-1], "PEPTIDE")

    def test_old_index_rejected_before_reading_object_records(self):
        CALLBACKS.clear()
        env = lmdb.open(str(self.path), subdir=False, map_size=1024 * 1024)
        with env.begin(write=True) as txn:
            txn.put(b"n_spectra", b"1")
            txn.put(b"ms_level", b"2")
            txn.put(b"0", pickle.dumps(LegacyMarker()))
        env.close()
        with self.assertRaisesRegex(ValueError, "Rebuild"):
            SmallIndex(self.path, lock=False)
        self.assertEqual(CALLBACKS, [])
        # Reopening proves failed initialization closed its environment.
        env = lmdb.open(str(self.path), subdir=False, readonly=True, lock=False)
        env.close()

    def test_version_header_does_not_authorize_pickle_record(self):
        CALLBACKS.clear()
        index = self.make_index()
        with index.env.begin(write=True) as txn:
            txn.put(b"0", pickle.dumps(LegacyMarker()))
        with self.assertRaisesRegex(ValueError, "Rebuild"):
            index[0]
        self.assertEqual(CALLBACKS, [])

    def test_invalid_batch_write_is_atomic(self):
        index = self.make_index()
        valid = fixture_record()
        invalid = copy.deepcopy(valid)
        invalid["intensity_array"] = [1.]
        with self.assertRaises(ValueError):
            index.write_to_db(parser_records([valid, invalid]), 2)
        self.assertEqual(len(index), 0)
        with index.env.begin() as txn:
            self.assertIsNone(txn.get(b"0"))
            self.assertEqual(txn.get(b"n_spectra"), b"0")
        index.write_to_db(parser_records([valid]), 1)
        self.assertEqual(len(index), 1)

    def test_actual_lmdb_record_budget_is_checked(self):
        index = self.make_index()
        index.write_to_db(parser_records([fixture_record()]), 1)
        with patch.object(codec, "MAX_RECORD_BYTES", 20):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                index[0]

    def test_missing_record_and_bad_metadata_raise_clear_errors(self):
        index = self.make_index()
        with self.assertRaisesRegex(ValueError, "Missing"):
            index[0]
        with index.env.begin(write=True) as txn:
            txn.put(b"n_spectra", b"not-a-count")
        index.env.close()
        with self.assertRaisesRegex(ValueError, "metadata"):
            SmallIndex(self.path, lock=False)


if __name__ == "__main__":
    unittest.main()
