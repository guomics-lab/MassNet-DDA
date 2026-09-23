"""CPU-only guard regressions; these do not compile or validate CUDA kernels."""

import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import threading
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "pmc_contract_under_test", ROOT / "XuanjiNovo/denovo/pmc_contract.py"
)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class TensorStub:
    def __init__(self, shape, value=1000.0, floating=True, finite=True):
        self.shape = shape
        self.value = value
        self.floating = floating
        self.finite = finite

    def __getitem__(self, index):
        return Scalar(self.value)

    def all(self):
        return Scalar(self.finite)


class PMCGuardTests(unittest.TestCase):
    def test_default_configuration_and_rounding(self):
        residues = dict(contract.PMC_RESIDUES)
        contract.validate_pmc_config(40, residues)
        residues["G"] += 0.000001
        contract.validate_pmc_config(40, residues)

    def test_incompatible_configuration_is_rejected(self):
        base = dict(contract.PMC_RESIDUES)
        for length in (0, 32, 41, 100):
            with self.subTest(length=length), self.assertRaises(ValueError):
                contract.validate_pmc_config(length, base)
        modified = dict(base)
        modified["G"] += 1
        missing = dict(base)
        del missing["G"]
        extra = dict(base, unexpected=1)
        for residues in (dict(reversed(list(base.items()))), modified, missing, extra):
            with self.subTest(residues=residues), self.assertRaises(ValueError):
                contract.validate_pmc_config(40, residues)

    def test_nonfinite_residue_mass_is_rejected(self):
        for mass in (float("nan"), float("inf"), True, "57"):
            residues = dict(contract.PMC_RESIDUES)
            residues["G"] = mass
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                contract.validate_pmc_config(40, residues)

    def test_mass_budget_without_allocations(self):
        self.assertEqual(contract.PMC_BYTES_PER_GRID_CELL, 7216)
        maximum = contract.PMC_MAX_WORK_BYTES // contract.PMC_BYTES_PER_GRID_CELL
        contract.validate_pmc_grid_length(maximum)
        for length in (maximum + 1, 1000000, 0, -1, True, 1.5):
            with self.subTest(length=length), self.assertRaises(ValueError):
                contract.validate_pmc_grid_length(length)

    def test_scalar_boundaries(self):
        self.assertAlmostEqual(contract.validate_pmc_scalars(1000.0, 0.1), 981.99)
        for mass in (-1, 0, 18.01, 1e6, float("inf"), float("nan")):
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                contract.validate_pmc_scalars(mass, 0.1)
        for tolerance in (-1, float("inf"), float("nan"), 1e40, True):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                contract.validate_pmc_scalars(1000.0, tolerance)

    def guard_harness(self):
        """Execute the actual Python entry guard, replacing only device work."""
        tree = ast.parse((ROOT / "XuanjiNovo/denovo/pmc.py").read_text())
        entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "knapDecode")
        calls = []
        namespace = {
            "validate_pmc_shape": contract.validate_pmc_shape,
            "validate_pmc_scalars": contract.validate_pmc_scalars,
            "torch": types.SimpleNamespace(
                is_floating_point=lambda value: value.floating,
                isfinite=lambda value: value,
            ),
            "_PMC_WORKSPACE_LOCK": threading.Lock(),
            "_knap_decode_locked": lambda *args: calls.append(args) or [1],
        }
        exec(compile(ast.Module(body=[entry], type_ignores=[]), "pmc.py", "exec"), namespace)
        return namespace["knapDecode"], calls

    def test_invalid_inputs_never_enter_device_work(self):
        cases = [
            (TensorStub((1, 32, 28)), TensorStub((1,)), 0.1),
            (TensorStub((1, 40, 21)), TensorStub((1,)), 0.1),
            (TensorStub((2, 40, 28)), TensorStub((1,)), 0.1),
            (TensorStub((1, 40, 28), floating=False), TensorStub((1,)), 0.1),
            (TensorStub((1, 40, 28), finite=False), TensorStub((1,)), 0.1),
            (TensorStub((1, 40, 28)), TensorStub((2,)), 0.1),
            (TensorStub((1, 40, 28)), TensorStub((1,), floating=False), 0.1),
            (TensorStub((1, 40, 28)), TensorStub((1,), value=1e6), 0.1),
            (TensorStub((1, 40, 28)), TensorStub((1,)), float("nan")),
        ]
        for args in cases:
            with self.subTest(args=args):
                entry, calls = self.guard_harness()
                with self.assertRaises(ValueError):
                    entry(*args)
                self.assertEqual(calls, [])

    def test_supported_input_enters_device_work_once(self):
        entry, calls = self.guard_harness()
        self.assertEqual(entry(TensorStub((1, 40, 28)), TensorStub((1,)), 0.1), [1])
        self.assertEqual(len(calls), 1)

    def test_workspace_budget_precedes_device_allocation(self):
        tree = ast.parse((ROOT / "XuanjiNovo/denovo/pmc.py").read_text())
        body = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_knap_decode_locked")
        checks = [node.lineno for node in ast.walk(body) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "validate_pmc_grid_length"]
        allocations = [node.lineno for node in ast.walk(body) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == "zeros"]
        self.assertTrue(checks and allocations)
        self.assertLess(max(checks), min(allocations))

    def test_prediction_worker_errors_reach_caller(self):
        """Execute the original pool block with a failing worker and fake tensors."""
        tree = ast.parse((ROOT / "XuanjiNovo/denovo/model.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Spec2Pep")
        forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        pool = next(node for node in ast.walk(forward) if isinstance(node, ast.With)
                    and "ThreadPoolExecutor" in ast.unparse(node.items[0].context_expr))

        class SliceStub:
            def __getitem__(self, index):
                return None

        def failed_worker(*args):
            raise ValueError("controlled worker failure")

        namespace = dict(ThreadPoolExecutor=ThreadPoolExecutor, batch_size=2,
                         log_prob=SliceStub(), precursors=SliceStub(),
                         output_logits=None, worker=failed_worker)
        with self.assertRaisesRegex(ValueError, "controlled worker failure"):
            exec(compile(ast.Module(body=[pool], type_ignores=[]), "model.py", "exec"), namespace)

    def test_source_tail_guard_precedes_reads(self):
        """Source-order regression only; not a CUDA memory/synchronization test."""
        cases = {
            "best_alignment.cu": ("ctc_alignment_log_alpha_gpu_kernel",),
            "imputer.cu": ("imputer_loss_log_alpha_gpu_kernel", "imputer_loss_backward_log_beta_gpu_kernel"),
        }
        for filename, kernels in cases.items():
            source = (ROOT / "imputer-pytorch/torch_imputer" / filename).read_text()
            for kernel in kernels:
                with self.subTest(kernel=kernel):
                    body = source[source.index(kernel):]
                    guard = body.index("if (b >= batch_size)")
                    self.assertLess(guard, body.index("input_lengths[b]"))
                    self.assertLess(guard, body.index("target_lengths[b]"))
                    self.assertLess(guard, body.index("tg_batch_offsets[b]"))


if __name__ == "__main__":
    unittest.main()
