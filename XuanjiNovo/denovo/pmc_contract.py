"""Dependency-free guards for the fixed-shape Precise Mass Control kernel.

These checks preserve the current kernel's supported configuration. They do not
extend its chemistry or certify its device-side synchronization.
"""

import math
from numbers import Real


PMC_TIME_STEPS = 40
PMC_VOCAB_SIZE = 28
PMC_MAX_WORK_BYTES = 64 * 1024 * 1024
# One int32 path cube and three float32/int32 state planes per mass grid cell.
PMC_BYTES_PER_GRID_CELL = 4 * ((PMC_TIME_STEPS + 1) ** 2 + 3 * (PMC_TIME_STEPS + 1))
PMC_WATER_MASS = 18.01  # Preserve the existing decoder convention.
PMC_RESIDUES = (
    ("G", 57.021464), ("A", 71.037114), ("S", 87.032028),
    ("P", 97.052764), ("V", 99.068414), ("T", 101.047670),
    ("C+57.021", 160.030649), ("L", 113.084064), ("I", 113.084064),
    ("N", 114.042927), ("D", 115.026943), ("Q", 128.058578),
    ("K", 128.094963), ("E", 129.042593), ("M", 131.040485),
    ("H", 137.058912), ("F", 147.068414), ("R", 156.101111),
    ("Y", 163.063329), ("W", 186.079313), ("M+15.995", 147.035400),
    ("N+0.984", 115.026943), ("Q+0.984", 129.042594),
    ("+42.011", 42.010565), ("+43.006", 43.005814),
    ("-17.027", -17.026549), ("+43.006-17.027", 25.980265),
)


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def validate_pmc_config(max_length, residues):
    """Reject configurations incompatible with the kernel's fixed index map."""
    if max_length != PMC_TIME_STEPS:
        raise ValueError("PMC requires max_length=40; disable PMC for other lengths")
    expected_tokens = [token for token, _ in PMC_RESIDUES]
    if list(residues) != expected_tokens:
        raise ValueError("PMC requires the default residue vocabulary and order")
    for token, expected_mass in PMC_RESIDUES:
        mass = _finite_real(residues[token], f"mass for {token}")
        if not math.isclose(mass, expected_mass, rel_tol=0, abs_tol=1e-5):
            raise ValueError(f"PMC requires the default mass for {token}")


def validate_pmc_shape(shape):
    """Validate a single-spectrum probability tensor before any device work."""
    if tuple(shape) != (1, PMC_TIME_STEPS, PMC_VOCAB_SIZE):
        raise ValueError("PMC probabilities must have shape (1, 40, 28)")


def validate_pmc_grid_length(length):
    """Bound the four main device arrays, without allocating them."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("PMC mass grid must have a positive integer length")
    if length * PMC_BYTES_PER_GRID_CELL > PMC_MAX_WORK_BYTES:
        raise ValueError("PMC mass grid exceeds the 64 MiB workspace limit")


def validate_pmc_scalars(precursor_mass, tolerance):
    """Validate scalar inputs and conservatively check the workspace budget."""
    precursor_mass = _finite_real(precursor_mass, "precursor mass")
    tolerance = _finite_real(tolerance, "mass tolerance")
    if precursor_mass <= PMC_WATER_MASS:
        raise ValueError("PMC precursor mass must exceed the water mass")
    # Avoid overflow when passing the tolerance to the kernel as float32.
    if tolerance < 0 or tolerance > 3.4028234663852886e38:
        raise ValueError("PMC mass tolerance must be nonnegative and fit float32")
    residue_mass = precursor_mass - PMC_WATER_MASS
    validate_pmc_grid_length(math.ceil(residue_mass) + 1)
    return residue_mass
