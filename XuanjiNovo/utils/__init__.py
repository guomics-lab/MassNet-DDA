"""Utility functions and modules for XuanjiNovo."""

import os
import multiprocessing
from typing import Any, List, Union

from .metadata import record_metadata, get_git_info, get_hardware_info, get_software_info

def n_workers() -> int:
    """
    Get the number of worker processes to use.

    This is determined by the environment variable "SLURM_CPUS_ON_NODE" if running
    on a SLURM cluster, otherwise by the number of CPU cores available.

    Returns
    -------
    int
        The number of worker processes to use.
    """
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    return multiprocessing.cpu_count()

def listify(x: Any) -> List:
    """
    Convert input to a list if it isn't already.

    Parameters
    ----------
    x : Any
        Input to convert to a list.

    Returns
    -------
    List
        The input as a list. If input was None, returns empty list.
        If input was already a list, returns it unchanged.
        Otherwise, returns a single-element list containing the input.
    """
    if x is None:
        return []
    elif isinstance(x, list):
        return x
    return [x]

__all__ = [
    'record_metadata',
    'get_git_info',
    'get_hardware_info',
    'get_software_info',
    'n_workers',
    'listify'
]
