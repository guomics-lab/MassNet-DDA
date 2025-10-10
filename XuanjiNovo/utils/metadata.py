"""Utilities for recording software and hardware metadata for reproducibility."""

import json
import os
import platform
import sys
import time
from datetime import datetime
import torch
import psutil
import pkg_resources
import git
import logging

logger = logging.getLogger("xuanjinovo")

def get_git_info():
    """Get git repository information if available."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            "git_commit": repo.head.object.hexsha,
            "git_branch": repo.active_branch.name,
            "git_dirty": repo.is_dirty(),
            "git_remote_url": list(repo.remotes[0].urls)[0] if repo.remotes else None
        }
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return {"git_info": "Not a git repository"}

def get_python_packages():
    """Get list of installed Python packages and versions."""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def get_hardware_info():
    """Get hardware information."""
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "name": torch.cuda.get_device_name(i),
                "total_memory": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB",
                "compute_capability": torch.cuda.get_device_capability(i)
            })
    
    return {
        "cpu": {
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "max_frequency": f"{psutil.cpu_freq().max:.2f}MHz" if psutil.cpu_freq() else "Unknown"
        },
        "memory": {
            "total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB"
        },
        "gpus": gpu_info
    }

def get_software_info():
    """Get software environment information."""
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform()
        },
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "Not available",
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "Not available",
        "installed_packages": get_python_packages()
    }

def record_metadata(output_dir, config, model_name="XuanjiNovo"):
    """
    Record comprehensive metadata about the software and hardware environment.
    
    Parameters
    ----------
    output_dir : str
        Directory where to save the metadata
    config : dict
        Model configuration dictionary
    model_name : str
        Name of the model being run
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "config": config,
            "git_info": get_git_info(),
            "hardware_info": get_hardware_info(),
            "software_info": get_software_info()
        }
        
        # Save detailed metadata
        metadata_file = os.path.join(output_dir, "run_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        
        # Save a quick summary
        summary = {
            "timestamp": metadata["timestamp"],
            "git_commit": metadata["git_info"].get("git_commit", "N/A"),
            "python_version": sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "gpu_count": len(metadata["hardware_info"]["gpus"]),
            "gpu_names": [gpu["name"] for gpu in metadata["hardware_info"]["gpus"]],
            "cuda_version": metadata["software_info"]["cuda_version"]
        }
        
        summary_file = os.path.join(output_dir, "run_summary.txt")
        with open(summary_file, "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Metadata recorded in {output_dir}")
        return metadata_file, summary_file
    
    except Exception as e:
        logger.error(f"Failed to record metadata: {str(e)}")
        return None, None

