"""Configuration logging utilities."""

import json
import os
from datetime import datetime
from typing import Dict, Optional

import yaml
from .model import XuanjiNovoConfig

class ConfigLogger:
    """Logger for configuration management and validation."""

    def __init__(self, output_dir: str):
        """
        Initialize configuration logger.

        Parameters
        ----------
        output_dir : str
            Directory where configuration logs will be saved
        """
        self.output_dir = output_dir
        self.config_dir = os.path.join(output_dir, "config_logs")
        os.makedirs(self.config_dir, exist_ok=True)

    def log_config(self, config: XuanjiNovoConfig, run_id: Optional[str] = None) -> Dict[str, str]:
        """
        Log configuration with validation.

        Parameters
        ----------
        config : XuanjiNovoConfig
            Configuration to log
        run_id : Optional[str]
            Unique identifier for this run. If None, timestamp will be used.

        Returns
        -------
        Dict[str, str]
            Paths to the saved configuration files
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create run-specific directory
        run_dir = os.path.join(self.config_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Save configurations in different formats
        paths = {}
        
        # JSON format (full configuration)
        json_path = os.path.join(run_dir, "config_full.json")
        with open(json_path, 'w') as f:
            json.dump(config.dict(), f, indent=2, sort_keys=True)
        paths['json'] = json_path

        # YAML format (human-readable)
        yaml_path = os.path.join(run_dir, "config.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(config.dict(), f, default_flow_style=False)
        paths['yaml'] = yaml_path

        # Device configuration
        device_path = os.path.join(run_dir, "device_config.json")
        with open(device_path, 'w') as f:
            json.dump(config.get_device_config(), f, indent=2)
        paths['device'] = device_path

        # Summary file (key configurations)
        summary_path = os.path.join(run_dir, "config_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"XuanjiNovo Configuration Summary\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"Model Architecture:\n")
            f.write(f"- Dimensions: {config.model.dim_model}\n")
            f.write(f"- Heads: {config.model.n_head}\n")
            f.write(f"- Layers: {config.model.n_layers}\n\n")
            f.write(f"Training Configuration:\n")
            f.write(f"- Batch Size: {config.training.train_batch_size}\n")
            f.write(f"- Learning Rate: {config.training.learning_rate}\n")
            f.write(f"- Max Epochs: {config.training.max_epochs}\n\n")
            f.write(f"Decoding Configuration:\n")
            f.write(f"- PMC Enabled: {config.decoding.PMC_enable}\n")
            f.write(f"- Beam Size: {config.decoding.n_beams}\n")
            f.write(f"- Refinement Iterations: {config.decoding.refine_iters}\n")
        paths['summary'] = summary_path

        return paths

    def load_config(self, run_id: str) -> XuanjiNovoConfig:
        """
        Load configuration from logs.

        Parameters
        ----------
        run_id : str
            Run identifier to load configuration from

        Returns
        -------
        XuanjiNovoConfig
            Loaded configuration
        """
        json_path = os.path.join(self.config_dir, run_id, "config_full.json")
        with open(json_path) as f:
            config_dict = json.load(f)
        return XuanjiNovoConfig.from_dict(config_dict)

    def list_runs(self) -> Dict[str, Dict]:
        """
        List all logged configuration runs.

        Returns
        -------
        Dict[str, Dict]
            Dictionary of run IDs and their metadata
        """
        runs = {}
        for run_id in os.listdir(self.config_dir):
            run_dir = os.path.join(self.config_dir, run_id)
            if os.path.isdir(run_dir):
                summary_path = os.path.join(run_dir, "config_summary.txt")
                if os.path.exists(summary_path):
                    with open(summary_path) as f:
                        summary = f.read()
                    runs[run_id] = {
                        "timestamp": run_id,
                        "summary_path": summary_path,
                        "summary": summary
                    }
        return runs
