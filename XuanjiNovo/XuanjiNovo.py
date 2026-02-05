"""The command line entry point for XuanjiNovo."""
import datetime
import functools
import logging
import os
import re
import shutil
import sys
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)

import appdirs
import click
import github
import requests
import torch
import tqdm
import yaml
from pytorch_lightning.lite import LightningLite

from .utils2 import n_workers
from .denovo import model_runner


logger = logging.getLogger("XuanjiNovo")

@click.version_option(version='1.1')
@click.command(help="""XuanjiNovo: A deep learning model for de novo peptide sequencing.

This tool provides three main functionalities:
1. De novo peptide sequencing from MS/MS spectra
2. Model training (from scratch or fine-tuning)
3. Model evaluation on labeled data

The model uses advanced techniques including:
- CTC beam search decoding
- Precise Mass Control (PMC)
- Iterative refinement
- Dynamic masking schedule

For detailed documentation, visit: https://github.com/path/to/XuanjiNovo
""")

@click.option(
    "--mode",
    required=True,
    default="denovo",
    help="\b\nOperation mode:\n"
    '- "denovo": Predict peptide sequences for unknown MS/MS spectra\n'
    '- "train": Train a model from scratch or continue training\n'
    '- "eval": Evaluate model performance on labeled data',
    type=click.Choice(["denovo", "train", "eval"]),
)
@click.option(
    "--model",
    help="Path to model checkpoint (.ckpt file). Required for 'denovo' and 'eval' modes.",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--peak_path",
    required=True,
    help="Path to input peak files, supporting both MGF format and MSDT files (Parquet format). For training, this argument specifies the training dataset.",
)
@click.option(
    "--peak_path_val",
    help="Path to validation data (MGF format). Only used in training mode.",
)
@click.option(
    "--peak_path_test",
    help="Path to test data (MGF format). Only used in training mode.",
)
@click.option(
    "--config",
    help="Path to custom configuration YAML file. If not provided, uses default config.yaml.",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output",
    help="Base output path for logs (.log) and results. Defaults to timestamped file in current directory.",
    type=click.Path(dir_okay=False),
)
@click.option(
    "--pmc-enable/--no-pmc-enable",
    help="Enable/disable Precise Mass Control module. Overrides config setting.",
    default=None,
)
@click.option(
    "--mass_control_tol",
    help="Mass tolerance for PMC module (in Da). Overrides config setting.",
    type=float,
    default=None,
)
@click.option(
    "--refine_iters",
    help="Number of refinement iterations. Overrides config setting.",
    type=int,
    default=None,
)
@click.option(
    "--n_beams",
    help="Number of beams for CTC beam search. Overrides config setting.",
    type=int,
    default=None,
)
@click.option(
    "--batch_size",
    help="Batch size for inference/training. Overrides config setting.",
    type=int,
    default=None,
)
@click.option(
    "--gpu",
    help="Comma-separated list of GPU IDs to use (default: all available)",
    type=str,
    default=None,
)
def main(
    mode: str,
    model: Optional[str],
    peak_path: str,
    peak_path_val: Optional[str],
    peak_path_test: Optional[str],
    config: Optional[str],
    output: Optional[str],
    pmc_enable: Optional[bool],
    mass_control_tol: Optional[float],
    refine_iters: Optional[int],
    n_beams: Optional[int],
    batch_size: Optional[int],
    gpu: Optional[str],
):
   
    # print("hello xiang")
    if output is None:
        output = os.path.join(
            os.getcwd(),
            f"XuanjiNovo_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        )
    else:
        output = os.path.splitext(os.path.abspath(output))[0]

    os.makedirs(output, exist_ok=True)
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
        "{message}",
        style="{",
    )
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    root.addHandler(console_handler)
    file_handler = logging.FileHandler(f"{output}.log")
    file_handler.setFormatter(log_formatter)
    root.addHandler(file_handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("depthcharge").setLevel(logging.INFO)
    logging.getLogger("github").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Read and validate parameters from the config file.
    if config is None:
        config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "config.yaml"
        )
    config_fn = config
    try:
        from .config import XuanjiNovoConfig, ConfigLogger
        
        # Load and validate configuration
        config_obj = XuanjiNovoConfig.from_yaml(config_fn)
        
        # Set up configuration logging
        config_logger = ConfigLogger(output)
        config_paths = config_logger.log_config(config_obj)
        
        logger.info(f"Configuration validated and logged:")
        logger.info(f"  Full config: {config_paths['json']}")
        logger.info(f"  Summary: {config_paths['summary']}")
        logger.info(f"  Device config: {config_paths['device']}")
        
        # Convert validated config back to flat dict for backward compatibility
        config2 = config_obj.to_flat_dict()
        
    except:
        print("skipping pydantic validation")
        config2 = None
    if config2:
        with open(config_fn) as f_in:
            config = yaml.safe_load(f_in)
        # Old type validation code remains as fallback
        config_types = dict(
            random_seed=int,
            n_peaks=int,
            min_mz=float,
            max_mz=float,
            min_intensity=float,
            remove_precursor_tol=float,
            max_charge=int,
            precursor_mass_tol=float,
            isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
            dim_model=int,
            n_head=int,
            dim_feedforward=int,
            n_layers=int,
            dropout=float,
            dim_intensity=int,
            max_length=int,
            n_log=int,
            warmup_iters=int,
            max_iters=int,
            learning_rate=float,
            weight_decay=float,
            train_batch_size=int,
            predict_batch_size=int,
            n_beams=int,
            max_epochs=int,
            num_sanity_val_steps=int,
            train_from_scratch=bool,
            save_model=bool,
            model_save_folder_path=str,
            save_weights_only=bool,
            every_n_train_steps=int,
        )
        for k, t in config_types.items():
            try:
                if config[k] is not None:
                    config[k] = t(config[k])
            except (TypeError, ValueError) as e:
                logger.error("Incorrect type for configuration value %s: %s", k, e)
                raise TypeError(f"Incorrect type for configuration value {k}: {e}")
        
        config["residues"] = {
            str(aa): float(mass) for aa, mass in config["residues"].items()
        }
    # Override config with command-line arguments if provided
    if pmc_enable is not None:
        config["PMC_enable"] = pmc_enable
    if mass_control_tol is not None:
        config["mass_control_tol"] = mass_control_tol
    if refine_iters is not None:
        config["refine_iters"] = refine_iters
    if n_beams is not None:
        config["n_beams"] = n_beams
    if batch_size is not None:
        config["predict_batch_size"] = batch_size
        config["train_batch_size"] = batch_size

    # Handle GPU selection
    if gpu is not None:
        # Set visible devices before any CUDA initialization
        gpu_ids = [id.strip() for id in gpu.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        logger.info(f"Using GPUs: {gpu_ids}")
    
    # Add extra configuration options and scale by the number of GPUs.
    n_gpus = torch.cuda.device_count()
    config["n_workers"] = n_workers()
    if n_gpus > 1:
        config["train_batch_size"] = config["train_batch_size"] // n_gpus
    
    logger.info(f"Number of available GPUs: {n_gpus}")
    if n_gpus > 0:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        logger.info(f"GPU devices: {gpu_names}")

    import random
    if(config["random_seed"]==-1):
        config["random_seed"]=random.randint(1, 9999)

    if not 'result_output_dir' in config:
        config["result_output_dir"] = output

    LightningLite.seed_everything(seed=config["random_seed"], workers=True)

    # Log the active configuration.
    logger.debug("mode = %s", mode)
    logger.debug("model = %s", model)
    logger.debug("peak_path = %s", peak_path)
    logger.debug("peak_path_val = %s", peak_path_val)
    logger.debug("peak_path_test = %s", peak_path_test)
    logger.debug("config = %s", config_fn)
    logger.debug("output = %s", output)
    for key, value in config.items():
        logger.debug("%s = %s", str(key), str(value))

    # Run XuanjiNovo in the specified mode.
    if mode == "denovo":
        logger.info("Predict peptide sequences with XuanjiNovo.")
        writer = None
        # writer.set_metadata(
        #     config, peak_path=peak_path, model=model, config_filename=config_fn
        # )
        model_runner.predict(peak_path, model, config, writer)
        #writer.save()
    elif mode == "eval":
        logger.info("Evaluate a trained XuanjiNovo model.")
        model_runner.evaluate(peak_path, model, config)
    elif mode == "train":
        logger.info("Train the XuanjiNovo model.")
        model_runner.train(peak_path, peak_path_val, peak_path_test, model, config)

if __name__ == "__main__":
    main()
