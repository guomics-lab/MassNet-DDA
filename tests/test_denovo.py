import os
import shutil
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from click.testing import CliRunner

import pandas as pd
import requests
from tqdm import tqdm
from XuanjiNovo.XuanjiNovo import main
from download_data import DOWNLOADS, download_from_mirror, verify_checksum


def ensure_verified_download(name, destination):
    """Use the release manifest for both existing files and fresh downloads."""
    artifact = DOWNLOADS[name]
    if not verify_checksum(destination, artifact["sha256"]):
        if not download_from_mirror(artifact["mirror_url"], destination):
            raise RuntimeError("Test artifact download failed")
        if not verify_checksum(destination, artifact["sha256"]):
            os.remove(destination)
            raise ValueError("Test artifact checksum mismatch")


def test_denovo_process():
    dest_dir = f'test_data'
    demo_model = os.path.join(dest_dir, 'XuanjiNovo_100M_massnet.ckpt')
    demo_file = os.path.join(dest_dir, 'bacillus.10k.mgf')
    demo_output = os.path.join(dest_dir, 'test_output')
    os.makedirs(dest_dir, exist_ok=True)
    result_file = os.path.join(demo_output, 'denovo.tsv')
    if os.path.exists(result_file):
        os.remove(result_file)
    print(f"Downloading XuanjiNovo_100M_massnet.ckpt...")
    ensure_verified_download('XuanjiNovo_100M_massnet.ckpt', demo_model)
    print(f"Finished download XuanjiNovo_100M_massnet.ckpt...")
    # download test file
    print(f"Downloading Xbacillus.10k.mgf...")
    ensure_verified_download('bacillus.10k.mgf', demo_file)
    print(f"Finished download Bacillus.10k.mgf...")

    runner = CliRunner()
    # run denovo
    print(f"Processing run denovo")
    result = runner.invoke(main, ['--mode', 'denovo', '--model', demo_model, '--peak_path', demo_file, '--output', demo_output])
    assert result.exit_code == 0, result.output
    print(f"Finished run denovo")

    result_file = os.path.join(demo_output, 'denovo.tsv')
    df = pd.read_csv(result_file, sep='\t')
    assert len(df) == 9804
