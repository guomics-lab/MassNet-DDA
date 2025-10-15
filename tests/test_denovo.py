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


def download_from_mirror(url, destination):
    """Download a file from HTTPS mirror."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(destination, 'wb') as f, tqdm(
                desc=destination,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading from mirror: {e}")
        return False


def test_denovo_process():
    dest_dir = f'test_data'
    demo_model = os.path.join(dest_dir, 'XuanjiNovo_100M_massnet.ckpt')
    demo_file = os.path.join(dest_dir, 'Xbacillus.10k.mgf')
    demo_output = os.path.join(dest_dir, 'test_output')
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading XuanjiNovo_100M_massnet.ckpt...")
    download_from_mirror('https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/XuanjiNovo_100M_massnet.ckpt',
                         demo_model)
    print(f"Finished download XuanjiNovo_100M_massnet.ckpt...")
    # download test file
    print(f"Downloading Xbacillus.10k.mgf...")
    download_from_mirror('https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/bacillus.10k.mgf', demo_file)
    print(f"Finished download Bacillus.10k.mgf...")

    runner = CliRunner()
    # run denovo
    print(f"Processing run denovo")
    runner.invoke(main, ['--mode', 'denovo', '--model', demo_model, '--peak_path', demo_file, '--output', demo_output])
    print(f"Finished run denovo")

    result_file = os.path.join(demo_output, 'denovo.tsv')
    df = pd.read_csv(result_file, sep='\t')
    assert len(df) == 9804
