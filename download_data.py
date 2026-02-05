#!/usr/bin/env python3
"""
Download and verify model checkpoints and datasets.
Supports both Google Drive and direct HTTPS downloads.
"""

import os
import hashlib
import argparse
from pathlib import Path
import requests
from tqdm import tqdm

# Data URLs and their corresponding checksums
DOWNLOADS = {
    "XuanjiNovo_100M_massnet.ckpt": {
        "gdrive_id": "1BtEYZ9FuWvQub2YQEHYMy5l2Y7bcmQDr",
        "mirror_url": "https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/XuanjiNovo_100M_massnet.ckpt",
        "sha256": "59469325807a2d1b666daad6b3b5c05a5d3b66a489d43b0486485a55a3bde3e6"
    },
    "XuanjiNovo_130M_massnet_massivekb.ckpt": {
        "gdrive_id": "1dcbdn5tV5x2tmUKT7nJe8deqMwGzpx4E",
        "mirror_url": "https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/XuanjiNovo_130M_massnet_massivekb.ckpt",
        "sha256": "d909f907aeaa9b215d343b95c6694d85809265ccaabc9f90b1e1b58942cbdcec"
    },
    "bacillus.10k.mgf": {
        "gdrive_id": "1HqfCETZLV9ZB-byU0pqNNRXbaPbTAceT",
        "mirror_url": "https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/bacillus.10k.mgf",
        "sha256": "47eba8c23197214ca7f40200c53ca2abe96a7824b613377738526353a67264df"
    }
}

def calculate_sha256(filepath):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_checksum(filepath, expected_sha256):
    """Verify file integrity using SHA-256."""
    if not os.path.exists(filepath):
        return False
    actual_sha256 = calculate_sha256(filepath)
    return actual_sha256.lower() == expected_sha256.lower()

def download_from_gdrive(file_id, destination):
    """Download a file from Google Drive."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False

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

def main():
    parser = argparse.ArgumentParser(description="Download and verify model checkpoints and datasets")
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save downloaded files')
    parser.add_argument('--prefer-mirror', action='store_true',
                       help='Prefer HTTPS mirror over Google Drive')
    parser.add_argument('--files', nargs='+', choices=list(DOWNLOADS.keys()),
                       help='Specific files to download. If not specified, downloads all.')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = args.files if args.files else DOWNLOADS.keys()

    for filename in files_to_download:
        file_info = DOWNLOADS[filename]
        destination = output_dir / filename
        destination = str(destination)

        # Check if file exists and is valid
        if verify_checksum(destination, file_info['sha256']):
            print(f"{filename} already exists and checksum matches.")
            continue

        print(f"\nDownloading {filename}...")
        success = False

        if args.prefer_mirror:
            # Try mirror first, then fallback to Google Drive
            if download_from_mirror(file_info['mirror_url'], destination):
                success = True
            elif download_from_gdrive(file_info['gdrive_id'], destination):
                success = True
        else:
            # Try Google Drive first, then fallback to mirror
            if download_from_gdrive(file_info['gdrive_id'], destination):
                success = True
            elif download_from_mirror(file_info['mirror_url'], destination):
                success = True

        if not success:
            print(f"Failed to download {filename}")
            continue

        # Verify downloaded file
        if verify_checksum(destination, file_info['sha256']):
            print(f"Successfully downloaded and verified {filename}")
        else:
            print(f"Checksum verification failed for {filename}")
            os.remove(destination)

if __name__ == "__main__":
    main()
