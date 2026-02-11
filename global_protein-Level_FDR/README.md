# Global Protein-Level FDR Control Script

This repository provides a Python script for **protein inference with
global protein-level FDR control**, implemented using the
`picked_group_fdr` method from the Käll/Kuster lab.

The script applies the *picked group target--decoy strategy* to control
protein-level false discovery rates (FDR) in large-scale proteomics
datasets.

------------------------------------------------------------------------

## 🔧 Dependencies

This script relies on the official implementation of:

-   **picked_group_fdr**\
    https://github.com/kusterlab/picked_group_fdr

Install via pip:

``` bash
pip install picked_group_fdr
```

------------------------------------------------------------------------

## 🚀 Usage

1.  Install the dependency:

``` bash
pip install picked_group_fdr
```

2.  Run the protein-level FDR inference script:

``` bash
cd ./global_fdr/; python /ajun/massnet/global_pro_fdr/global_protein_inference_fdr.py
```

------------------------------------------------------------------------

## 📌 Methodological Background

Protein-level FDR is controlled globally using the **picked group FDR**
strategy. For methodological details, please refer to the original implementation:

> Käll et al., Picked Group FDR\
> https://github.com/kusterlab/picked_group_fdr

------------------------------------------------------------------------
