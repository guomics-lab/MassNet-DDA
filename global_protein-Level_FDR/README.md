# Global Protein-Level FDR Control

This script applies the *picked group target--decoy strategy* developed by the Kuster lab for global protein-level FDR control in proteomics analyses.

------------------------------------------------------------------------

## 🔧 Dependencies

This script relies on the official implementation of:

-   **picked_group_fdr**\: https://github.com/kusterlab/picked_group_fdr

------------------------------------------------------------------------

## 🚀 Usage

1.  Install the dependency:

``` bash
pip install picked_group_fdr
```

2.  Run the protein-level FDR inference script:

``` bash
cd ./global_fdr/; python /ajun/massnet/global_protein-Level_FDR/global_protein_inference_fdr.py
```

------------------------------------------------------------------------
