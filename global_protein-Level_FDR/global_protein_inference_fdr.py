from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from picked_group_fdr import picked_group_fdr

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Protein inference from peptide list (PEP provided)."
    )
    p.add_argument(
        "-i", "--input", default="/ajun/global_fdr/human.tsv",
        help="Input TSV with columns: peptide, proteins, posterior_error_probability"
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="Output TSV for protein groups (default: <input_dir>/protein_groups.tsv)"
    )
    p.add_argument(
        "--sep", default="\t",
        help="Field delimiter of input file (default: tab)"
    )
    p.add_argument(
        "--qvalue", type=float, default=0.01,
        help="Protein group-level FDR threshold on qValue (default: 0.01)"
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    
    output_dir = Path("/ajun/massnet/global_pro_fdr/picked_group_fdr")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = output_dir / "pick_group_fdr.tsv"

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path, sep=args.sep)
    needed = {"peptide", "proteins", "posterior_error_probability"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df['proteins'] = df['proteins'].str.removesuffix(";")
    df["proteins"] = df["proteins"].astype(str).str.split(";")
    df = df.sort_values(by="posterior_error_probability", ascending=True)
    df = df.drop_duplicates(subset="peptide", keep="first")

    peptide_info_split = df.set_index("peptide")[["posterior_error_probability", "proteins"]].to_dict("split")
    peptide_info_dict = dict(zip(peptide_info_split["index"], peptide_info_split["data"]))

    results = picked_group_fdr.get_protein_group_results(peptide_info_dict)

    protein_groups_df = pd.DataFrame(results.protein_group_results)
    for col in ("precursorQuants", "extraColumns"):
        if col in protein_groups_df.columns:
            protein_groups_df = protein_groups_df.drop(columns=[col])

    if protein_groups_df.empty:
        print("No protein groups produced by inference (empty results).")
        return

    protein_groups_df = protein_groups_df.loc[protein_groups_df["qValue"] < args.qvalue]
    protein_groups_df.to_csv(out_path, sep="\t", index=False)

    print(f"Saved {len(protein_groups_df)} protein groups to: {out_path}")

if __name__ == "__main__":
    main()
