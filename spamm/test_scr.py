from pathlib import Path
import scanpy as sc
def load_test_data(idx):
    ds_names = [ "H3K27me3","E18_5-S2", "ME13_50um"]
    root = Path(__file__).parent.parent.absolute() / "datasets/"
    data_path = root / ds_names[idx]
    om1_adata = sc.read_h5ad(data_path / "rna_adata.h5ad")
    om2_adata = sc.read_h5ad(data_path / "peak_adata.h5ad")
    return om1_adata, om2_adata
