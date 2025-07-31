from pathlib import Path
import scanpy as sc
def load_test_data(idx = 0):
    ds_names = ["H3K27me3", "E18_5-S2", "ME13_50um"]
    root = Path(__file__).parent.parent.absolute()
    data_path = root / "datasets"
    adata = sc.read_h5ad(data_path / f"{ds_names[idx]}.h5ad")
    print(f"Loaded {ds_names[idx]}.h5ad successfully.")
    return adata
