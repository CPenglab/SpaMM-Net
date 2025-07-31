import pandas as pd
import scanpy as sc
import seaborn as sns

import spamm as spm

# load data
adata = spm.load_test_data(idx = 0)

# training
output, model = spm.run_spamm(
    adata, scale = 3, epochs = 600, lr = 0.001, lr_step = [100, 200], gamma = 0.1,
    rtn_model = True, acc_ctrl=False
)

# clustering
feat = output["feat"].cpu().detach().numpy()
adata_feat = sc.AnnData(X = feat)
sc.pp.neighbors(
    adata_feat,
    n_neighbors=50,
    random_state=2025
)
sc.tl.leiden(
    adata_feat,
    resolution = 0.85,
    flavor="igraph",
    random_state=2025
)
