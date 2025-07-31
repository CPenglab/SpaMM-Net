import pandas as pd
import scanpy as sc

import spamm as spm

# load data
om1_adata, om2_adata = spm.load_test_data(idx = 0)

# processing
spm.preprocessing(
    om1_adata, om2_adata,
    n_comps = 50, random_state = 2025
)
adata = spm.adata_const(
    om1_adata, om2_adata,
    spatial_net_k = 8,
    om_net_k = [8, 15]
)

# training
output, model = spm.run_spamm(
    adata, scale = 3, epochs = 600, lr = 0.001, lr_step = [100, 200], gamma = 0.1,
    rtn_model = True
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
    resolution = 1.2,
    flavor="igraph",
    random_state=2025
)
