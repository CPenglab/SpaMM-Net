import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import knn_graph
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GATConv

def multi_corr(x: np.ndarray, dist = ["eu", "cos"], weight = None): # [.6, .4]
    dist = dist if isinstance(dist, list) else [dist]
    mtx = []
    if "eu" in dist:
        x_eu = squareform(pdist(x, metric = "eu"))
        sigma = np.median(x_eu[x_eu != 0])
        x_eu = np.exp(-x_eu**2 / (2*(sigma**2)))
        np.fill_diagonal(x_eu, 0)
        mtx.append(torch.tensor(x_eu))
    if "cos" in dist:
        x_cos = squareform(pdist(x, metric = "cosine"))
        x_cos = 1 - x_cos
        x_cos = (x_cos - x_cos.min())/(x_cos.max() - x_cos.min())
        np.fill_diagonal(x_cos, 0)
        mtx.append(torch.tensor(x_cos))
    if "corr" in dist:
        x_corr = (1 - np.corrcoef(x)) / 2
        mtx.append(torch.tensor(x_corr))

    weight = torch.tensor([1] * len(dist) if weight is None else weight)

    adj = torch.stack(mtx) * weight.reshape(len(dist), 1, 1)
    return adj 

def knn(x: torch.Tensor, k = 8, weight = False):
    value, idx = x.topk(k)
    adj = torch.zeros_like(x)
    value = value if weight else 1.0
    adj.scatter_(dim = -1, index = idx, value = value)
    return adj

class DynamicBalancedLoss(nn.Module):
    def __init__(self, loss_n: int, init_scale: float = 1.0, reg_clamp = [0, 10]):
        super().__init__()

        self.log_sigmas = nn.Parameter(
            torch.full((loss_n,), torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        )
    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        if len(losses) != len(self.log_sigmas):
            raise ValueError(f"Expected {len(self.log_sigmas)} losses, got {len(losses)}")

        with torch.no_grad():
            self.log_sigmas.clamp_(min=0, max=10)
        sigmas = torch.exp(self.log_sigmas)
        
        weighted_losses = [loss / (2 * sigma**2) for loss, sigma in zip(losses, sigmas)]
        
        reg_term = torch.sum(self.log_sigmas)
        
        total_loss = torch.sum(torch.stack(weighted_losses)) + reg_term
        
        return total_loss



def init_weights(m):
    if isinstance(m, nn.Linear):
        with torch.random.fork_rng():
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, GATConv):
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
        with torch.random.fork_rng():
            torch.nn.init.xavier_uniform_(m.lin.weight)
        if m.lin.bias is not None:
            torch.nn.init.constant_(m.lin.bias, 0.0)
        with torch.random.fork_rng():
            torch.nn.init.xavier_uniform_(m.att_dst)
        with torch.random.fork_rng():
            torch.nn.init.xavier_uniform_(m.att_src)

def preprocessing(
    om1_adata, om2_adata, n_comps = 50,
    random_state = 2025
):
    if isinstance(n_comps, list) and len(n_comps) == 2:
        om1_comps, om2_comps = n_comps
    elif isinstance(n_comps, int):
        om1_comps = om2_comps = n_comps
    else:
        raise ValueError("n_comps type should be list or int.")

    sc.pp.normalize_total(om1_adata)
    sc.pp.log1p(om1_adata)
    sc.pp.highly_variable_genes(om1_adata, n_top_genes=3000)
    sc.pp.pca(om1_adata, n_comps = om1_comps, random_state = random_state)

    sc.pp.normalize_total(om2_adata)
    sc.pp.log1p(om2_adata)
    sc.pp.highly_variable_genes(om2_adata, n_top_genes=3000)
    sc.pp.pca(om2_adata, n_comps = om2_comps, random_state = random_state)


def adata_const(
    om1_adata, om2_adata,
    spatial_net_k: int = 8,
    om_net_k: Optional[list] = 8,
    use_pca: Optional[None] = "X_pca",
    om_net_dist = ["eu", "cos"],
    om_net_weight = None
):
    om1 = om1_adata.X if use_pca is None else om1_adata.obsm[use_pca]
    om2 = om2_adata.X if use_pca is None else om2_adata.obsm[use_pca]

    adata = sc.AnnData(
        X = np.concatenate([om1, om2], axis = 1),
        obs = om1_adata.obs
    )
    adata.obsm["spatial"] = om1_adata.obsm["spatial"]

    sp_net = knn_graph(torch.tensor(om1_adata.obsm["spatial"]), k = spatial_net_k)
    adata.uns["sp_net"] = sp_net.detach().cpu().numpy()

    om1_net = multi_corr(om1, dist = om_net_dist, weight = om_net_weight)
    om2_net = multi_corr(om2, dist = om_net_dist, weight = om_net_weight)
    om1_k = om_net_k[0] if isinstance(om_net_k, (list, tuple)) else om_net_k
    om2_k = om_net_k[1] if isinstance(om_net_k, (list, tuple)) else om_net_k
    om1_net = knn(om1_net.sum(0), k = om1_k, weight=False)
    om2_net = knn(om2_net.sum(0), k = om2_k, weight=False)

    adata.uns[f"om1_net"] = dense_to_sparse(om1_net)[0].detach().cpu().numpy()
    adata.uns[f"om2_net"] = dense_to_sparse(om2_net)[0].detach().cpu().numpy()
    return adata


def pca(data, n_components, random_state=2025):
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    pca_ = PCA(n_components=n_components,random_state=random_state)
    return pca_.fit_transform(data)


def set_random_seed(seed, acc_ctrl = False) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(acc_ctrl, warn_only=False)

    os.environ['PYTHONHASHSEED'] = str(seed)
