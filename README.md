# SpaMM-Net
SpaMM-Net: spatially multimodal and multiscale network for representation learning from spatial multi-omics
## Installation
1. Clone this repo.
2. Copy "SpaMM-Net" folder in your project.

## Example
### Quick Start
#### 1. data loading
```
import spamm as spm

# load data
om1_adata, om2_adata = spm.load_test_data(idx = 1)

# data preprocessing (Optional)
spm.preprocessing(
    om1_adata, om2_adata,
    n_comps = 50, random_state = 2025
)

# generate the input adata of SpaMM
adata = spm.adata_const(
    om1_adata, om2_adata,
    spatial_net_k = 8,
    om_net_k = [8, 15]
)
```
#### 2. data training
```
output, model = spm.run_spamm(
    adata, scale = 3, epochs = 600, lr = 0.001, lr_step = [100, 200], gamma = 0.1,
    rtn_model = True
)
```
The 'output' is a dictionary with:
- 'feat' as a fixed key
- Dynamic keys 'scale1', 'scale2', ..., up to the number you specified in the 'scale' parameter.
