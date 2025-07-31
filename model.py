from tqdm import tqdm
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torch_geometric.nn import GATConv, GCNConv

from .utils import *

class SC_attn(nn.Module):
    # Spatially-guided Cross-Attention 
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.q = nn.Linear(in_ch, out_ch)
        
        self.om1_k = nn.Linear(in_ch, out_ch)
        self.om2_k = nn.Linear(in_ch, out_ch)
        
        self.om_v = nn.Linear(3 * in_ch, out_ch)
        
        self.conf1 = nn.Linear(2 * in_ch, out_ch) # 置信度
        self.conf2 = nn.Linear(2 * in_ch, out_ch)
    def forward(self, sp, om1, om2):
        om_cat = torch.cat([sp, om1, om2], dim = -1)
          
        q = self.q(sp)
        om1_k = self.om1_k(om1)
        om2_k = self.om2_k(om2)
        om_v = self.om_v(om_cat)
        
        attn1 = torch.softmax(q @ om1_k.T / (q.shape[-1] ** .5), dim = -1) @ om_v
        attn2 = torch.softmax(q @ om2_k.T / (q.shape[-1] ** .5), dim = -1) @ om_v
        
        conf1 = torch.sigmoid(self.conf1(torch.cat([sp, om1], dim = -1)))
        conf2 = torch.sigmoid(self.conf1(torch.cat([sp, om2], dim = -1)))
        conf = torch.softmax(torch.stack([conf1, conf2], dim = -2), dim = -2)
        
        out = conf[:, 0, :] * attn1 + conf[:, 1, :] * attn2
        return out


class MSF_sample(nn.Module):
    def __init__(self, in_ch: int, scale_n: int):
        super().__init__()
        self.scale_n = scale_n
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_ch, in_ch),
                nn.LayerNorm(in_ch)
            )
            for i in range(scale_n)
        ])
        
        self.weight = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_ch, 1), 
                nn.Sigmoid()
            )
            for _ in range(scale_n)
        ])

    def forward(self, *args):
       
        weight = []
        args_proj = []
        for i in range(len(args)):
            args_proj.append(self.proj[i](args[i]))
            weight.append(self.weight[i](args_proj[i]))
        weight = torch.cat(weight, dim = -1)
        weight = torch.softmax(weight, dim = -1)
        weight = weight.transpose(-2, -1)
        weight = weight.unsqueeze(-1)
        
        data = torch.stack(args_proj, dim = 0)
        data = data * weight

        return data.sum(0), weight

class MFUnit(nn.Module):
    # Multimodal Fusion Unit
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

        self.omics_gcn = GATConv(in_ch, out_ch, dropout=0.0)

        self.gmu = SC_attn(out_ch, out_ch)
        

    def forward(self, omics, sp_net, om1_net, om2_net):
        omics_sp = self.omics_gcn(omics, edge_index = sp_net)
        if om1_net.shape[0] == 2 and om2_net.shape[0] == 2:
            omics_om2 = self.omics_gcn(omics, edge_index = om2_net)
            omics_om1 = self.omics_gcn(omics, edge_index = om1_net)
        else:
            omics_om2 = om2_net
            omics_om1 = om1_net

        omics = self.gmu(omics_sp, omics_om1, omics_om2)

        return omics, omics_om1, omics_om2

class MSUnit(nn.Module):
    # Multi-scale Unit for k-hop
    def __init__(self, in_ch, scales: int = 3):
        # scales: k-hop
        super().__init__()
        self.gcns = nn.ModuleList([
            MFUnit(in_ch, in_ch)
            for i in range(scales)
        ])

    def forward(self, omics, sp_net, om1_net, om2_net):
        gmu_emb, om1_emb, om2_emb = self.gcns[0](omics, sp_net, om1_net, om2_net)
        embs = [gmu_emb]

        for i in range(1, len(self.gcns)):
            embs.append(self.gcns[i](embs[-1], sp_net, om1_emb, om2_emb)[0])

        return embs

class SGU(nn.Module):
    # Cross-Scale Gating Unit 
    def __init__(self, in_ch, up_ch):
        super().__init__()
        self.up_proj = nn.Linear(up_ch, in_ch)
        self.up_g = nn.Linear(2*in_ch, in_ch)
        
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_in, x_up):
        x_up = F.tanh(self.up_proj(x_up))
        x = torch.cat([x_in, x_up], dim = -1)
        g = self.up_g(x)
        g = torch.sigmoid(g)
        return x_in + self.alpha * g * x_up

class Encoder(nn.Module):
    def __init__(self, in_ch, scales: int):
        super().__init__()

        self.msuit = MSUnit(in_ch, scales = scales)
        self.up_list = nn.ModuleList([
            SGU(in_ch, in_ch)
            for i in range(scales)
        ])
        self.msf = MSF_sample(in_ch = in_ch, scale_n = scales)
    def forward(
        self, omics: torch.Tensor,
        sp_net: torch.Tensor,
        om1_net: torch.Tensor,
        om2_net: torch.Tensor):

        embs = self.msuit(omics, sp_net, om1_net, om2_net)

        emb_ups = [self.up_list[0](embs[0], embs[0])]
        for i in range(1, len(self.up_list)):
            emb_ups.append(self.up_list[i](embs[i], embs[i - 1]))

        emb, self.msf_w = self.msf(*emb_ups)

        return emb, emb_ups

class SpaMM(nn.Module):
    def __init__(
        self, in_ch, scales: int = 3, device = None
    ):
        super().__init__()

        self.encoder = Encoder(in_ch, scales)

        self.de_list = nn.ModuleList([
            GATConv(in_ch, in_ch // 2, dropout = 0.0), # rna
            GATConv(in_ch, in_ch // 2, dropout = 0.0) # peak
        ])

        self.dl_recon = DynamicBalancedLoss(loss_n = 2)
        self.dl_total = DynamicBalancedLoss(loss_n = 3)

        self.apply(lambda m: init_weights(m))

        if device == "cpu":
            device_finder = "cpu"
        else:
            device_finder = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_finder)

    def forward(
        self, omics: torch.Tensor,
        sp_net: torch.Tensor,
        om1_net: torch.Tensor,
        om2_net: torch.Tensor,
        ):

        emb, scale_embs = self.encoder(omics, sp_net, om1_net, om2_net)
        recons = [
            self.de_list[i](emb, edge_index = sp_net)
            for i in range(len(self.de_list))
        ]


        emb_, scale_embs_ = self.encoder(
            torch.cat(recons, dim = -1),
            sp_net, om1_net, om2_net
        )

        return emb, recons, emb_, scale_embs, scale_embs_

    def loss_recon(self, recons: list, omics, recon_w = None):
        x1, x2 = torch.chunk(omics, chunks = 2, dim = -1)
        if recon_w is None:
            loss = self.dl_recon(
                F.mse_loss(recons[0], x1),
                F.mse_loss(recons[1], x2)           
            )
        else:
            assert len(recon_w)==2, "len(recon_w) must be 2 if recon_w is not None."
            loss = recon_w[0] * F.mse_loss(recons[0], x1) +\
                   recon_w[1] * F.mse_loss(recons[1], x2)           
        return loss

    def loss_scales_struct(self, scale_embs_, scale_embs, w = [.1, .2, .2]):
        # latent_space_structural_loss
        loss = [w[i] * F.mse_loss(scale_embs_[i], scale_embs[i])
            for i in range(len(scale_embs))]
        loss = torch.tensor(loss).sum()
        return loss

    def loss_top_scale_align(self, scale_embs):
        # Top-scale features supervised by middle-scale
        return F.mse_loss(scale_embs[-1], scale_embs[-2].detach())

    def loss_emb_struct(self, emb_, emb):
        return F.mse_loss(emb_, emb)
        
    def loss_fn(self,
        recons: list,omics,        
        emb_, emb,
        scale_embs_, scale_embs,
        recon_w,
        scales_w,
        total_w,
        top_align_w):
        loss1 = self.loss_recon(recons, omics, recon_w)
        loss2 = self.loss_scales_struct(scale_embs_, scale_embs, w = scales_w)
        loss3 = self.loss_emb_struct(emb_, emb)
        loss4 = self.loss_top_scale_align(scale_embs)

        if total_w is None:
            total = self.dl_total(loss1, loss2, loss3) + top_align_w * loss4
        else:
            assert len(total_w) == 3, "len(total_w) must be 3 if total_w is not None."
            total = total_w[0] * loss1 + total_w[1] * loss2 + \
                    total_w[2] * loss3 + top_align_w * loss4

        loss = {
            "recon": loss1,
            "scales": loss2,
            "emb": loss3,
            "align": loss4,
            "total": total
        }        
        return loss

    def fit(
        self, 
        omics: torch.Tensor, 
        sp_net: torch.Tensor,
        om1_net: torch.Tensor,
        om2_net: torch.Tensor,
        epochs = 600,
        lr = 0.001,
        lr_step = 100,
        gamma = 0.1,
        recon_w = None,
        scales_w = [0.1, 0.2, 0.2],
        total_w = None,
        top_align_w = 0.1,
        weight_decay = 1e-4
):

        omics = omics.to(self.device)

        sp_net = sp_net.to(self.device)
        om1_net = om1_net.to(self.device)
        om2_net = om2_net.to(self.device)

        self.to(self.device)
        self.train()

        optimizer = Adam(self.parameters(), lr = lr, weight_decay = weight_decay)

        if isinstance(lr_step, int):
            lr_step = [lr_step]
        steplr = MultiStepLR(optimizer, milestones = lr_step, gamma = gamma)

        loss1_log = np.zeros(0)
        loss2_log = np.zeros(0)
        loss3_log = np.zeros(0)
        loss4_log = np.zeros(0)
        loss_total_log = np.zeros(0)

        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()

            emb, recons, emb_, scale_embs, scale_embs_ = self(
                omics, sp_net, om1_net, om2_net
            )
            loss = self.loss_fn(
                recons, omics, emb_, emb,
                scale_embs_, scale_embs,
                recon_w,
                scales_w,
                total_w,
                top_align_w
            )
            loss["total"].backward()
            optimizer.step()

            loss1_log = np.append(loss1_log, loss["recon"].cpu().detach().numpy())
            loss2_log = np.append(loss2_log, loss["scales"].cpu().detach().numpy())
            loss3_log = np.append(loss3_log, loss["emb"].cpu().detach().numpy())
            loss4_log = np.append(loss4_log, loss["align"].cpu().detach().numpy())

            loss_total_log = np.append(loss_total_log, loss["total"].cpu().detach().numpy())

        self.loss_log = pd.DataFrame({
            "epoch": list(range(epochs)),
            "recon": loss1_log,
            "scales": loss2_log,
            "emb": loss3_log,
            "align": loss4_log,
            "total_loss": loss_total_log
        })

        return self

    def trans(
        self, omics: torch.Tensor, 
        sp_net: torch.Tensor,
        om1_net: torch.Tensor,
        om2_net: torch.Tensor):

        omics = omics.to(self.device)
        sp_net = sp_net.to(self.device)
        om1_net = om1_net.to(self.device)
        om2_net = om2_net.to(self.device)

        self.eval()
        with torch.no_grad():
            emb, recons, emb_, scale_embs, scale_embs_ = self(
                omics, sp_net, om1_net, om2_net
            )
        out = {"feat": F.normalize(emb.cpu().detach(), p=2, eps=1e-12, dim=1)}
        for i in range(len(scale_embs)):
            out[f"scale{i+1}"] = F.normalize(scale_embs[i].cpu().detach(), p=2, eps=1e-12, dim=1)

        return out

    def fit_trans(
        self, omics: torch.Tensor, 
        sp_net: torch.Tensor,
        om1_net: torch.Tensor,
        om2_net: torch.Tensor,
        epochs = 600,
        lr = 0.001,
        lr_step = 10,
        gamma = 0.1,
        recon_w = None,
        scales_w = [.1, .2, .2],
        total_w = None,
        top_align_w = 0.1,
        weight_decay = 1e-4
):
        self.fit(
            omics = omics,
            sp_net = sp_net,
            om1_net = om1_net,
            om2_net = om2_net,
            epochs = epochs,
            lr = lr,
            lr_step = lr_step,
            gamma = gamma,
            recon_w = recon_w,
            scales_w = scales_w,
            total_w = total_w,
            top_align_w = top_align_w,
            weight_decay = weight_decay
        )
        return self.trans(
            omics = omics,
            sp_net = sp_net,
            om1_net = om1_net,
            om2_net = om2_net
        )

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)

