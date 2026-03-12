
"""
Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics.
Refined pre-training script using hierarchical sparse 3D encoder (PTv3) and Sonata (self-distillation).
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
import umap
from timm.models.layers import trunc_normal_
from functools import partial
from itertools import chain

# Add project root to path to import Panda and src modules
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "Panda"))

from Panda.panda.model_base import PointTransformerV3
from Panda.panda.structure import Point
from Panda.panda.module import PointModule, PointSequential
from Panda.panda.utils import offset2bincount, bincount2offset, offset2batch, batch2offset, set_seed
from src.dataset import CalorimeterDataset, NeighborhoodCalorimeterDataset
from Panda.panda.transform import Compose

# --- KNN Fallback ---
try:
    import pointops
except ImportError:
    pointops = None

def knn_query(k, query, query_offset, support, support_offset):
    """Fallback KNN using torch.cdist when pointops is not available."""
    if pointops is not None:
        return pointops.knn_query(k, support.float(), support_offset.int(), query.float(), query_offset.int())
    
    indices = []
    distances = []
    q_start = 0
    s_start = 0
    for q_off, s_off in zip(query_offset, support_offset):
        q_end = q_off.item()
        s_end = s_off.item()
        q_batch = query[q_start:q_end]
        s_batch = support[s_start:s_end]
        if q_batch.shape[0] == 0 or s_batch.shape[0] == 0:
            indices.append(torch.zeros((q_batch.shape[0], k), device=query.device, dtype=torch.long))
            distances.append(torch.zeros((q_batch.shape[0], k), device=query.device))
        else:
            dist = torch.cdist(q_batch, s_batch)
            d, idx = dist.topk(min(k, s_batch.shape[0]), dim=1, largest=False)
            indices.append(idx + s_start)
            distances.append(d)
        q_start, s_start = q_end, s_end
    return torch.cat(indices, dim=0), torch.cat(distances, dim=0)

# --- Schedulers ---
class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, start_value=0, warmup_iters=0):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        self.iter = 0

    def step(self):
        value = self.schedule[self.iter] if self.iter < self.total_iters else self.final_value
        self.iter += 1
        return value

# --- Sonata Model Components ---
class OnlineCluster(nn.Module):
    def __init__(self, in_channels, hidden_channels=4096, embed_channels=256, num_prototypes=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_channels),
        )
        self.prototype = nn.utils.weight_norm(nn.Linear(embed_channels, num_prototypes, bias=False))
        self.prototype.weight_g.data.fill_(1)
        self.prototype.weight_g.requires_grad = False
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        feat = self.mlp(feat)
        feat = F.normalize(feat, dim=-1, p=2)
        return self.prototype(feat)

class Sonata(PointModule):
    def __init__(self, backbone_config, head_in_channels, head_num_prototypes=4096, 
                 num_global_view=2, num_local_view=6, up_cast_level=4):
        super().__init__()
        self.num_global_view = num_global_view
        self.num_local_view = num_local_view
        self.up_cast_level = up_cast_level
        
        self.student_backbone = PointTransformerV3(**backbone_config)
        self.teacher_backbone = PointTransformerV3(**backbone_config)
        
        # Sonata uses two types of heads in the original code, but often they are identical
        # Here we use mask_head for global views and unmask_head for local views
        self.student_head = OnlineCluster(in_channels=head_in_channels, num_prototypes=head_num_prototypes)
        self.teacher_head = OnlineCluster(in_channels=head_in_channels, num_prototypes=head_num_prototypes)
        
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_backbone.parameters(): p.requires_grad = False
        for p in self.teacher_head.parameters(): p.requires_grad = False

    @staticmethod
    def sinkhorn_knopp(feat, temp, num_iter=3):
        q = torch.exp(feat / temp).t()
        k, n = q.shape
        sum_q = q.sum()
        q /= sum_q
        for _ in range(num_iter):
            q /= q.sum(dim=1, keepdim=True)
            q /= k
            q /= q.sum(dim=0, keepdim=True)
            q /= n
        q *= n
        return q.t()

    def generate_mask(self, coord, offset, mask_size, mask_ratio):
        batch = offset2batch(offset)
        # Simplify: unique grid patches
        grid_coord = (coord // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster = torch.unique(grid_coord, dim=0, return_inverse=True)
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * mask_ratio)
        mask_patch_index = torch.randperm(patch_num, device=coord.device)[:mask_patch_num]
        return torch.isin(point_cluster, mask_patch_index)

    def up_cast(self, point):
        for _ in range(self.up_cast_level):
            if "pooling_parent" not in point.keys(): break
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def forward(self, data_dict, mask_size=0.05, mask_ratio=0.7, teacher_temp=0.07, student_temp=0.1):
        # 1. Prepare Points
        global_point = Point(feat=data_dict["global_feat"], coord=data_dict["global_coord"],
                             origin_coord=data_dict["global_origin_coord"], offset=data_dict["global_offset"],
                             grid_size=data_dict["grid_size"][0])
        
        local_point = Point(feat=data_dict["local_feat"], coord=data_dict["local_coord"],
                            origin_coord=data_dict["local_origin_coord"], offset=data_dict["local_offset"],
                            grid_size=data_dict["grid_size"][0])

        # 2. Teacher Forward (Global)
        with torch.no_grad():
            t_point = self.teacher_backbone(global_point, upcast=False)
            t_point = self.up_cast(t_point)
            t_logits = self.teacher_head(t_point.feat)
            t_target = self.sinkhorn_knopp(t_logits, teacher_temp)

        # 3. Student Forward (Global with Masking)
        mask = self.generate_mask(global_point.coord, global_point.offset, mask_size, mask_ratio)
        m_global_point = Point(feat=data_dict["global_feat"], coord=data_dict["global_coord"], mask=mask,
                               offset=data_dict["global_offset"], grid_size=data_dict["grid_size"][0])
        s_g_point = self.student_backbone(m_global_point, upcast=False)
        s_g_point = self.up_cast(s_g_point)
        s_g_logits = self.student_head(s_g_point.feat)

        # 4. Student Forward (Local)
        s_l_point = self.student_backbone(local_point, upcast=False)
        s_l_point = self.up_cast(s_l_point)
        s_l_logits = self.student_head(s_l_point.feat)

        # 5. Loss Calculation (Multi-view consistency)
        # Global Masked Loss
        mask_loss = -torch.sum(t_target * F.log_softmax(s_g_logits / student_temp, dim=-1), dim=-1)
        mask_loss = mask_loss[mask].mean() if mask.any() else torch.tensor(0.0, device=mask_loss.device)

        # Local Loss (match to principal global view)
        with torch.no_grad():
            # In Sonata, local views are matched against the first (principal) global view
            # principal_view_mask selects the first global view for each sample in the batch
            # For batch_size=1, this is just all global points of the first view
            principal_mask = t_point.batch % self.num_global_view == 0
            t_principal_coord = t_point.origin_coord[principal_mask]
            t_principal_target = t_target[principal_mask]
            
            # We need to calculate the offset for the principal global views
            # For each sample, we only take the first view
            t_principal_offset = t_point.offset[0::self.num_global_view]
            if self.num_global_view > 1:
                # Need to adjust offsets if they were cumulative
                t_counts = offset2bincount(t_point.offset)
                t_principal_counts = t_counts[0::self.num_global_view]
                t_principal_offset = torch.cumsum(t_principal_counts, dim=0)

            # Match each local view to its corresponding principal global view
            # We repeat the principal offset for each of its local views
            t_support_offset = t_principal_offset.repeat_interleave(self.num_local_view)
            
            match_idx, _ = knn_query(1, s_l_point.origin_coord, s_l_point.offset, 
                                     t_principal_coord, t_support_offset)
            l_target = t_principal_target[match_idx.squeeze(-1)]
        
        local_loss = -torch.sum(l_target * F.log_softmax(s_l_logits / student_temp, dim=-1), dim=-1).mean()

        return {"mask_loss": mask_loss, "local_loss": local_loss, "loss": 0.5 * mask_loss + 0.5 * local_loss}

# --- Dataset and Collate ---
class MultiViewDatasetWrapper(IterableDataset):
    def __init__(self, dataset, n_global_views=2, n_local_views=6, 
                 global_view_scale=(0.4, 1.0), local_view_scale=(0.1, 0.4)):
        self.dataset = dataset
        self.n_global_views = n_global_views
        self.n_local_views = n_local_views
        self.global_view_scale = global_view_scale
        self.local_view_scale = local_view_scale
        self.transform = Compose([
            dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=1.0),
            dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True),
        ])

    def __len__(self): return len(self.dataset)
    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"): self.dataset.set_epoch(epoch)

    def _get_view(self, hits, scale):
        n_points = hits.shape[0]
        target_size = int(np.random.uniform(*scale) * n_points)
        if target_size < 1: target_size = 1
        anchor_idx = np.random.randint(0, n_points)
        anchor_pos = hits[anchor_idx, :3]
        dists = torch.sum((hits[:, :3] - anchor_pos)**2, dim=1)
        indices = torch.topk(dists, target_size, largest=False)[1]
        view_hits = hits[indices]
        data_dict = {
            "coord": view_hits[:, :3].numpy(),
            "origin_coord": view_hits[:, :3].numpy(),
            "energy": view_hits[:, 3:4].numpy(),
            "hit_type": view_hits[:, 4:5].numpy(),
            "index_valid_keys": ["coord", "origin_coord", "energy", "hit_type"]
        }
        return self.transform(data_dict)

    def __iter__(self):
        for hits in self.dataset:
            yield {
                "global_views": [self._get_view(hits, self.global_view_scale) for _ in range(self.n_global_views)],
                "local_views": [self._get_view(hits, self.local_view_scale) for _ in range(self.n_local_views)]
            }

def panda_collate_fn(batch):
    def process_view(view):
        # Features: (x, y, z, energy, type)
        feat = np.concatenate([view["coord"], view["energy"], view["hit_type"]], axis=1)
        return {
            "coord": torch.from_numpy(view["coord"]).float(),
            "origin_coord": torch.from_numpy(view["origin_coord"]).float(),
            "grid_coord": torch.from_numpy(view["grid_coord"]).long(),
            "feat": torch.from_numpy(feat).float(),
            "offset": torch.tensor([view["coord"].shape[0]], dtype=torch.long)
        }

    global_views_flat = []
    local_views_flat = []
    # Interleave for Sonata: [v1_1, v2_1, v1_2, v2_2, ...]
    for item in batch:
        for v in item["global_views"]: global_views_flat.append(process_view(v))
        for v in item["local_views"]: local_views_flat.append(process_view(v))

    def collate_flat(views):
        res = {}
        for k in ["coord", "origin_coord", "grid_coord", "feat"]:
            res[k] = torch.cat([v[k] for v in views], dim=0)
        res["offset"] = torch.cumsum(torch.cat([v["offset"] for v in views]), dim=0)
        return res

    return {
        "global": collate_flat(global_views_flat),
        "local": collate_flat(local_views_flat),
        "grid_size": torch.tensor([0.001])
    }

# --- Trainer ---
class PandaTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        backbone_config = dict(
            in_channels=5, # (x, y, z, energy, type)
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            shuffle_orders=True,
            enable_flash=False,
            enc_mode=True,
            mask_token=True
        )
        
        # head_in_channels = 32 + 64 + 128 + 256 + 512 = 992
        self.model = Sonata(backbone_config, head_in_channels=992, 
                            head_num_prototypes=args.num_prototypes, 
                            num_local_view=args.n_local_views).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.writer = SummaryWriter(log_dir=args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    def update_teacher(self, m):
        with torch.no_grad():
            for ps, pt in zip(self.model.student_backbone.parameters(), self.model.teacher_backbone.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1 - m)
            for ps, pt in zip(self.model.student_head.parameters(), self.model.teacher_head.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    @torch.no_grad()
    def visualize_embeddings(self, epoch):
        self.model.eval()
        base_ds = NeighborhoodCalorimeterDataset(num_hits=self.args.num_hits, max_events=self.args.max_events)
        transform = Compose([
            dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=1.0),
            dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True),
        ])
        event_data = base_ds.get_full_event(0)
        hits = event_data["all_hits"]
        processed = transform({"coord": hits[:, :3], "energy": hits[:, 3:4], "hit_type": hits[:, 4:5], 
                               "index_valid_keys": ["coord", "energy", "hit_type"]})
        feat = np.concatenate([processed["coord"], processed["energy"], processed["hit_type"]], axis=1)
        batch = {"coord": torch.from_numpy(processed["coord"]).float().to(self.device),
                 "grid_coord": torch.from_numpy(processed["grid_coord"]).long().to(self.device),
                 "feat": torch.from_numpy(feat).float().to(self.device),
                 "offset": torch.tensor([processed["coord"].shape[0]], dtype=torch.long).to(self.device)}
        
        point = self.model.student_backbone(batch, upcast=False)
        point = self.model.up_cast(point)
        embeddings = point.feat.cpu().numpy()
        
        labels = DBSCAN(eps=0.7, min_samples=5).fit(embeddings).labels_
        umap_proj = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(embeddings)
        
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(processed["coord"][:, 0], processed["coord"][:, 1], processed["coord"][:, 2], 
                    c=labels, s=2, cmap='tab20')
        ax2 = fig.add_subplot(122)
        ax2.scatter(umap_proj[:, 0], umap_proj[:, 1], c=labels, s=2, cmap='tab20')
        plt.savefig(os.path.join(self.args.output_dir, f"viz_epoch_{epoch+1}.png"))
        plt.close()

    def run(self):
        base_ds = NeighborhoodCalorimeterDataset(num_hits=self.args.num_hits, max_events=self.args.max_events, verbose=False)
        wrapper_ds = MultiViewDatasetWrapper(base_ds, n_local_views=self.args.n_local_views)
        dataloader = DataLoader(wrapper_ds, batch_size=self.args.batch_size, shuffle=False, 
                                num_workers=4, collate_fn=panda_collate_fn, persistent_workers=True)
        
        total_steps = self.args.epochs * len(dataloader)
        mask_size_sched = CosineScheduler(0.01, 0.05, total_steps, start_value=0.01, warmup_iters=int(0.05*total_steps))
        mask_ratio_sched = CosineScheduler(0.5, 0.9, total_steps, start_value=0.5, warmup_iters=int(0.05*total_steps))
        temp_t_sched = CosineScheduler(0.04, 0.07, total_steps, start_value=0.04, warmup_iters=int(0.05*total_steps))
        momentum_sched = CosineScheduler(0.994, 1.0, total_steps)

        n_steps = 0
        for epoch in range(self.args.epochs):
            wrapper_ds.set_epoch(epoch)
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for batch in pbar:
                m = momentum_sched.step()
                m_size = mask_size_sched.step()
                m_ratio = mask_ratio_sched.step()
                t_temp = temp_t_sched.step()
                
                self.optimizer.zero_grad()
                data_dict = {
                    "global_feat": batch["global"]["feat"].to(self.device),
                    "global_coord": batch["global"]["coord"].to(self.device),
                    "global_origin_coord": batch["global"]["origin_coord"].to(self.device),
                    "global_offset": batch["global"]["offset"].to(self.device),
                    "local_feat": batch["local"]["feat"].to(self.device),
                    "local_coord": batch["local"]["coord"].to(self.device),
                    "local_origin_coord": batch["local"]["origin_coord"].to(self.device),
                    "local_offset": batch["local"]["offset"].to(self.device),
                    "grid_size": batch["grid_size"].to(self.device)
                }
                
                out = self.model(data_dict, mask_size=m_size, mask_ratio=m_ratio, teacher_temp=t_temp)
                out["loss"].backward()
                self.optimizer.step()
                self.update_teacher(m)
                
                n_steps += 1
                if n_steps % 10 == 0:
                    pbar.set_postfix(loss=f"{out['loss'].item():.4f}", m=f"{m:.4f}")
                    self.writer.add_scalar("Train/Loss", out["loss"].item(), n_steps)
                    self.writer.add_scalar("Params/Mask_Ratio", m_ratio, n_steps)
            
            self.visualize_embeddings(epoch)
            torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, f"checkpoint_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hits", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_prototypes", type=int, default=4096)
    parser.add_argument("--n_local_views", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="results/panda_pretrain_sonata")
    parser.add_argument("--max_events", type=int, default=None)
    args = parser.parse_args()
    trainer = PandaTrainer(args)
    trainer.run()
