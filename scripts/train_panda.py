
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
    if pointops is not None and support.shape[0] > 0:
        try:
            return pointops.knn_query(k, support.float(), support_offset.int(), query.float(), query_offset.int())
        except Exception as e:
            print(f"pointops.knn_query failed: {e}, falling back to torch.cdist")
    
    indices = []
    distances = []
    q_start = 0
    s_start = 0
    for q_off, s_off in zip(query_offset, support_offset):
        q_end = q_off.item()
        s_end = s_off.item()
        q_batch = query[q_start:q_end]
        s_batch = support[s_start:s_end]
        if q_batch.shape[0] == 0:
            indices.append(torch.zeros((0, k), device=query.device, dtype=torch.long))
            distances.append(torch.zeros((0, k), device=query.device))
        elif s_batch.shape[0] == 0:
            # No support points, return -1 indices or similar
            indices.append(torch.full((q_batch.shape[0], k), -1, device=query.device, dtype=torch.long))
            distances.append(torch.full((q_batch.shape[0], k), float('inf'), device=query.device))
        else:
            dist = torch.cdist(q_batch, s_batch)
            actual_k = min(k, s_batch.shape[0])
            d, idx = dist.topk(actual_k, dim=1, largest=False)
            if actual_k < k:
                # Pad with -1 if not enough neighbors
                idx = F.pad(idx, (0, k - actual_k), value=-1)
                d = F.pad(d, (0, k - actual_k), value=float('inf'))
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
        iters = np.arange(max(0, total_iters - warmup_iters))
        if len(iters) > 0:
            schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        else:
            schedule = np.array([])
        self.schedule = np.concatenate((warmup_schedule, schedule))
        self.iter = 0

    def step(self):
        value = self.schedule[self.iter] if self.iter < len(self.schedule) else self.final_value
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

    def forward(self, feat, return_embed=False):
        if feat.shape[0] == 0:
            logits = torch.zeros((0, self.prototype.weight_v.shape[0]), device=feat.device)
            if return_embed:
                return logits, feat
            return logits
        feat = self.mlp(feat)
        feat = F.normalize(feat, dim=-1, p=2)
        logits = self.prototype(feat)
        if return_embed:
            return logits, feat
        return logits

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
        if feat.shape[0] == 0:
            return torch.zeros_like(feat)
        q = torch.exp(feat / temp).t()
        k, n = q.shape
        sum_q = q.sum()
        if sum_q == 0:
            return torch.zeros_like(q).t()
        q /= sum_q
        for _ in range(num_iter):
            row_sum = q.sum(dim=1, keepdim=True)
            q /= (row_sum + 1e-12)
            q /= k
            col_sum = q.sum(dim=0, keepdim=True)
            q /= (col_sum + 1e-12)
            q /= n
        q *= n
        return q.t()

    def generate_mask(self, coord, offset, mask_size, mask_ratio):
        if coord.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=coord.device)
        batch = offset2batch(offset)
        # Simplify: unique grid patches
        grid_coord = (coord // mask_size).long()
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
            if point.feat.shape[0] > 0:
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            else:
                # Handle case where point.feat is empty but inverse exists
                parent.feat = torch.cat([parent.feat, torch.zeros((parent.feat.shape[0], point.feat.shape[1]), device=parent.feat.device)], dim=-1)
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
            t_logits, t_embed = self.teacher_head(t_point.feat, return_embed=True)
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
        s_g_log_probs = F.log_softmax(s_g_logits / student_temp, dim=-1)
        mask_loss = -torch.sum(t_target * s_g_log_probs, dim=-1)
        mask_loss_reduced = mask_loss[mask].mean() if mask.any() else torch.tensor(0.0, device=mask_loss.device)

        # Local Loss (match to principal global view)
        local_loss = torch.tensor(0.0, device=mask_loss.device)
        if s_l_point.feat.shape[0] > 0 and t_target.shape[0] > 0:
            with torch.no_grad():
                principal_mask = t_point.batch % self.num_global_view == 0
                t_principal_coord = t_point.origin_coord[principal_mask]
                t_principal_target = t_target[principal_mask]
                t_principal_offset = t_point.offset[0::self.num_global_view]
                if self.num_global_view > 1:
                    t_counts = offset2bincount(t_point.offset)
                    t_principal_counts = t_counts[0::self.num_global_view]
                    t_principal_offset = torch.cumsum(t_principal_counts, dim=0)
                t_support_offset = t_principal_offset.repeat_interleave(self.num_local_view)
                
                if t_principal_coord.shape[0] > 0:
                    match_idx, _ = knn_query(1, s_l_point.origin_coord, s_l_point.offset, 
                                            t_principal_coord, t_support_offset)
                    # Filter out invalid indices (-1)
                    valid_mask = (match_idx.squeeze(-1) >= 0)
                    if valid_mask.any():
                        l_target = t_principal_target[match_idx.squeeze(-1)[valid_mask]]
                        local_loss = -torch.sum(l_target * F.log_softmax(s_l_logits[valid_mask] / student_temp, dim=-1), dim=-1).mean()
        
        # Monitoring quantities
        with torch.no_grad():
            teacher_entropy = -torch.sum(t_target * torch.log(t_target + 1e-12), dim=-1).mean() if t_target.shape[0] > 0 else torch.tensor(0.0, device=mask_loss.device)
            s_g_probs = F.softmax(s_g_logits / student_temp, dim=-1)
            student_entropy = -torch.sum(s_g_probs * torch.log(s_g_probs + 1e-12), dim=-1).mean() if s_g_probs.shape[0] > 0 else torch.tensor(0.0, device=mask_loss.device)
            # KL Divergence between teacher and student on masked points
            kl_div = F.kl_div(s_g_log_probs[mask], t_target[mask], reduction='batchmean') if mask.any() else torch.tensor(0.0, device=mask_loss.device)

        return {
            "mask_loss": mask_loss_reduced, 
            "local_loss": local_loss, 
            "loss": 0.5 * mask_loss_reduced + 0.5 * local_loss,
            "teacher_entropy": teacher_entropy,
            "student_entropy": student_entropy,
            "kl_divergence": kl_div,
            "t_target": t_target,
            "s_g_logits": s_g_logits,
            "t_embed": t_embed
        }

    @torch.no_grad()
    def track_prototype_usage(self, t_target, t_embed=None):
        assignments = t_target.argmax(dim=-1)
        num_prototypes = self.teacher_head.prototype.weight_v.shape[0]
        counts = torch.bincount(assignments, minlength=num_prototypes)
        used_prototypes = (counts > 0).sum().item()
        usage_ratio = used_prototypes / num_prototypes
        # Per-prototype assignment entropy
        prob_usage = counts.float() / (counts.sum() + 1e-12)
        usage_entropy = -torch.sum(prob_usage * torch.log(prob_usage + 1e-12)).item()
        
        stats = {
            "used_prototypes": used_prototypes,
            "usage_ratio": usage_ratio,
            "usage_entropy": usage_entropy
        }

        if t_embed is not None:
            # Cluster cohesion: average cosine similarity between points and their assigned prototype
            prototypes = self.teacher_head.prototype.weight
            assigned_prototypes = prototypes[assignments]
            cohesion = torch.sum(t_embed * assigned_prototypes, dim=-1).mean().item()
            stats["cohesion"] = cohesion
            
        return stats

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
                            num_global_view=args.n_global_views,
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

    @torch.no_grad()
    def visualize_batch_views(self, batch, n_steps):
        """Visualize how local views are cropped from the global views."""
        self.model.eval()
        # Extract first event's data from the collated batch
        n_global = self.args.n_global_views
        n_local = self.args.n_local_views
        
        g_offset = batch["global"]["offset"]
        l_offset = batch["local"]["offset"]
        
        # Get hits for the first event
        g_end = g_offset[n_global - 1].item()
        l_end = l_offset[n_local - 1].item()
        
        g_coords = batch["global"]["origin_coord"][:g_end].cpu().numpy()
        l_coords = batch["local"]["origin_coord"][:l_end].cpu().numpy()
        
        # Split into individual views
        g_views_coords = []
        start = 0
        for i in range(n_global):
            end = g_offset[i].item()
            g_views_coords.append(batch["global"]["origin_coord"][start:end].cpu().numpy())
            start = end
            
        l_views_coords = []
        start = 0
        for i in range(n_local):
            end = l_offset[i].item()
            l_views_coords.append(batch["local"]["origin_coord"][start:end].cpu().numpy())
            start = end

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all global hits in light grey for context
        ax.scatter(g_coords[:, 0], g_coords[:, 1], g_coords[:, 2], 
                   c='lightgrey', s=1, alpha=0.3, label='Global Context')
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_local))
        
        # Helper to draw wireframe cuboid
        def draw_cuboid(ax, points, color, label=None):
            if len(points) == 0: return
            min_p = points.min(axis=0)
            max_p = points.max(axis=0)
            x = [min_p[0], max_p[0]]
            y = [min_p[1], max_p[1]]
            z = [min_p[2], max_p[2]]
            import itertools
            vertices = list(itertools.product(x, y, z))
            edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
            for i, (start, end) in enumerate(edges):
                ax.plot3D(*zip(vertices[start], vertices[end]), color=color, 
                          linewidth=1.5, alpha=0.8, label=label if i == 0 else "")

        # Plot local views and their bounding boxes
        for i, l_view in enumerate(l_views_coords):
            if len(l_view) == 0: continue
            color = colors[i]
            ax.scatter(l_view[:, 0], l_view[:, 1], l_view[:, 2], 
                       color=color, s=5, alpha=0.6)
            draw_cuboid(ax, l_view, color, label=f"Local View {i}")
            
            # Add marker label at the center of the local view
            center = l_view.mean(axis=0)
            ax.text(center[0], center[1], center[2], f"L{i}", color="black", 
                    fontsize=10, fontweight="bold", bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))

        # Add marker labels for global views
        for i, g_view in enumerate(g_views_coords):
            if len(g_view) == 0: continue
            center = g_view.mean(axis=0)
            ax.text(center[0], center[1], center[2], f"G{i}", color="darkred", 
                    fontsize=10, fontweight="bold", alpha=0.8)

        ax.set_title(f"Batch View Visualization - Step {n_steps}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
        
        save_path = os.path.join(self.args.output_dir, f"batch_viz_step_{n_steps}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # Save individual global-local pairs
        step_dir = os.path.join(self.args.output_dir, f"step_{n_steps}_pairs")
        os.makedirs(step_dir, exist_ok=True)
        g_principal = g_views_coords[0]
        
        for i, l_view in enumerate(l_views_coords):
            if len(l_view) == 0: continue
            fig_p = plt.figure(figsize=(8, 8))
            ax_p = fig_p.add_subplot(111, projection='3d')
            
            # Plot principal global view in light grey
            ax_p.scatter(g_principal[:, 0], g_principal[:, 1], g_principal[:, 2], 
                         c='lightgrey', s=2, alpha=0.3, label='Principal Global (G0)')
            
            # Plot the specific local view
            color = colors[i]
            ax_p.scatter(l_view[:, 0], l_view[:, 1], l_view[:, 2], 
                         color=color, s=10, alpha=0.8, label=f'Local View {i}')
            
            draw_cuboid(ax_p, l_view, color)
            
            ax_p.set_title(f"Global-Local Pair: G0 - L{i} (Step {n_steps})")
            ax_p.set_xlabel("X")
            ax_p.set_ylabel("Y")
            ax_p.set_zlabel("Z")
            ax_p.legend()
            
            plt.savefig(os.path.join(step_dir, f"pair_g0_l{i}.png"), bbox_inches='tight')
            plt.close()

    def run(self):
        base_ds = NeighborhoodCalorimeterDataset(num_hits=self.args.num_hits, max_events=self.args.max_events, verbose=False)
        wrapper_ds = MultiViewDatasetWrapper(base_ds, n_global_views=self.args.n_global_views, n_local_views=self.args.n_local_views)
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
                
                if n_steps % self.args.viz_batch_freq == 0:
                    self.visualize_batch_views(batch, n_steps)

                out = self.model(data_dict, mask_size=m_size, mask_ratio=m_ratio, teacher_temp=t_temp)
                out["loss"].backward()
                self.optimizer.step()
                self.update_teacher(m)
                
                n_steps += 1
                if n_steps % 10 == 0:
                    usage_stats = self.model.track_prototype_usage(out["t_target"], out.get("t_embed"))
                    
                    pbar.set_postfix(loss=f"{out['loss'].item():.4f}", 
                                     t_ent=f"{out['teacher_entropy'].item():.2f}",
                                     used=f"{usage_stats['used_prototypes']}",
                                     coh=f"{usage_stats.get('cohesion', 0):.3f}")
                    
                    self.writer.add_scalar("Train/Loss", out["loss"].item(), n_steps)
                    self.writer.add_scalar("Train/Mask_Loss", out["mask_loss"].item(), n_steps)
                    self.writer.add_scalar("Train/Local_Loss", out["local_loss"].item(), n_steps)
                    self.writer.add_scalar("Train/Teacher_Entropy", out["teacher_entropy"].item(), n_steps)
                    self.writer.add_scalar("Train/Student_Entropy", out["student_entropy"].item(), n_steps)
                    self.writer.add_scalar("Train/KL_Divergence", out["kl_divergence"].item(), n_steps)
                    self.writer.add_scalar("Params/Mask_Ratio", m_ratio, n_steps)
                    self.writer.add_scalar("Params/Mask_Size", m_size, n_steps)
                    self.writer.add_scalar("Params/Teacher_Temp", t_temp, n_steps)
                    self.writer.add_scalar("Params/Momentum", m, n_steps)
                    self.writer.add_scalar("Prototypes/Used_Count", usage_stats["used_prototypes"], n_steps)
                    self.writer.add_scalar("Prototypes/Usage_Ratio", usage_stats["usage_ratio"], n_steps)
                    self.writer.add_scalar("Prototypes/Usage_Entropy", usage_stats["usage_entropy"], n_steps)
                    if "cohesion" in usage_stats:
                        self.writer.add_scalar("Prototypes/Cohesion", usage_stats["cohesion"], n_steps)
            
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
    parser.add_argument("--n_global_views", type=int, default=2)
    parser.add_argument("--n_local_views", type=int, default=6)
    parser.add_argument("--viz_batch_freq", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/panda_pretrain_sonata")
    parser.add_argument("--max_events", type=int, default=None)
    args = parser.parse_args()
    trainer = PandaTrainer(args)
    trainer.run()
