
"""
Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics.
Pre-training script using hierarchical sparse 3D encoder (PTv3) and self-distillation.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

# Add project root to path to import Panda and src modules
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "Panda"))

from Panda.panda.model_base import PointTransformerV3
from Panda.panda.structure import Point
from Panda.panda.utils import collate_fn, set_seed
from src.dataset import CalorimeterDataset, NeighborhoodCalorimeterDataset

class PandaHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_prototypes=4096, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
        )
        self.prototypes = nn.Linear(bottleneck_dim, num_prototypes, bias=False)
        # Weight Normalization for prototypes
        self.prototypes = nn.utils.weight_norm(self.prototypes, name="weight", dim=0)
        self.prototypes.weight_g.data.fill_(1)
        self.prototypes.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        logits = self.prototypes(x)
        return logits

from Panda.panda.transform import Compose

class MultiViewDatasetWrapper(IterableDataset):
    def __init__(self, dataset, n_local_views=2, global_view_scale=(0.4, 1.0), local_view_scale=(0.1, 0.4), mask_ratio=0.5):
        self.dataset = dataset
        self.n_local_views = n_local_views
        self.global_view_scale = global_view_scale
        self.local_view_scale = local_view_scale
        self.mask_ratio = mask_ratio
        
        # PANDA-like normalization and sparse grid sampling
        self.transform = Compose([
            dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=1.0), # dataset already scaled by 5000
            dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True),
        ])

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def _get_view(self, hits, scale):
        n_points = hits.shape[0]
        target_size = int(np.random.uniform(*scale) * n_points)
        if target_size < 1: target_size = 1
        
        # Random anchor
        anchor_idx = np.random.randint(0, n_points)
        anchor_pos = hits[anchor_idx, :3]
        
        # Take nearest points
        dists = torch.sum((hits[:, :3] - anchor_pos)**2, dim=1)
        indices = torch.topk(dists, target_size, largest=False)[1]
        
        # Process view
        view_hits = hits[indices]
        data_dict = {
            "coord": view_hits[:, :3].numpy(),
            "energy": view_hits[:, 3:4].numpy(),
            "hit_type": view_hits[:, 4:5].numpy(),
            "index_valid_keys": ["coord", "energy", "hit_type"]
        }
        processed = self.transform(data_dict)
        return processed

    def _mask_hits(self, processed_view):
        """
        Randomly mask points for the student.
        """
        n_points = processed_view["coord"].shape[0]
        mask = torch.rand(n_points) < self.mask_ratio
        return mask

    def __iter__(self):
        for hits in self.dataset:
            # Global views
            g1 = self._get_view(hits, self.global_view_scale)
            g2 = self._get_view(hits, self.global_view_scale)
            
            # Masks
            m1 = self._mask_hits(g1)
            m2 = self._mask_hits(g2)
            
            # Local views
            locals = [self._get_view(hits, self.local_view_scale) for _ in range(self.n_local_views)]
            
            yield {
                "global_views": [g1, g2],
                "global_masks": [m1, m2],
                "local_views": locals
            }


def panda_collate_fn(batch):
    """
    Custom collate to handle multi-view batches.
    """
    global_views = [[] for _ in range(len(batch[0]["global_views"]))]
    local_views = [[] for _ in range(len(batch[0]["local_views"]))]
    global_masks = [[] for _ in range(len(batch[0]["global_masks"]))]
    
    for item in batch:
        for i, v in enumerate(item["global_views"]):
            # Features: energy and type
            feat = torch.from_numpy(np.concatenate([v["energy"], v["hit_type"]], axis=1)).float()
            coord = torch.from_numpy(v["coord"]).float()
            grid_coord = torch.from_numpy(v["grid_coord"]).long()
            global_views[i].append({
                "coord": coord, 
                "grid_coord": grid_coord, 
                "feat": feat,
                "offset": torch.tensor([coord.shape[0]], dtype=torch.long)
            })
        for i, m in enumerate(item["global_masks"]):
            global_masks[i].append(m)
        for i, v in enumerate(item["local_views"]):
            feat = torch.from_numpy(np.concatenate([v["energy"], v["hit_type"]], axis=1)).float()
            coord = torch.from_numpy(v["coord"]).float()
            grid_coord = torch.from_numpy(v["grid_coord"]).long()
            local_views[i].append({
                "coord": coord, 
                "grid_coord": grid_coord, 
                "feat": feat,
                "offset": torch.tensor([coord.shape[0]], dtype=torch.long)
            })
            
    # Use Panda's collate_fn for each view
    collated_globals = [collate_fn(gv) for gv in global_views]
    collated_locals = [collate_fn(lv) for lv in local_views]
    
    # Collate masks
    collated_masks = [torch.cat(m) for m in global_masks]
    
    return {
        "global_views": collated_globals,
        "global_masks": collated_masks,
        "local_views": collated_locals
    }

@torch.no_grad()
def sinkhorn_knopp(teacher_logits, eps=0.05, iterations=3):
    Q = torch.exp(teacher_logits / eps).t()
    B = Q.shape[1]
    K = Q.shape[0]
    # Standard Sinkhorn normalization
    Q /= Q.sum()
    for _ in range(iterations):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= K
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    Q *= B
    return Q.t()

class PandaTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Backbone config (matching PTv3 defaults but scaled for task)
        backbone_config = dict(
            in_channels=2, # (energy, type)
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(256, 256, 256, 256, 256),
            shuffle_orders=True,
            enable_flash=False, # FlashAttention might not be available
            upcast_attention=False,
            enc_mode=True # Pre-training usually uses encoder features
        )
        
        self.student_backbone = PointTransformerV3(**backbone_config, mask_token=True).to(self.device)
        self.teacher_backbone = PointTransformerV3(**backbone_config, mask_token=True).to(self.device)
        
        # Feature dim after upcasting all stages
        # PTv3 upcasts by concatenating features. Total dim = sum(enc_channels)
        total_feat_dim = sum(backbone_config["enc_channels"])
        
        self.student_head = PandaHead(total_feat_dim, 4096).to(self.device)
        self.teacher_head = PandaHead(total_feat_dim, 4096).to(self.device)
        
        # Sync teacher with student
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        # Freeze teacher
        for p in self.teacher_backbone.parameters(): p.requires_grad = False
        for p in self.teacher_head.parameters(): p.requires_grad = False
        
        self.optimizer = optim.AdamW(
            list(self.student_backbone.parameters()) + list(self.student_head.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
        
        self.writer = SummaryWriter(log_dir=args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    def update_teacher(self, m):
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    @torch.no_grad()
    def visualize_embeddings(self, epoch, n_events=1):
        """
        Visualize the learned representations for full events using DBSCAN clustering,
        hit coordinates (3D), and UMAP projection (2D).
        """
        self.student_backbone.eval()
        # Create a fresh dataset for visualization to avoid iterator conflicts
        base_ds = NeighborhoodCalorimeterDataset(num_hits=self.args.num_hits, max_events=self.args.max_events)
        
        # PANDA-like normalization and sparse grid sampling
        transform = Compose([
            dict(type="NormalizeCoord", center=[0.0, 0.0, 0.0], scale=1.0),
            dict(type="GridSample", grid_size=0.001, hash_type="fnv", mode="train", return_grid_coord=True),
        ])

        for i in range(min(n_events, len(base_ds))):
            event_data = base_ds.get_full_event(i)
            hits = event_data["all_hits"] # (N, 5)
            
            data_dict = {
                "coord": hits[:, :3],
                "energy": hits[:, 3:4],
                "hit_type": hits[:, 4:5],
                "index_valid_keys": ["coord", "energy", "hit_type"]
            }
            processed = transform(data_dict)
            
            # Prepare batch for student backbone
            feat = torch.from_numpy(np.concatenate([processed["energy"], processed["hit_type"]], axis=1)).float().to(self.device)
            coord = torch.from_numpy(processed["coord"]).float().to(self.device)
            grid_coord = torch.from_numpy(processed["grid_coord"]).long().to(self.device)
            offset = torch.tensor([coord.shape[0]], dtype=torch.long).to(self.device)
            
            batch = {
                "coord": coord,
                "grid_coord": grid_coord,
                "feat": feat,
                "offset": offset
            }
            
            # Get hierarchical features upcasted to full resolution
            point = self.student_backbone(batch, upcast=True)
            embeddings = point.feat.cpu().numpy()
            
            # DBSCAN in embedding space
            # Tune eps based on feature scale (normalized features usually work well with eps=0.5-1.5)
            clustering = DBSCAN(eps=0.7, min_samples=5).fit(embeddings)
            labels = clustering.labels_
            
            # 2D UMAP projection of embeddings
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            umap_proj = reducer.fit_transform(embeddings)
            
            # Plotting (Side-by-side: 3D Coords and 2D UMAP)
            fig = plt.figure(figsize=(18, 8))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
            
            coords = processed["coord"]
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            cmap = plt.get_cmap('tab20')
            colors = [cmap(l % 20) if l != -1 else (0, 0, 0, 0.1) for l in labels]
            
            # Constant marker size for both plots
            marker_size = 10
            
            # Plot 1: 3D Hit Coordinates
            ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                        c=colors, s=marker_size, alpha=0.6)
            ax1.set_title(f"3D Hit Coordinates (Event {event_data['event_id']})\n"
                          f"Colored by DBSCAN on Embeddings")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            
            # Plot 2: 2D UMAP Projection
            ax2.scatter(umap_proj[:, 0], umap_proj[:, 1],
                        c=colors, s=marker_size, alpha=0.6)
            ax2.set_title(f"2D UMAP Embedding Projection\n"
                          f"({n_clusters} DBSCAN clusters)")
            ax2.set_xlabel("UMAP Dimension 1")
            ax2.set_ylabel("UMAP Dimension 2")
            
            plt.suptitle(f"Event {event_data['event_id']} - Epoch {epoch+1} - Learned Representations Analysis")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = os.path.join(self.args.output_dir, f"embeddings_viz_epoch_{epoch+1}_ev{i}.png")
            plt.savefig(save_path)
            self.writer.add_figure(f"Embeddings/Event_{i}", fig, global_step=epoch)
            plt.close()
            print(f"Saved extended embedding visualization to {save_path}")

    def train_step(self, batch, m):
        self.optimizer.zero_grad()
        
        # 1. Teacher outputs for global views (unmasked)
        teacher_targets = []
        with torch.no_grad():
            for g_view in batch["global_views"]:
                for k in g_view.keys(): g_view[k] = g_view[k].to(self.device)
                # Teacher temperature: scheduled from 0.04 to 0.07 in paper
                # For simplicity, we use fixed or can be scheduled
                t_point = self.teacher_backbone(g_view, upcast=True)
                t_logits = self.teacher_head(t_point.feat)
                # Sinkhorn-Knopp centering
                target = sinkhorn_knopp(t_logits)
                teacher_targets.append(target)
                
        # 2. Student outputs for all views
        loss = 0
        n_loss_terms = 0
        
        # Student global views (masked)
        for i, g_view in enumerate(batch["global_views"]):
            # Add mask to data_dict
            g_view["mask"] = batch["global_masks"][i].to(self.device)
            for k in g_view.keys(): g_view[k] = g_view[k].to(self.device)
            
            s_point = self.student_backbone(g_view, upcast=True)
            s_logits = self.student_head(s_point.feat)
            
            # Match each teacher target (cross-view consistency)
            for j, t_target in enumerate(teacher_targets):
                temp_s = 0.1
                # Point-wise match for same view (masked student vs unmasked teacher)
                if i == j:
                    l = -torch.sum(t_target * F.log_softmax(s_logits / temp_s, dim=-1), dim=-1).mean()
                # Global-mean match for cross-view (different crops)
                else:
                    l = -torch.sum(t_target.mean(dim=0) * F.log_softmax(s_logits / temp_s, dim=-1), dim=-1).mean()
                loss += l
                n_loss_terms += 1
        
        # Student local views (unmasked)
        for lv in batch["local_views"]:
            for k in lv.keys(): lv[k] = lv[k].to(self.device)
            s_point = self.student_backbone(lv, upcast=True)
            s_logits = self.student_head(s_point.feat)
            
            for t_target in teacher_targets:
                temp_s = 0.1
                # In paper, local views are matched to global teacher views
                # We need to map local view points to global view points if we want exact matching,
                # but typically in self-distillation, we just match distributions.
                # However, PANDA matches corresponding unmasked regions.
                # For simplicity here, we match the mean distribution or random sample
                # or just use the whole target. DINO matches local views to global views.
                l = -torch.sum(t_target.mean(dim=0) * F.log_softmax(s_logits / temp_s, dim=-1), dim=-1).mean()
                loss += l
                n_loss_terms += 1
                
        loss /= n_loss_terms
        loss.backward()
        self.optimizer.step()
        
        # Update teacher
        self.update_teacher(m)
        
        return loss.item()

    def run(self):
        # Dataset
        base_ds = NeighborhoodCalorimeterDataset(num_hits=self.args.num_hits, max_events=self.args.max_events, verbose=False)
        wrapper_ds = MultiViewDatasetWrapper(base_ds, n_local_views=self.args.n_local_views)
        
        dataloader = DataLoader(
            wrapper_ds, 
            batch_size=self.args.batch_size, 
            shuffle=False,
            num_workers=4,
            collate_fn=panda_collate_fn,
            persistent_workers=True
        )
        
        n_steps = 0
        for epoch in range(self.args.epochs):
            wrapper_ds.set_epoch(epoch)
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for i, batch in enumerate(pbar):
                # EMA momentum schedule
                m = 0.996 + (1.0 - 0.996) * (n_steps / (self.args.epochs * len(base_ds) / self.args.batch_size + 1e-6))
                m = min(m, 0.9999)
                
                loss = self.train_step(batch, m)
                n_steps += 1
                
                if i % 10 == 0:
                    pbar.set_postfix(loss=f"{loss:.4f}", m=f"{m:.4f}")
                    self.writer.add_scalar("Train/Loss", loss, n_steps)
                    self.writer.add_scalar("Train/EMA_Momentum", m, n_steps)
            
            # Save checkpoint
            save_path = os.path.join(self.args.output_dir, f"panda_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "backbone_state_dict": self.student_backbone.state_dict(),
                "head_state_dict": self.student_head.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

            # Visualization
            print(f"Running embedding visualization for epoch {epoch+1}...")
            self.visualize_embeddings(epoch, n_events=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hits", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--n_local_views", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/panda_pretrain")
    parser.add_argument("--max_events", type=int, default=None)
    args = parser.parse_args()
    
    trainer = PandaTrainer(args)
    trainer.run()
