"""
Point Cloud Pretraining for Calorimeter Hits.
Uses a Masked Point Modeling (MPM) approach with an efficient Transformer encoder.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CalorimeterDataset, NeighborhoodCalorimeterDataset
import numpy as np
import awkward as ak
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import entropy
import seaborn as sns
import os
import json
from torch.utils.tensorboard import SummaryWriter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^src - 2*src*dst + dst^dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Return:
        dist: per-point square distance, [B, N, M]
    """
    B = src.shape[0]
    N = src.shape[1]
    M = dst.shape[1]
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint: int):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :3].view(B, 1, 3)
        dist = torch.sum((xyz[:, :, :3] - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_point(k: int, xyz, new_xyz):
    sqdist = square_distance(new_xyz, xyz)
    group_idx = sqdist.topk(k, dim=-1, largest=False)[1]
    return group_idx

class PatchEmbed(nn.Module):
    """
    Patch-level tokenization following Point-MAE.
    Groups points into patches using FPS and KNN, then embeds them with a PointNet.
    """
    def __init__(self, n_patches=64, k=32, in_chans=5, embed_dim=128):
        super().__init__()
        self.n_patches = n_patches
        self.k = k
        self.point_net = nn.Sequential(
            nn.Linear(in_chans, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Position embedding for patch centers
        self.center_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        """
        x: (B, N, C) - raw hits
        returns: tokens (B, G, E), centers (B, G, 3), patches (B, G, K, C), group_idx (B, G, K)
        """
        xyz = x[:, :, :3]
        
        # 1. FPS to find patch centers
        fps_idx = farthest_point_sample(xyz, self.n_patches)
        centers = index_points(xyz, fps_idx) # (B, G, 3)
        
        # 2. KNN to group points into patches
        group_idx = knn_point(self.k, xyz, centers) # (B, G, K)
        patches = index_points(x, group_idx) # (B, G, K, C)
        
        # 3. Local normalization (relative to center)
        patches_norm = patches.clone()
        patches_norm[:, :, :, :3] = patches[:, :, :, :3] - centers.unsqueeze(2)
        
        # 4. Embed patches with PointNet (Max pooling over points in patch)
        B, G, K, C = patches_norm.shape
        tokens = self.point_net(patches_norm.view(B * G, K, C)) # (B*G, K, E)
        tokens = torch.max(tokens, dim=1)[0] # (B*G, E)
        tokens = tokens.view(B, G, -1) # (B, G, E)
        
        # 5. Add spatial information via center positional embedding
        pos_embed = self.center_embed(centers)
        tokens = tokens + pos_embed
        
        return tokens, centers, patches, group_idx

def compute_all_hit_representations(model, hits, window_size=256, overlap=128):
    """
    Computes embeddings for patches in an event.
    Returns embeddings for patch centers.
    """
    N = hits.shape[0]
    device = next(model.parameters()).device
    
    # Sort hits by Z coordinate
    z_indices = torch.argsort(hits[:, 2])
    sorted_hits = hits[z_indices]
    
    all_latents = []
    all_coords = []
    
    step = max(1, window_size - overlap)
    
    model.eval()
    with torch.no_grad():
        for start in range(0, N, step):
            end = min(start + window_size, N)
            window_hits = sorted_hits[start:end]
            
            curr_size = window_hits.shape[0]
            if curr_size < 10: continue # Skip too small windows
            
            # Pad if window is smaller than window_size (for FPS consistency if needed)
            if curr_size < window_size:
                padding = window_hits[torch.randint(0, curr_size, (window_size - curr_size,))]
                input_hits = torch.cat([window_hits, padding], dim=0).unsqueeze(0)
            else:
                input_hits = window_hits.unsqueeze(0)
                
            # Patch mode: returns (B, G, E), (B, G, 3), (B, G, K, C), (B, G, K)
            tokens, centers, _, _ = model.patch_embed(input_hits)
            tokens = tokens + model.pos_embed[:, :tokens.shape[1], :]
            latent = model.encoder(tokens)
            all_latents.append(latent[0].cpu())
            all_coords.append(centers[0].cpu())
            
            if end == N: break
                
    final_latents = torch.cat(all_latents, dim=0)
    final_coords = torch.cat(all_coords, dim=0)
    return final_latents, final_coords

def compute_collapse_metrics(masked_preds, decoded, mask, targets=None):
    """
    Compute metrics to detect mode collapse in reconstruction and latents.
    masked_preds: (N_masked, K, D) - reconstructed patches
    decoded: (B, G, D_dec) - decoder output tokens
    mask: (B, G) - binary mask
    targets: (N_masked, K, D) - ground truth patches
    """
    metrics = {}
    
    # 1. Prediction Variance across different patches (Mode Collapse in output)
    # Mean of each patch: (N_masked, D)
    patch_means = masked_preds.mean(dim=1)
    # Variance across patches for each feature, then mean over features
    metrics["var_across_patches"] = patch_means.var(dim=0).mean().item()
    
    # 2. Prediction Variance within each patch (Structural collapse)
    metrics["var_within_patches"] = masked_preds.var(dim=1).mean().item()

    # 3. Average Nearest Neighbor Distance (within patch)
    # If points are on top of each other, this will be close to 0
    p_xyz = masked_preds[:, :, :3]
    dist_self = torch.cdist(p_xyz, p_xyz)
    # Get 2nd smallest distance (1st is 0 to self)
    nn_dist = dist_self.topk(2, dim=-1, largest=False)[0][:, :, 1]
    metrics["avg_nn_dist"] = nn_dist.mean().item()

    # 4. Chamfer Precision and Recall
    if targets is not None:
        _, precision, recall = compute_chamfer_loss(masked_preds, targets, return_components=True)
        metrics["chamfer_precision"] = precision.item()
        metrics["chamfer_recall"] = recall.item()
    
    # 5. Latent Cosine Similarity (Mode Collapse in latents)
    mask_bool = mask.bool()
    decoded_masked = decoded[mask_bool] # (N_masked, D_dec)
    if len(decoded_masked) > 1:
        # Sample if too many to avoid O(N^2) memory
        if len(decoded_masked) > 512:
            indices = torch.randperm(len(decoded_masked))[:512]
            decoded_masked = decoded_masked[indices]
            
        decoded_norm = nn.functional.normalize(decoded_masked, p=2, dim=1)
        cos_sim = torch.mm(decoded_norm, decoded_norm.t())
        avg_cos_sim = (cos_sim.sum() - len(cos_sim)) / (len(cos_sim) * (len(cos_sim) - 1))
        metrics["latent_cos_sim"] = avg_cos_sim.item()
    else:
        metrics["latent_cos_sim"] = 1.0
        
    return metrics

def compute_representation_metrics(all_embeddings, all_hits, epoch, output_dir):
    """
    Compute PCA, Clustering, and Correlation metrics to evaluate representation expressiveness.
    """
    metrics = {}
    
    # 1. PCA Explained Variance Entropy
    pca = PCA(n_components=min(all_embeddings.shape[0], all_embeddings.shape[1], 16))
    pca.fit(all_embeddings)
    var_ratio = pca.explained_variance_ratio_
    metrics["pca_entropy"] = entropy(var_ratio)
    metrics["pca_top_1"] = var_ratio[0]
    metrics["pca_top_3"] = np.sum(var_ratio[:3])
    
    # 2. Embedding-Input Correlation
    # We use first 3 dims (x,y,z) for correlation if it's patch centers
    corr_matrix = np.zeros((all_embeddings.shape[1], 3))
    for i in range(all_embeddings.shape[1]):
        for j in range(3):
            corr_matrix[i, j] = np.abs(np.corrcoef(all_embeddings[:, i], all_hits[:, j])[0, 1])
    
    metrics["mean_max_input_corr"] = np.mean(np.max(corr_matrix, axis=1))
    
    # 3. Clustering Metrics
    sample_size = min(len(all_embeddings), 2000)
    indices = np.random.choice(len(all_embeddings), sample_size, replace=False)
    emb_sample = all_embeddings[indices]
    hits_sample = all_hits[indices]
    
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(emb_sample)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    metrics["n_clusters"] = n_clusters
    
    if n_clusters > 1:
        valid_mask = labels != -1
        if np.sum(valid_mask) > n_clusters:
            metrics["silhouette"] = silhouette_score(emb_sample[valid_mask], labels[valid_mask])
            cluster_vars = []
            for l in unique_labels:
                if l == -1: continue
                c_hits = hits_sample[labels == l]
                if len(c_hits) > 1:
                    cluster_vars.append(np.mean(np.var(c_hits[:, :3], axis=0)))
            metrics["cluster_physical_cohesion"] = np.mean(cluster_vars) if cluster_vars else 0
        else:
            metrics["silhouette"] = 0
            metrics["cluster_physical_cohesion"] = 0
    else:
        metrics["silhouette"] = 0
        metrics["cluster_physical_cohesion"] = 0
        
    return metrics

def plot_metrics_history(history, output_dir, writer=None):
    """
    Plot the evolution of representation metrics over epochs.
    """
    epochs = [h["epoch"] for h in history]
    
    fig, axs = plt.subplots(4, 2, figsize=(15, 24))
    
    axs[0, 0].plot(epochs, [h["train_loss"] for h in history], marker='o', label='Train Loss')
    axs[0, 0].plot(epochs, [h["val_loss"] for h in history], marker='o', label='Val Loss')
    axs[0, 0].set_title("Training & Validation Loss")
    axs[0, 0].set_yscale('log')
    axs[0, 0].legend()
    
    if "pca_entropy" in history[0]:
        axs[0, 1].plot(epochs, [h["pca_entropy"] for h in history], marker='o', color='green')
        axs[0, 1].set_title("PCA Entropy (Effective Dim)")
    
    if "silhouette" in history[0]:
        axs[1, 0].plot(epochs, [h["silhouette"] for h in history], marker='o', color='red')
        axs[1, 0].set_title("Embedding Silhouette Score")
    
    if "density_corr" in history[0]:
        corrs = [h["density_corr"] for h in history if "density_corr" in h]
        axs[1, 1].plot(epochs[:len(corrs)], corrs, marker='o', color='purple')
        axs[1, 1].set_title("Density vs Loss Correlation")

    # Mode Collapse Monitoring
    if "var_across_patches" in history[0]:
        axs[2, 0].plot(epochs, [h["var_across_patches"] for h in history], marker='o', color='orange', label='Var Across Patches')
        axs[2, 0].plot(epochs, [h["var_within_patches"] for h in history], marker='s', color='brown', label='Var Within Patches')
        axs[2, 0].set_title("Prediction Variance (higher is better)")
        axs[2, 0].set_yscale('log')
        axs[2, 0].legend()

    if "latent_cos_sim" in history[0]:
        axs[2, 1].plot(epochs, [h["latent_cos_sim"] for h in history], marker='o', color='cyan')
        axs[2, 1].set_title("Latent Cosine Similarity (lower is better)")
        axs[2, 1].set_ylim(0, 1.05)

    if "avg_nn_dist" in history[0]:
        axs[3, 0].plot(epochs, [h["avg_nn_dist"] for h in history], marker='o', color='magenta')
        axs[3, 0].set_title("Avg Nearest Neighbor Dist (higher is better)")
        axs[3, 0].set_yscale('log')

    if "chamfer_precision" in history[0]:
        axs[3, 1].plot(epochs, [h["chamfer_precision"] for h in history], marker='o', label='Precision (Acc)')
        axs[3, 1].plot(epochs, [h["chamfer_recall"] for h in history], marker='s', label='Recall (Comp)')
        axs[3, 1].set_title("Chamfer Accuracy vs Completeness")
        axs[3, 1].set_yscale('log')
        axs[3, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "representation_metrics_evolution.png"))
    if writer:
        writer.add_figure("Summary/Metrics History", fig, global_step=epochs[-1])
    plt.close()

def plot_fidelity_vs_density(density_stats, epoch, output_dir, writer=None):
    """
    Plot reconstruction fidelity (loss) vs local energy density.
    density_stats: list of (energy_density, loss) tuples.
    """
    density_stats = np.array(density_stats)
    if len(density_stats) == 0: return
    
    densities = density_stats[:, 0]
    losses = density_stats[:, 1]
    
    # Filter out invalid values for log-log visualization
    # Note: densities from compute_density are already log10(sum + 1)
    mask = (densities > 0) & (losses > 0)
    densities = densities[mask]
    losses = losses[mask]
    
    if len(densities) < 10: return

    fig = plt.figure(figsize=(10, 8))
    # hexbin naturally handles density of points for better visualization than scatter
    hb = plt.hexbin(densities, np.log10(losses), gridsize=50, cmap='viridis', bins='log')
    plt.colorbar(hb, label='log10(count)')
    plt.xlabel('Local Energy Density (log10(sum E + 1))')
    plt.ylabel('Reconstruction Loss (log10(MAE))')
    plt.title(f'Reconstruction Fidelity vs Energy Density - Epoch {epoch+1}')
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, f"fidelity_vs_density_epoch_{epoch+1}.png"))
    if writer:
        writer.add_figure(f"Validation/Fidelity vs Density", fig, global_step=epoch)
    plt.close()

def visualize_embeddings(model, full_dataset, epoch, output_dir, n_events=1, writer=None):
    model.eval()
    device = next(model.parameters()).device

    for i in range(min(n_events, len(full_dataset))):
        event_data = full_dataset.get_full_event(i)
        all_hits_np = event_data["all_hits"]
        all_hits = torch.from_numpy(all_hits_np).to(device)
        
        if all_hits.shape[0] < 10: continue
        
        # Compute embeddings
        embeddings, coords = compute_all_hit_representations(model, all_hits, window_size=256)
        embeddings_np = embeddings.numpy()
        coords_np = coords.numpy()
        
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(embeddings_np)
        clusters = clustering.labels_
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        cmap = plt.get_cmap('tab20')
        def get_colors(lbls):
            cols = []
            for l in lbls:
                if l == -1: cols.append((0, 0, 0, 1))
                else: cols.append(cmap(l % 20))
            return cols

        # Plot hits colored by cluster of their corresponding embedding
        ax.scatter(coords_np[:, 0], coords_np[:, 1],
                  c=get_colors(clusters), marker='o', s=20)
            
        ax.set_title(f'Event {event_data["event_id"]} - Epoch {epoch+1}\n(DBSCAN on Patches)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.savefig(os.path.join(output_dir, f"point_cloud_epoch_{epoch+1}_ev{i}.png"))
        if writer:
            writer.add_figure(f"Embeddings/Event {i}", fig, global_step=epoch)
        plt.close()


def visualize_reconstruction(model, dataloader, epoch, output_dir, n_events=2, mask_ratio=0.5, writer=None):
    """
    Visualize ground truth vs model reconstruction for a few events.
    """
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, batch_hits in enumerate(dataloader):
            if i >= n_events: break
            
            batch_hits = batch_hits.to(device, non_blocking=True)
            
            # Compute auxiliary targets for display
            aux_targets = compute_density(batch_hits, return_all=True)
            full_targets = torch.cat([batch_hits, aux_targets], dim=-1)
            
            reconstructed, mask, _, centers, group_idx = model(batch_hits, mask_ratio=mask_ratio)
            
            # Use pre-computed indices to extract targets
            target_pts = index_points(full_targets, group_idx)
            
            # Use local coordinates for display
            target_pts_local = target_pts.clone()
            target_pts_local[:, :, :, :3] = target_pts[:, :, :, :3] - centers.unsqueeze(2)
            
            # target_pts is (B, G, K, D)
            # reconstructed is (B, G, K, D)
            hits_np = target_pts_local[0].cpu().numpy() # (G, K, D)
            recon_np = reconstructed[0].cpu().numpy() # (G, K, D)
            mask_np = mask[0].cpu().numpy().astype(bool) # (G,)
            
            fig = plt.figure(figsize=(16, 8))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            cmap = plt.get_cmap('tab20')
            for g in range(hits_np.shape[0]):
                p_hits = hits_np[g]
                p_recon = recon_np[g]
                color = cmap(g % 20)
                if mask_np[g]:
                    # Masked patches: 'x' for Ground Truth, 'x' for Reconstruction
                    ax1.scatter(p_hits[:, 0], p_hits[:, 1], color=color, s=30, marker='x')
                    ax2.scatter(p_recon[:, 0], p_recon[:, 1], color=color, s=30, marker='x')
                else:
                    # Visible patches: '.' for both
                    ax1.scatter(p_hits[:, 0], p_hits[:, 1], color=color, s=15, marker='.')
                    ax2.scatter(p_hits[:, 0], p_hits[:, 1], color=color, s=15, marker='.')
            
            ax1.set_title("Ground Truth (x=Masked, .=Visible)")
            ax1.set_xlabel('Local X')
            ax1.set_ylabel('Local Y')
            ax2.set_title("Reconstruction (x=Pred, .=Visible)")
            ax2.set_xlabel('Local X')
            ax2.set_ylabel('Local Y')
            plt.savefig(os.path.join(output_dir, f"reconstruction_epoch_{epoch+1}_ev{i}.png"))
            if writer:
                writer.add_figure(f"Reconstruction/Event {i}", fig, global_step=epoch)
            plt.close()

def compute_chamfer_loss(preds, targets, return_components=False):
    """
    Computes Chamfer Distance between predicted and target patches.
    preds: (N, K, D), targets: (N, K, D)
    """
    # Just for the first 3 coordinates for shape diversity
    p_xyz = preds[:, :, :3]
    t_xyz = targets[:, :, :3]
    
    # (N, K, K)
    dist = torch.cdist(p_xyz, t_xyz)
    
    # Precision: distance from each pred to nearest target
    min_dist_p = dist.min(dim=2)[0].mean(dim=1)
    # Recall: distance from each target to nearest pred
    min_dist_t = dist.min(dim=1)[0].mean(dim=1)
    
    if return_components:
        return (min_dist_p + min_dist_t).mean(), min_dist_p.mean(), min_dist_t.mean()
    return (min_dist_p + min_dist_t).mean()

class MaskedPointModel(nn.Module):
    def __init__(self, embed_dim=128, decoder_embed_dim=64, nhead=8, 
                 encoder_layers=6, decoder_layers=4, n_patches=64, k=32, output_dim=11):
        super().__init__()
        self.n_patches = n_patches
        self.k = k
        self.output_dim = output_dim
        
        # --- Encoder ---
        self.patch_embed = PatchEmbed(n_patches=n_patches, k=k, in_chans=5, embed_dim=embed_dim)
        # Use learned absolute pos_embed as fallback/additional signal
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, 
            activation='relu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        
        # --- Decoder ---
        # Decoder will take concatenated (latent + spatial_pos)
        # So decoder_input_dim = decoder_embed_dim + decoder_pos_dim
        self.decoder_pos_dim = 32
        self.decoder_embed_dim = decoder_embed_dim
        
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.randn(decoder_embed_dim))
        
        self.decoder_pos_mlp = nn.Sequential(
            nn.Linear(3, self.decoder_pos_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_pos_dim, self.decoder_pos_dim)
        )
        
        decoder_total_dim = decoder_embed_dim + self.decoder_pos_dim
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_total_dim, nhead=nhead // 2, batch_first=True, 
            activation='relu', norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        
        # Folding Seed for diversity within patches
        # Each point in a patch has its own learnable seed
        self.folding_seed = nn.Parameter(torch.randn(1, k, 16))
        # Spatial grid prior (3D) to provide a geometric baseline
        self.register_buffer('spatial_grid', torch.randn(1, k, 3))
        
        # Reconstruction head: Processes each point separately
        self.reconstructor = nn.Sequential(
            nn.Linear(decoder_total_dim + 16 + 3, decoder_total_dim),
            nn.LayerNorm(decoder_total_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_total_dim, decoder_total_dim),
            nn.LayerNorm(decoder_total_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_total_dim, output_dim)
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by shuffling.
        x: [N, L, D], sequence
        """
        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first len_keep indices
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(self, hits, mask_ratio=0.5):
        # 1. Embedding and Tokenization
        # tokens: (B, G, E), centers: (B, G, 3), patches: (B, G, K, C), group_idx: (B, G, K)
        x, centers, patches, group_idx = self.patch_embed(hits)

        # Add positional embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Masking (masking tokens, which represent patches)
        x_visible, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        
        # Heavy encoding
        latent_visible = self.encoder(x_visible)
        
        # 2. Decoder
        x_dec = self.decoder_proj(latent_visible)
        
        B, G_vis, D_dec = x_dec.shape
        L = centers.shape[1]
        
        # mask_tokens: (B, L - G_vis, D_dec)
        mask_tokens = self.mask_token.repeat(B, L - G_vis, 1)
        x_full = torch.cat([x_dec, mask_tokens], dim=1) 
        
        # Unshuffle to original positions
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D_dec))
        
        # Concatenate spatial positional embeddings
        spatial_pos = self.decoder_pos_mlp(centers)
        x_full = torch.cat([x_full, spatial_pos], dim=-1) # (B, L, D_dec + D_pos)
        
        # Light decoding
        decoded = self.decoder(x_full)
        
        # 3. Folding Reconstruction
        B, L, D_total = decoded.shape
        # seed: (1, K, 16) -> (B, L, K, 16)
        seed = self.folding_seed.view(1, 1, self.k, 16).expand(B, L, self.k, 16)
        # spatial_grid: (1, K, 3) -> (B, L, K, 3)
        grid = self.spatial_grid.view(1, 1, self.k, 3).expand(B, L, self.k, 3)
        
        # Add stochastic noise during training to force the MLP to use the seed
        if self.training:
            noise = torch.randn_like(seed) * 0.1
            seed = seed + noise
            grid = grid + torch.randn_like(grid) * 0.05

        # decoded: (B, L, D_total) -> (B, L, K, D_total)
        decoded_expanded = decoded.unsqueeze(2).expand(B, L, self.k, D_total)
        
        combined = torch.cat([decoded_expanded, seed, grid], dim=-1) # (B, L, K, D_total + 16 + 3)
        
        reconstructed = self.reconstructor(combined.reshape(-1, D_total + 16 + 3))
        reconstructed = reconstructed.view(B, L, self.k, self.output_dim)
            
        return reconstructed, mask, decoded, centers, group_idx

def compute_density(hits, radii=[0.01, 0.02, 0.05], return_all=False):
    """
    Computes multi-scale local density and energy-weighted features for hits.
    hits: (N, 4) or (B, N, 4)
    radii: list of radii to compute density at
    return_all: If True, returns (B, N, 2 * len(radii)) with multi-scale densities and energy sums.
                If False, returns (B, N) with the average density (backward compatible).
    """
    if len(hits.shape) == 2:
        hits = hits.unsqueeze(0)
    
    B, N, C = hits.shape
    coords = hits[:, :, :3] # (B, N, 3)
    energies = hits[:, :, 3] # (B, N)
    
    # Pairwise distances
    dist_sq = torch.cdist(coords, coords)**2
    
    density_features = []
    energy_features = []
    for r in radii:
        mask = (dist_sq < r**2).float()
        # Log-scale auxiliary features for better training stability
        d = torch.log10(mask.sum(dim=2) + 1.0)
        # Energy-weighted density (sum of neighbor energies)
        e_sum = torch.log10(torch.bmm(mask, energies.unsqueeze(-1)).squeeze(-1) + 1.0)
        
        density_features.append(d)
        energy_features.append(e_sum)
        
    if return_all:
        return torch.stack(density_features + energy_features, dim=-1)
    
    return torch.stack(density_features, dim=-1).mean(dim=-1)

def train(num_hits=256, embed_dim=16, max_events=None, epochs=1, batch_size=4, 
          output_dir="results", output_loss=None, output_checkpoint=None, use_neighborhood=True, 
          mask_ratio=0.5, lr=1e-4, n_patches=64, k_neighbors=32,
          train_dataset_name="ttbar", val_dataset_name="ggf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: num_hits={num_hits}, embed_dim={embed_dim}, patches={n_patches}x{k_neighbors}, mask_ratio={mask_ratio}, lr={lr}")
    print(f"Datasets: train={train_dataset_name}, val={val_dataset_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)
    
    # Initialize/Clear metrics log
    with open(os.path.join(output_dir, "metrics.log"), "w") as f:
        f.write(f"Training session started. Config: hits={num_hits}, embed={embed_dim}, ratio={mask_ratio}\n")
        f.write(f"Datasets: train={train_dataset_name}, val={val_dataset_name}\n\n")

    # Dataset Selection
    val_size = 500
    if use_neighborhood:
        train_dataset = NeighborhoodCalorimeterDataset(dataset_name=train_dataset_name, num_hits=num_hits, max_events=max_events, skip_events=0, verbose=False)
        val_dataset = NeighborhoodCalorimeterDataset(dataset_name=val_dataset_name, num_hits=num_hits, max_events=val_size, skip_events=0, verbose=False)
    else:
        train_dataset = CalorimeterDataset(dataset_name=train_dataset_name, num_hits=num_hits, max_events=max_events, skip_events=0, verbose=False)
        val_dataset = CalorimeterDataset(dataset_name=val_dataset_name, num_hits=num_hits, max_events=val_size, skip_events=0, verbose=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        prefetch_factor=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=4,
        pin_memory=True
    )
    
    output_dim = 5 + 6
    model = MaskedPointModel(
        embed_dim=embed_dim, 
        output_dim=output_dim, 
        n_patches=n_patches, 
        k=k_neighbors
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training Loop
    epoch_losses = []
    metrics_history = []
    for epoch in range(epochs):
        curr_mask_ratio = 0.1 + (0.75 - 0.1) * (epoch / (epochs - 1)) if epochs > 1 else 0.1
        
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train] (mask={curr_mask_ratio:.2f})")
        
        for i, batch_hits in enumerate(pbar):
            batch_hits = batch_hits.to(device, non_blocking=True)
            
            # 1. Compute auxiliary targets on GPU
            aux_targets = compute_density(batch_hits, return_all=True)
            full_targets = torch.cat([batch_hits, aux_targets], dim=-1)
            
            # 2. Forward pass returns patches/centers computed during tokenization
            reconstructed, mask, decoded, centers, group_idx = model(batch_hits, mask_ratio=curr_mask_ratio)
            
            # 3. Use pre-computed indices to extract targets (Zero Redundancy)
            target_pts = index_points(full_targets, group_idx)
            
            # --- Local Reconstruction Target ---
            # Subtract centers from target XYZ
            # target_pts: (B, G, K, D), centers: (B, G, 3)
            target_pts_local = target_pts.clone()
            target_pts_local[:, :, :, :3] = target_pts[:, :, :, :3] - centers.unsqueeze(2)

            # Extract masked targets and preds
            mask_bool = mask.bool()
            masked_targets = target_pts_local[mask_bool]
            masked_preds = reconstructed[mask_bool]
            
            # 1. Standard Recon Loss (SmoothL1)
            # Energy-weighted loss for all features
            weights = target_pts[mask_bool][:, :, 3:4] # (N_masked, K, 1)
            loss_raw = nn.SmoothL1Loss(reduction='none')(masked_preds, masked_targets)
            recon_loss = (loss_raw * weights).mean()
            
            # 2. Chamfer Loss for Coordinate Diversity (Focus on XYZ)
            # This encourages the model to place points in the right places regardless of order
            chamfer_loss = compute_chamfer_loss(masked_preds, masked_targets)
            
            # 3. Variance Loss (Stronger Diversity)
            # Variance across patches (Global diversity)
            patch_means = masked_preds.mean(dim=1)
            var_across = patch_means.var(dim=0).mean()
            target_var_across = 0.05
            var_across_loss = torch.relu(target_var_across - var_across) / target_var_across
            
            # Variance within patches (Local diversity/spread)
            # This directly penalizes points overlapping at the center
            var_within = masked_preds[:, :, :3].var(dim=1).mean()
            target_var_within = masked_targets[:, :, :3].var(dim=1).mean().detach()
            var_within_loss = nn.MSELoss()(var_within, target_var_within)
            
            # 4. Latent Diversity Loss (Encourage unique mask tokens)
            # Minimizing cosine similarity between decoder latents for masked patches
            mask_bool = mask.bool()
            decoded_masked = decoded[mask_bool] # (N_masked, D_dec)
            if len(decoded_masked) > 1:
                # Sample if too many
                if len(decoded_masked) > 256:
                    indices = torch.randperm(len(decoded_masked))[:256]
                    decoded_masked = decoded_masked[indices]
                
                dec_norm = nn.functional.normalize(decoded_masked, p=2, dim=-1)
                cos_sim = torch.mm(dec_norm, dec_norm.t()) # (N, N)
                # Average off-diagonal similarity
                latent_loss = (cos_sim.sum() - len(cos_sim)) / (len(cos_sim) * (len(cos_sim) - 1))
            else:
                latent_loss = torch.tensor(0.0, device=device)
            
            # Increased weight for Chamfer and added var_within_loss
            loss = recon_loss + 2.0 * chamfer_loss + 0.5 * var_across_loss + 1.0 * var_within_loss + 0.2 * latent_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            writer.add_scalar("Batch/Loss", loss.item(), epoch * len(train_dataloader) + i)
            writer.add_scalar("Batch/Chamfer", chamfer_loss.item(), epoch * len(train_dataloader) + i)
            
            pbar.set_postfix(
                loss=loss.item(), 
                chamfer=chamfer_loss.item(), 
                v_in=var_within.item(), 
                v_out=var_across.item()
            )

        scheduler.step()
        num_batches = i + 1
        avg_train_loss = total_train_loss / num_batches
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.6f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        density_stats = [] 
        
        val_embeddings_sample = []
        val_hits_sample = []
        
        # Accumulate collapse metrics
        collapse_metrics_list = []
        
        with torch.no_grad():
            v_batches = 0
            for batch_hits in val_dataloader:
                v_batches += 1
                batch_hits = batch_hits.to(device, non_blocking=True)
                
                # Pre-compute auxiliary targets on GPU
                aux_targets = compute_density(batch_hits, return_all=True)
                full_targets = torch.cat([batch_hits, aux_targets], dim=-1)
                
                # Forward pass returns pre-computed geometric metadata
                reconstructed, mask, latent, centers, group_idx = model(batch_hits, mask_ratio=curr_mask_ratio)
                
                # Extract target patches
                target_pts = index_points(full_targets, group_idx)
                
                # Use local coordinates for loss calculation
                target_pts_local = target_pts.clone()
                target_pts_local[:, :, :, :3] = target_pts[:, :, :, :3] - centers.unsqueeze(2)

                if v_batches <= 10:
                    val_embeddings_sample.append(latent.cpu().numpy().reshape(-1, latent.shape[-1]))
                    val_hits_sample.append(centers.cpu().numpy().reshape(-1, 3))
                    
                    # Compute collapse metrics for a subset of batches
                    mask_bool = mask.bool()
                    m_preds = reconstructed[mask_bool]
                    m_targets = target_pts_local[mask_bool]
                    c_metrics = compute_collapse_metrics(m_preds, latent, mask, targets=m_targets)
                    collapse_metrics_list.append(c_metrics)

                mask_bool = mask.bool()
                m_targets = target_pts_local[mask_bool]
                m_preds = reconstructed[mask_bool]
                
                recon_loss = nn.SmoothL1Loss()(m_preds, m_targets)
                chamfer_loss = compute_chamfer_loss(m_preds, m_targets)
                
                loss = recon_loss + 0.5 * chamfer_loss
                total_val_loss += loss.item()
                
                # Density analysis
                # Column 5-7 are count densities, 8-10 are energy densities (log10(sum E + 1))
                m_energy_densities = target_pts[:, :, :, 8:11].mean(dim=-1).mean(dim=-1)[mask_bool]
                
                # Reconstruction fidelity (MAE on first 5 hit features)
                hit_losses = torch.mean(torch.abs(m_targets[:, :, :5] - m_preds[:, :, :5]), dim=(1, 2))
                
                for d, l in zip(m_energy_densities.cpu().numpy().flatten(), hit_losses.cpu().numpy().flatten()):
                    density_stats.append((float(d), float(l)))
        
        avg_val_loss = total_val_loss / v_batches
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f}")

        
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }

        if collapse_metrics_list:
            avg_collapse = {k: np.mean([m[k] for m in collapse_metrics_list]) for k in collapse_metrics_list[0].keys()}
            epoch_stats.update(avg_collapse)

        if val_embeddings_sample:
            all_emb = np.concatenate(val_embeddings_sample, axis=0)
            all_hits_pos = np.concatenate(val_hits_sample, axis=0)
            all_hits_padded = np.zeros((all_hits_pos.shape[0], 5))
            all_hits_padded[:, :3] = all_hits_pos
            rep_metrics = compute_representation_metrics(all_emb, all_hits_padded, epoch, output_dir)
            epoch_stats.update(rep_metrics)

        if density_stats:
            density_stats = np.array(density_stats)
            valid_mask = (density_stats[:, 0] > 0) & (density_stats[:, 1] > 0)
            if np.any(valid_mask):
                log_density = density_stats[valid_mask, 0] # Already log10 from compute_density
                log_loss = np.log10(density_stats[valid_mask, 1])
                corr = np.corrcoef(log_density, log_loss)[0, 1]
                epoch_stats["density_corr"] = corr

        metrics_history.append(epoch_stats)
        
        # Save metrics to a text log for LLM analysis
        log_file_path = os.path.join(output_dir, "metrics.log")
        with open(log_file_path, "a") as f:
            f.write(f"--- Epoch {epoch+1} ---\n")
            f.write(json.dumps(epoch_stats, indent=2, cls=NumpyEncoder))
            f.write("\n\n")

        # Log all epoch stats to TensorBoard
        for k, v in epoch_stats.items():
            if isinstance(v, (int, float, np.float32, np.float64)):
                writer.add_scalar(f"Epoch/{k}", v, epoch)

        plot_metrics_history(metrics_history, output_dir, writer=writer)
        epoch_losses.append((epoch + 1, avg_train_loss, avg_val_loss, 0, 0)) 

        visualize_embeddings(model, train_dataset, epoch, output_dir, n_events=1, writer=writer)
        visualize_reconstruction(model, val_dataloader, epoch, output_dir, n_events=2, mask_ratio=curr_mask_ratio, writer=writer)
        plot_fidelity_vs_density(density_stats, epoch, output_dir, writer=writer)

    # Save model
    ckpt_name = output_checkpoint if output_checkpoint else f"checkpoint_h{num_hits}_patches.pth"
    save_path = os.path.join(output_dir, ckpt_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    writer.close()

    if output_loss:
        loss_file_path = os.path.join(output_dir, output_loss)
        with open(loss_file_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
            for e, t, v, _, _ in epoch_losses:
                f.write(f"{e},{t:.6f},{v:.6f}\n")
        print(f"Losses saved to {loss_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hits", type=int, default=2048)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_loss", type=str, default=None)
    parser.add_argument("--output_checkpoint", type=str, default=None)
    parser.add_argument("--neighborhood", type=str, choices=["True", "False"], default="True")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_patches", type=int, default=128)
    parser.add_argument("--k_neighbors", type=int, default=64)
    parser.add_argument("--train_dataset", type=str, default="ttbar")
    parser.add_argument("--val_dataset", type=str, default="ttbar")
    args = parser.parse_args()
    
    use_neighborhood = args.neighborhood == "True"
    
    train(
        num_hits=args.num_hits,
        embed_dim=args.embed_dim,
        max_events=args.max_events,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        output_loss=args.output_loss,
        output_checkpoint=args.output_checkpoint,
        use_neighborhood=use_neighborhood,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        n_patches=args.n_patches,
        k_neighbors=args.k_neighbors,
        train_dataset_name=args.train_dataset,
        val_dataset_name=args.val_dataset
    )
