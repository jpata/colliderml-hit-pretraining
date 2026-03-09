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
import os

def compute_all_hit_representations(model, hits, window_size=256, overlap=128):
    """
    Computes embeddings for all hits in an event using overlapping windows.
    hits: (N, 4) tensor
    """
    N = hits.shape[0]
    embed_dim = model.mask_token.shape[0]
    device = next(model.parameters()).device
    
    # Sort hits by Z coordinate to give some spatial order to the sliding window
    z_indices = torch.argsort(hits[:, 2])
    sorted_hits = hits[z_indices]
    
    # Storage for accumulated embeddings and counts
    accumulated_embeddings = torch.zeros((N, embed_dim), device=device)
    counts = torch.zeros((N, 1), device=device)
    
    step = max(1, window_size - overlap)
    
    model.eval()
    with torch.no_grad():
        for start in range(0, N, step):
            end = min(start + window_size, N)
            window_hits = sorted_hits[start:end]
            
            # Pad if window is smaller than window_size (for model compatibility if needed)
            curr_size = window_hits.shape[0]
            if curr_size < window_size:
                padding = torch.zeros((window_size - curr_size, 4), device=device)
                input_hits = torch.cat([window_hits, padding], dim=0).unsqueeze(0)
            else:
                input_hits = window_hits.unsqueeze(0)
                
            # Bypassing forward() masking logic
            hit_embeddings = model.hit_encoder(input_hits)
            latent = model.transformer(hit_embeddings)
            
            valid_latent = latent[0, :curr_size]
            accumulated_embeddings[start:end] += valid_latent
            counts[start:end] += 1
            if end == N: break
                
    final_sorted_embeddings = accumulated_embeddings / counts
    final_embeddings = torch.zeros_like(final_sorted_embeddings)
    final_embeddings[z_indices] = final_sorted_embeddings
    return final_embeddings

def visualize_embeddings_3d(model, full_dataset, epoch, output_dir, n_events=2):
    """
    Visualize x, y, z point cloud of calo and tracker hits, 
    colored by their clustering in the embedding space.
    """
    model.eval()
    device = next(model.parameters()).device
    print(f"Generating 3D point cloud visualizations for {n_events} events...")

    for i in range(min(n_events, len(full_dataset))):
        event_data = full_dataset.get_full_event(i)
        calo_hits = torch.from_numpy(event_data["calo_hits"]).to(device)
        tracker_hits = torch.from_numpy(event_data["tracker_hits"]).to(device)
        
        all_hits = torch.cat([calo_hits, tracker_hits], dim=0)
        if all_hits.shape[0] == 0: continue
        
        # Compute embeddings for all hits in the event
        embeddings = compute_all_hit_representations(model, all_hits, window_size=256)
        embeddings_np = embeddings.cpu().numpy()
        
        # DBSCAN clustering: eps (radius) and min_samples
        # These may need tuning based on the embedding space scale
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(embeddings_np)
        clusters = clustering.labels_
        
        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color normalization for clusters (using a discrete map)
        # Handle noise (-1) by making it a separate color (e.g., black)
        cmap = plt.get_cmap('tab20')
        unique_labels = np.unique(clusters)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        n_calo = calo_hits.shape[0]
        
        def get_colors(lbls):
            # Map -1 to black, others to discrete colors
            cols = []
            for l in lbls:
                if l == -1:
                    cols.append('black')
                else:
                    cols.append(cmap(l % 20))
            return cols

        # Calo hits (circles)
        if n_calo > 0:
            ax.scatter(
                event_data["calo_hits"][:, 0], 
                event_data["calo_hits"][:, 1], 
                event_data["calo_hits"][:, 2],
                c=get_colors(clusters[:n_calo]), marker='o', s=15, alpha=0.7, label='Calo Hits'
            )
            
        # Tracker hits (triangles)
        if tracker_hits.shape[0] > 0:
            ax.scatter(
                event_data["tracker_hits"][:, 0], 
                event_data["tracker_hits"][:, 1], 
                event_data["tracker_hits"][:, 2],
                c=get_colors(clusters[n_calo:]), marker='^', s=8, alpha=0.5, label='Tracker Hits'
            )
            
        ax.set_xlabel('X (normalized)')
        ax.set_ylabel('Y (normalized)')
        ax.set_zlabel('Z (normalized)')
        ax.set_title(f'Event {event_data["event_id"]} - Epoch {epoch+1}\n(DBSCAN: {n_clusters_found} clusters found, black = noise)')
        ax.legend()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"point_cloud_epoch_{epoch+1}_ev{i}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

class PointNetEncoder(nn.Module):
    """Simple MLP-based encoder per hit."""
    def __init__(self, input_dim=4, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.mlp(x)

class MaskedPointModel(nn.Module):
    def __init__(self, embed_dim=128, nhead=8, num_layers=4):
        super().__init__()
        self.hit_encoder = PointNetEncoder(input_dim=4, embed_dim=embed_dim)
        
        # Learned mask token
        self.mask_token = nn.Parameter(torch.randn(embed_dim))
        
        # Efficient Transformer encoder using scaled_dot_product_attention
        # We use the standard TransformerEncoderLayer but ensure it uses the fast path
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, activation='relu',
            norm_first=True # Better for deep transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reconstruction head (predicts x, y, z, e)
        self.reconstructor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)
        )

    def forward(self, hits, mask_ratio=0.5):
        # hits: (B, N, 4)
        B, N, C = hits.shape
        device = hits.device
        
        # Embed each hit
        hit_embeddings = self.hit_encoder(hits) # (B, N, embed_dim)
        
        # Random masking
        num_masked = int(N * mask_ratio)
        mask_indices = torch.rand(B, N, device=device).argsort(dim=1)[:, :num_masked]
        
        # Prepare tokens for transformer
        input_embeddings = hit_embeddings.clone()
        for b in range(B):
            input_embeddings[b, mask_indices[b]] = self.mask_token
            
        # Transformer processes all tokens (automatically uses FlashAttention if available)
        latent = self.transformer(input_embeddings) # (B, N, embed_dim)
        
        # Reconstruct only the masked ones
        reconstructed = self.reconstructor(latent) # (B, N, 4)
        
        return reconstructed, mask_indices, latent

def compute_density(hits, radius=0.05):
    """
    Computes local density for hits.
    hits: (N, 4) or (B, N, 4)
    """
    if len(hits.shape) == 2:
        hits = hits.unsqueeze(0)
    
    B, N, C = hits.shape
    coords = hits[:, :, :3] # (B, N, 3)
    
    # Pairwise distances
    dist_sq = torch.cdist(coords, coords)**2
    density = (dist_sq < radius**2).float().sum(dim=2) # (B, N)
    return density

def train(num_hits=256, embed_dim=16, max_events=None, epochs=1, batch_size=4, 
          output_dir="results", output_loss=None, use_neighborhood=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: num_hits={num_hits}, embed_dim={embed_dim}, max_events={max_events}, neighborhood={use_neighborhood}")
    
    os.makedirs(output_dir, exist_ok=True)

    lr = 1e-3
    
    # Dataset Selection
    if use_neighborhood:
        full_dataset = NeighborhoodCalorimeterDataset(num_hits=num_hits, max_events=max_events)
    else:
        full_dataset = CalorimeterDataset(num_hits=num_hits, max_events=max_events)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MaskedPointModel(embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none') # Need per-hit loss for density analysis

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for i, batch_hits in enumerate(pbar):
            batch_hits = batch_hits.to(device)
            reconstructed, mask_indices, latent = model(batch_hits, mask_ratio=0.5)
            
            # Extract masked targets and preds
            all_masked_targets = []
            all_masked_preds = []
            for b in range(batch_hits.shape[0]):
                all_masked_targets.append(batch_hits[b, mask_indices[b]])
                all_masked_preds.append(reconstructed[b, mask_indices[b]])
            
            loss = nn.MSELoss()(torch.cat(all_masked_preds), torch.cat(all_masked_targets))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.6f}")

        # Validation phase with Density Analysis
        model.eval()
        total_val_loss = 0
        density_stats = [] # Store (density, loss) tuples
        
        with torch.no_grad():
            for batch_hits in val_dataloader:
                batch_hits = batch_hits.to(device)
                reconstructed, mask_indices, latent = model(batch_hits, mask_ratio=0.5)
                
                densities = compute_density(batch_hits) # (B, N)
                
                for b in range(batch_hits.shape[0]):
                    masked_targets = batch_hits[b, mask_indices[b]]
                    masked_preds = reconstructed[b, mask_indices[b]]
                    masked_densities = densities[b, mask_indices[b]]
                    
                    hit_losses = torch.mean((masked_targets - masked_preds)**2, dim=1)
                    
                    for d, l in zip(masked_densities.cpu().numpy(), hit_losses.cpu().numpy()):
                        density_stats.append((d, l))
                
                # Global val loss
                total_val_loss += nn.MSELoss()(reconstructed, batch_hits).item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f}")

        # Plot Reconstruction Fidelity vs Density
        if density_stats:
            density_stats = np.array(density_stats)
            plt.figure(figsize=(10, 6))
            plt.hexbin(density_stats[:, 0], density_stats[:, 1], gridsize=30, cmap='YlOrRd', bins='log')
            plt.colorbar(label='Log10(Count)')
            plt.xlabel('Local Hit Density (Neighbors in radius 0.05)')
            plt.ylabel('Reconstruction MSE')
            plt.title(f'Fidelity vs Density - Epoch {epoch+1}')
            plt.savefig(os.path.join(output_dir, f"fidelity_vs_density_epoch_{epoch+1}.png"))
            plt.close()

        # Generate 3D point cloud visualization with embedding-based coloring
        visualize_embeddings_3d(model, full_dataset, epoch, output_dir, n_events=1)

    # Save model
    save_path = os.path.join(output_dir, f"checkpoint_h{num_hits}_e{embed_dim}_neigh{use_neighborhood}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    if output_loss:
        loss_file_path = os.path.join(output_dir, output_loss)
        with open(loss_file_path, "w") as f:
            f.write(f"{avg_val_loss:.6f}\n")
        print(f"Loss saved to {loss_file_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hits", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_loss", type=str, default=None)
    parser.add_argument("--neighborhood", type=str, choices=["True", "False"], default="True")
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
        use_neighborhood=use_neighborhood
    )
