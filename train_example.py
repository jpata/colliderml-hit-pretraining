"""
Point Cloud Pretraining for Calorimeter Hits.
Uses a Masked Point Modeling (MPM) approach with a Transformer encoder.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import awkward as ak
from tqdm import tqdm

class CalorimeterDataset(Dataset):
    def __init__(self, file_path, num_hits=2048, max_events=None):
        print(f"Loading data from {file_path}...")
        self.events = ak.from_parquet(file_path)
        if max_events is not None:
            self.events = self.events[:max_events]
        self.num_hits = num_hits
        
        # Pre-calculated normalization constants (approximate)
        self.coord_scale = 5000.0
        self.energy_scale = 0.1

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        
        # Extract features (x, y, z, energy)
        x = np.array(event.x) / self.coord_scale
        y = np.array(event.y) / self.coord_scale
        z = np.array(event.z) / self.coord_scale
        e = np.array(event.total_energy) / self.energy_scale
        
        # Combine into a single hit array (N, 4)
        hits = np.stack([x, y, z, e], axis=1).astype(np.float32)
        
        # Fixed-size sampling/padding for batching
        n_hits = hits.shape[0]
        if n_hits >= self.num_hits:
            # Randomly sample num_hits
            indices = np.random.choice(n_hits, self.num_hits, replace=False)
            hits = hits[indices]
        else:
            # Pad with zeros
            padding = np.zeros((self.num_hits - n_hits, 4), dtype=np.float32)
            hits = np.concatenate([hits, padding], axis=0)
            
        return torch.from_numpy(hits)

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
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, activation='relu'
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
        
        # Prepare tokens for transformer (keep some original, mask some)
        # For simplicity in this basic version, we replace masked tokens with self.mask_token
        # and feed the entire sequence to the transformer.
        input_embeddings = hit_embeddings.clone()
        for b in range(B):
            input_embeddings[b, mask_indices[b]] = self.mask_token
            
        # Transformer processes all tokens
        latent = self.transformer(input_embeddings) # (B, N, embed_dim)
        
        # Reconstruct only the masked ones (for loss calculation, we can predict all and index)
        reconstructed = self.reconstructor(latent) # (B, N, 4)
        
        return reconstructed, mask_indices

def train(num_hits=256, embed_dim=16, max_events=None, epochs=1, batch_size=4):
    # Force CPU due to GPU incompatibility in this environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: num_hits={num_hits}, embed_dim={embed_dim}, max_events={max_events}")

    # Hyperparameters
    lr = 1e-3
    
    # Dataset and Loader
    full_dataset = CalorimeterDataset("train-00000-of-00100.parquet", num_hits=num_hits, max_events=max_events)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, Optimizer, Loss
    model = MaskedPointModel(embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training Loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_hits in pbar:
            batch_hits = batch_hits.to(device)
            
            # Forward pass
            reconstructed, mask_indices = model(batch_hits, mask_ratio=0.5)
            
            # Extract only the masked original hits and predictions
            masked_targets = []
            masked_preds = []
            for b in range(batch_hits.shape[0]):
                masked_targets.append(batch_hits[b, mask_indices[b]])
                masked_preds.append(reconstructed[b, mask_indices[b]])
            
            loss = criterion(torch.cat(masked_preds), torch.cat(masked_targets))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.6f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_hits in val_dataloader:
                batch_hits = batch_hits.to(device)
                reconstructed, mask_indices = model(batch_hits, mask_ratio=0.5)
                
                masked_targets = []
                masked_preds = []
                for b in range(batch_hits.shape[0]):
                    masked_targets.append(batch_hits[b, mask_indices[b]])
                    masked_preds.append(reconstructed[b, mask_indices[b]])
                
                loss = criterion(torch.cat(masked_preds), torch.cat(masked_targets))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.6f}")

    print("Pretraining complete!")
    
    # Save model
    save_path = f"checkpoint_h{num_hits}_e{embed_dim}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_hits", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    train(
        num_hits=args.num_hits,
        embed_dim=args.embed_dim,
        max_events=args.max_events,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
