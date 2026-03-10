import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    if not os.path.exists("scan_results_snakemake.csv"):
        print("Results file not found.")
        return
        
    df = pd.read_csv("scan_results_snakemake.csv")
    
    # Simple plot: Loss vs Dataset Size for different neighborhood settings
    plt.figure(figsize=(10, 6))
    for neigh in df['neighborhood'].unique():
        subset = df[df['neighborhood'] == neigh]
        # Average across num_hits/embed_dims for now
        avg = subset.groupby('max_events')['loss'].mean().reset_index()
        plt.plot(avg['max_events'], avg['loss'], marker='o', label=f'Neighborhood={neigh}')
        
    plt.xlabel('Max Events (Dataset Size)')
    plt.ylabel('Loss')
    plt.title('Loss vs Training Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_scaling.png')
    print("Saved loss_scaling.png")

if __name__ == "__main__":
    main()
