import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    if not os.path.exists("scan_results_snakemake.csv"):
        print("Error: scan_results_snakemake.csv not found.")
        return

    df = pd.read_csv("scan_results_snakemake.csv")
    
    # Ensure correct data types
    df['num_hits'] = df['num_hits'].astype(int)
    df['embed_dim'] = df['embed_dim'].astype(int)
    df['max_events'] = df['max_events'].astype(int)
    df['loss'] = df['loss'].astype(float)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Scaling with Dataset Size
    sns.lineplot(
        data=df, 
        x="max_events", 
        y="loss", 
        hue="embed_dim", 
        style="num_hits", 
        markers=True,
        palette="viridis",
        ax=axes[0]
    )
    axes[0].set_title("Scaling with Dataset Size")
    axes[0].set_xlabel("max_events")
    axes[0].set_ylabel("Validation Loss")

    # Plot 2: Scaling with Embedding Dimension
    sns.lineplot(
        data=df, 
        x="embed_dim", 
        y="loss", 
        hue="max_events", 
        style="num_hits", 
        markers=True,
        palette="magma",
        ax=axes[1]
    )
    axes[1].set_title("Scaling with Embedding Dimension")
    axes[1].set_xlabel("embed_dim")
    axes[1].set_ylabel("Validation Loss")

    # Plot 3: Scaling with Number of Hits
    sns.lineplot(
        data=df, 
        x="num_hits", 
        y="loss", 
        hue="max_events", 
        style="embed_dim", 
        markers=True,
        palette="plasma",
        ax=axes[2]
    )
    axes[2].set_title("Scaling with Number of Hits")
    axes[2].set_xlabel("num_hits")
    axes[2].set_ylabel("Validation Loss")

    plt.tight_layout()
    plt.savefig("loss_scaling.png")
    print("Multi-parameter scaling plots saved to loss_scaling.png")

if __name__ == "__main__":
    main()
