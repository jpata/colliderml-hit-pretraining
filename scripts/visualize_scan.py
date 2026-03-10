import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    if not os.path.exists("scan_results_snakemake.csv"):
        print("Results file not found.")
        return
        
    df = pd.read_csv("scan_results_snakemake.csv")
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
    df = df.dropna(subset=['loss'])
    
    # Simple plot: Loss vs Dataset Size for different neighborhood settings
    plt.figure(figsize=(10, 6))
    
    # Identify unique parameter combinations
    group_cols = [c for c in df.columns if c not in ['max_events', 'loss']]
    for name, group in df.groupby(group_cols):
        # Create a legend label from the parameter combination
        if isinstance(name, tuple):
            label = ", ".join([f"{col}={val}" for col, val in zip(group_cols, name)])
        else:
            label = f"{group_cols[0]}={name}"
            
        avg = group.groupby('max_events')['loss'].mean().reset_index()
        plt.plot(avg['max_events'], avg['loss'], marker='o', label=label)
        
    plt.xlabel('Max Events (Dataset Size)')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Loss vs Training Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_scaling.png')
    print("Saved loss_scaling.png")

if __name__ == "__main__":
    main()
