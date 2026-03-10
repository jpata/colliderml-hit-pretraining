DATASET_SIZES = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
NUM_HITS = 256
EMBED_DIM = 32
NEIGHBORHOOD = "True"

rule all:
    input:
        "scan_results_snakemake.csv",
        "loss_scaling.png"

rule train:
    output:
        loss = "results/s{max_events}/loss.txt",
        model = "results/s{max_events}/checkpoint.pth"
    params:
        output_dir = "results/s{max_events}"
    shell:
        """
        python scripts/train_example.py \
            --num_hits {NUM_HITS} \
            --embed_dim {EMBED_DIM} \
            --max_events {wildcards.max_events} \
            --neighborhood {NEIGHBORHOOD} \
            --epochs 2 \
            --batch_size 16 \
            --output_dir {params.output_dir} \
            --output_loss loss.txt \
            --output_checkpoint checkpoint.pth
        """

rule aggregate:
    input:
        expand("results/s{max_events}/loss.txt", max_events=DATASET_SIZES)
    output:
        "scan_results_snakemake.csv"
    run:
        import os
        import pandas as pd
        rows = []
        for s in DATASET_SIZES:
            loss_file = f"results/s{s}/loss.txt"
            if os.path.exists(loss_file):
                df = pd.read_csv(loss_file)
                if not df.empty:
                    final_val_loss = df.iloc[-1]["val_loss"]
                    rows.append({
                        "max_events": s, 
                        "loss": final_val_loss, 
                        "neighborhood": NEIGHBORHOOD,
                        "num_hits": NUM_HITS,
                        "embed_dim": EMBED_DIM
                    })
        
        pd.DataFrame(rows).to_csv(output[0], index=False)

rule visualize:
    input:
        "scan_results_snakemake.csv"
    output:
        "loss_scaling.png"
    shell:
        """
        python scripts/visualize_scan.py
        """
