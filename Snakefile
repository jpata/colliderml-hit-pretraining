DATASET_SIZES = [100, 500, 1000, 2000, 5000]
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
        with open(output[0], "w") as out:
            out.write("max_events,loss,neighborhood\n")
            for s in DATASET_SIZES:
                loss_file = f"results/s{s}/loss.txt"
                if os.path.exists(loss_file):
                    with open(loss_file) as f:
                        loss = f.read().strip()
                    out.write(f"{s},{loss},{NEIGHBORHOOD}\n")

rule visualize:
    input:
        "scan_results_snakemake.csv"
    output:
        "loss_scaling.png"
    shell:
        """
        python scripts/visualize_scan.py
        """
