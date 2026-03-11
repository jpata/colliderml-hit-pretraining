DATASET_SIZES = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
NUM_HITS = 1024
EMBED_DIM = 64
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
    resources:
        slurm_partition="gpu",
        gres="gpu:l40:1",
        mem_mb=20000,
        runtime=240,
        cpus_per_task=8
    shell:
        """
        env
        python scripts/train_example.py \
            --num_hits {NUM_HITS} \
            --embed_dim {EMBED_DIM} \
            --max_events {wildcards.max_events} \
            --neighborhood {NEIGHBORHOOD} \
            --epochs 20 \
            --batch_size 128 \
            --output_dir {params.output_dir} \
            --output_loss loss.txt \
            --output_checkpoint checkpoint.pth
        """

rule aggregate:
    input:
        expand("results/s{max_events}/loss.txt", max_events=DATASET_SIZES)
    output:
        "scan_results_snakemake.csv"
    resources:
        mem_mb=8000,
        runtime=60,
        cpus_per_task=1
    shell:
        """
        env
        python scripts/aggregate_results.py \
            --dataset_sizes {DATASET_SIZES} \
            --num_hits {NUM_HITS} \
            --embed_dim {EMBED_DIM} \
            --neighborhood {NEIGHBORHOOD} \
            --output {output[0]}
        """

rule visualize:
    input:
        "scan_results_snakemake.csv"
    output:
        "loss_scaling.png"
    resources:
        mem_mb=8000,
        runtime=60,
        cpus_per_task=1
    shell:
        """
        env
        python scripts/visualize_scan.py
        """
