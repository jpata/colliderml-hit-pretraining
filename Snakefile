NUM_HITS = [128, 256]
EMBED_DIMS = [16, 32]
DATASET_SIZES = [100, 500]
NEIGHBORHOOD = ["True", "False"]

rule all:
    input:
        "scan_results_snakemake.csv",
        "loss_scaling.png"

rule train:
    output:
        loss = "results/h{num_hits}_e{embed_dim}_s{max_events}_n{neighborhood}/loss.txt",
        model = "results/h{num_hits}_e{embed_dim}_s{max_events}_n{neighborhood}/checkpoint_h{num_hits}_e{embed_dim}_neigh{neighborhood}.pth"
    params:
        output_dir = "results/h{num_hits}_e{embed_dim}_s{max_events}_n{neighborhood}"
    shell:
        """
        LD_LIBRARY_PATH=$PIXI_PROJECT_ROOT/.pixi/envs/default/lib \
        python train_example.py \
            --num_hits {wildcards.num_hits} \
            --embed_dim {wildcards.embed_dim} \
            --max_events {wildcards.max_events} \
            --neighborhood {wildcards.neighborhood} \
            --epochs 5 \
            --batch_size 16 \
            --output_dir {params.output_dir} \
            --output_loss loss.txt
        """

rule aggregate:
    input:
        expand("results/h{num_hits}_e{embed_dim}_s{max_events}_n{neighborhood}/loss.txt", 
               num_hits=NUM_HITS, embed_dim=EMBED_DIMS, max_events=DATASET_SIZES, neighborhood=NEIGHBORHOOD)
    output:
        "scan_results_snakemake.csv"
    run:
        with open(output[0], "w") as out:
            out.write("num_hits,embed_dim,max_events,neighborhood,loss\n")
            for h in NUM_HITS:
                for e in EMBED_DIMS:
                    for s in DATASET_SIZES:
                        for n in NEIGHBORHOOD:
                            loss_file = f"results/h{h}_e{e}_s{s}_n{n}/loss.txt"
                            with open(loss_file) as f:
                                loss = f.read().strip()
                            out.write(f"{h},{e},{s},{n},{loss}\n")

rule visualize:
    input:
        "scan_results_snakemake.csv"
    output:
        "loss_scaling.png"
    shell:
        """
        LD_LIBRARY_PATH=$PIXI_PROJECT_ROOT/.pixi/envs/default/lib \
        python visualize_scan.py
        """
