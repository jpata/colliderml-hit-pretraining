import subprocess
import os
import re

def run_train(num_hits, embed_dim, max_events):
    # Set LD_LIBRARY_PATH to ensure correct libstdc++.so.6 is used
    env = os.environ.copy()
    pixi_lib = os.path.join(os.getcwd(), ".pixi/envs/default/lib")
    env["LD_LIBRARY_PATH"] = f"{pixi_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        "python", "train_example.py",
        "--num_hits", str(num_hits),
        "--embed_dim", str(embed_dim),
        "--max_events", str(max_events),
        "--epochs", "1",
        "--batch_size", "4"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    
    # Extract validation loss using regex
    # Expected line: "Epoch 1 Validation Loss: 0.123456"
    match = re.search(r"Validation Loss: ([\d.]+)", result.stdout)
    if match:
        return float(match.group(1))
    return None

def main():
    # Define ranges to scan
    num_hits_list = [128, 256]
    embed_dims = [16, 32]
    dataset_sizes = [100, 500]

    results_file = "scan_results.txt"
    
    with open(results_file, "w") as f:
        f.write("num_hits,embed_dim,max_events,loss\n")
        
        for num_hits in num_hits_list:
            for embed_dim in embed_dims:
                for max_events in dataset_sizes:
                    try:
                        loss = run_train(num_hits, embed_dim, max_events)
                        if loss is not None:
                            line = f"{num_hits},{embed_dim},{max_events},{loss:.6f}"
                            print(f"Result: {line}")
                            f.write(line + "\n")
                            f.flush() # Ensure it's written immediately
                        else:
                            print(f"Could not find validation loss in output for num_hits={num_hits}, embed_dim={embed_dim}, max_events={max_events}")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed configuration: num_hits={num_hits}, embed_dim={embed_dim}, max_events={max_events}")
                        print(e.stderr)

if __name__ == "__main__":
    main()
