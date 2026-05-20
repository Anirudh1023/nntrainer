import os
from huggingface_hub import snapshot_download

model_id = "LiquidAI/LFM2.5-350M"
save_dir = "/Users/anirudh/Anirudh/nntrainer/Applications/CausalLM/res/lfm2/lfm2.5-350m"

os.makedirs(save_dir, exist_ok=True)

print(f"Downloading {model_id} to {save_dir}...")
snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    allow_patterns=["*.json", "*.model", "*.txt", "*.py"],
    ignore_patterns=["*.bin", "*.safetensors", "*.msgpack", "*.h5"]
)
print("Download complete.")
