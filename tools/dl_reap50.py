from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF",
    local_dir="/work/GLM-5.2-GGUF/REAP50",
    allow_patterns=["*.gguf", "*.json", "*.md"],
    max_workers=8,
)
print("DONE", p)
