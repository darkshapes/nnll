# def mass_download(local_folder: str, repo_id_stack: List):
#     from huggingface_hub import snapshot_download

#     for repo_id in repo_id_stack:
#         snapshot_download(local_dir=local_folder, repo_id=repo_id, allow_patterns=["*.safetensors*", "*.gguf"])
