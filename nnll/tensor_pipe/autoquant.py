### <!-- // /*  SPDX-License-Identifier: LGPL-3.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


def convert_repo(conditions: tuple) -> str:
    """card"""
    from nnll.configure import ensure_path
    from nnll.monitor.file import nfo
    from huggingface_hub import constants, snapshot_download
    import subprocess

    constants.HF_HUB_OFFLINE = 0
    constants.HF_XET_HIGH_PERFORMANCE = 1
    constants.HF_HUB_ENABLE_HF_TRANSFER = 1

    ensure_path(conditions["folder_path_named"])

    if conditions["library"] == "gguf":
        folder_path_named = snapshot_download(repo_id=conditions["repo"], local_dir=conditions["folder_path_named"])
        command = ["convert_hf_to_gguf.py", folder_path_named]
        if conditions["quantization"]:
            command.extend([" --outtype", f"{conditions['quantization']}"])

    else:
        if conditions["library"] == "mlx":
            command = ["mlx_lm.convert", "--hf-path", conditions["repo"], "--mlx-path", conditions["folder_path_named"]]
            if conditions["quantization"]:
                command.extend(["-q", "--q-bits", f"{conditions['quantization']}"])
        else:
            command = ["mflux-save", "--model", conditions["repo"], "--base-model", conditions["library"], "--path", conditions["folder_path_named"]]
            if conditions["quantization"]:
                command.extend(["--quantize", f"{conditions['quantization']}"])

    try:
        output = subprocess.run(command, check=True)
    except (TypeError, subprocess.CalledProcessError):
        nfo(f"command was {command}")
    nfo(f"status: {output}")
    print(output)
    return conditions["folder_path_named"]
