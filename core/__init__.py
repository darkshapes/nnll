from config import  contents as config

index = config.model_indexer()
capacity = config.sys_cap()

system_capacity = capacity.write_capacity()
create_index = index.write_index()

# import subprocess
# import sys
# from dulwich import porcelain

# os.environ['HF_HUB_OFFLINE'] = "True" # visible in this process + all children
# os.environ['DISABLE_TELEMETRY'] = "True"
# os.environ['GIT_LFS_SKIP_SMUDGE'] = "True"

# if not os.path.exists("metadata/STA-XL"):
#     repo_target = { "url":  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0", "target": ".metadata/STA-XL"}
#     porcelain.clone((k, v) for k, v in repo_target.items())