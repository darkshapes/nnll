# null

Approaching zero-setup generative AI.

<!-- ###### install [uv](https://docs.astral.sh/uv/)
###### create virtual environment
> ```
> uv venv .null
> ```

###### activate (windows)
> ```
> Set-ExecutionPolicy Bypass -Scope Process -Force; .null\Scripts\Activate.ps1
> ```

###### activate( linux | macos)
> ```
> .null\bin\activate
> ```

###### install torch (nvidia/cuda device)
> ```
> uv pip install torch==2.3.1+cu121 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121 --compile-bytecode
> ```

###### install torch (apple/mps device)
> ```
> uv pip install torch torchvision torchaudio xformers --compile-bytecode
> ```

###### add environment variables (windows)
>
> set HF_HUB_OFFLINE=True; set DISABLE_TELEMETRY=YES
>

###### add environment variables (linux/macos)
>
> export HF_HUB_OFFLINE=True && export DISABLE_TELEMETRY=YES
>

###### clone repo
> ```
> git clone https://github.com/darkshapes/null.git
> ```

###### install null
> ```
> uv pip install -e null --compile-bytecode
> ``` -->


<
###### create virtual environment
> ```
> py -3.12 -m venv .venv_null
> ``` -->

###### activate (windows)
> ```
> Set-ExecutionPolicy Bypass -Scope Process -Force; .venv_null\Scripts\Activate.ps1
> ```

###### activate( linux | macos)
> ```
> .venv_null\bin\activate
> ```

###### upgrade pip
> ```
> python -m pip install --upgrade pip
> ```

###### install torch (nvidia/cuda device)
> ```
> pip install torch==2.3.1+cu121 torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
> ```

###### install torch (apple/mps device)
> ```
> pip install torch torchvision torchaudio xformers flash-attn
> ```

###### clone repo
> ```
> git clone https://github.com/darkshapes/null.git
> ```

###### install null
> ```
> pip install -e null
> ```

###### add environment variables (windows)
>
> $env:HF_HUB_OFFLINE = "True"; $env:DISABLE_TELEMETRY = "YES"; $env:GIT_LFS_SKIP_SMUDGE = "1"
>

###### add environment variables (linux/macos)
>
> export HF_HUB_OFFLINE=True && export DISABLE_TELEMETRY=YES && export GIT_LFS_SKIP_SMUDGE=1
>

##### clone metadata
> ```
> git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 null/metadata/STA-XL
> ```

<!--
###### install ninja
> ```
> uv pip install ninja --compile-bytecode
> ```

###### set max jobs for compilation
> ```
> set MAX_JOBS=4;
> ```

###### install flash attention
> ```
>  uv pip install flash_attn-2.5.9.post1+cu122torch2.3.1cxx11abiFALSE-cp312-cp312-win_amd64.whl --no-build-isolation --compile-bytecode;
> ``` -->
