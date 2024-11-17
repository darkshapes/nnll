# null

Approaching zero-setup generative AI. Experimental modules. [darkshapes](https://github.com/darkshapes/)

<details open><summary>
index:

</summary>

> nll 00 :
> `map` `lambda`  nested `dict` traversal

 > nll 01 :
 > `lambda` nested `OrderedDict` traversal

> nll 02 :
> recursive dict traversal method

> nll 03 :
> recursive dict traversal alternate method

> nll 04 :
> comparison of safetensor loading methods

>

</details>

<details><summary>
setup

</summary>

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
> git clone https://github.com/exdysa/null.git
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

</details>