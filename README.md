# mull

Approaching zero-setup generative AI.  Experimental machine learning library modules. [darkshapes](https://github.com/darkshapes/)

<details open><summary>
index:

</summary>

> core :
> refined veteran methods

> [mll 00 - > lambda-condensed nested dict traversal](https://github.com/exdysa/mull/blob/main/modules/mll_00/src.py#L29) :

> [mll 01 - lambda nested dict traversal:](https://github.com/exdysa/mull/blob/main/modules/mll_00/src.py#L29)

> [mll 02 - recursive nested dict crawl](https://github.com/exdysa/mull/blob/main/modules/mll_02/src.py#L76)

> [mll 03 - basic nested dict comparison](https://github.com/exdysa/mull/blob/main/modules/mll_03/src.py#L19):

> [mll 04 - loading safetensors](https://github.com/exdysa/mull/blob/main/modules/mll_04/src.py#L5)

> [mll 05 - loading gguf](https://github.com/exdysa/mull/blob/main/modules/mll_05/src.py#L2)

> [mll 06 - dict crawler](https://github.com/exdysa/mull/blob/main/modules/mll_06/src.py#L14)

> [mll 07 - nn id system](https://github.com/exdysa/mull/blob/main/modules/mll_07/src.py#L2)

> [mll 08 - seed methods](https://github.com/exdysa/mull/blob/main/modules/mll_08/src.py#L7)

> [mll 09 - token encoder type 1](https://github.com/exdysa/mull/modules/mll_09/src.py#L12)

> mll 10 - minimal inference code for generative images

> [mll 11 - pipe constructor](https://github.com/exdysa/mull/blob/main/modules/mll_11/src.py#L93)

> [mll 13 - token encoder type 2]

> [mll 14 - token encoder type 3]

> [mll 15 - prototype token sculptor revisiting mll 08]

> [mll 16 - alternate methods of torch.no_grad inference]

> [mll 17 - modular variable autoencoder component]

> [mll 18 - output image formatting]

> [mll 19 - metadata encoding method 1/comparison]

> [mll 20 - self-embedding hash/snapshots]

> ...
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
> git clone https://github.com/exdysa/mull.git
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
> git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 mull/metadata/STA-XL
> ```

</details>