# nnll

Approaching zero-setup generative AI.  Experimental machine learning library modules. [darkshapes](https://github.com/darkshapes/)

neural network link library : Approaching zero-setup generative AI.
Experimental python machine learning modules. [darkshapes](https://github.com/darkshapes/)

Please link if you purpose our code:
```
[nnll](https://github.com/darkshapes/nnll)
```

<details open><summary>
index:

</summary>


> `modules` : lab experiments, prototype and comparative work

> `core` : future sub-module highlighting apex methods from `modules`,

> [nnll 00 - lambda-condensed nested dict traversal](https://github.com/darkshapes/nnll/blob/main/modules/nnll_00/src.py#L29)

> [nnll 01 - lambda-condensed nested dict traversal:](https://github.com/darkshapes/nnll/blob/main/modules/nnll_01/src.py#L8)

> [nnll 02 - recursive nested dict crawl](https://github.com/darkshapes/nnll/blob/main/modules/nnll_02/src.py#L76)

> [nnll 03 - basic nested dict comparison](https://github.com/darkshapes/nnll/blob/main/modules/nnll_03/src.py#L19)

> [nnll 04 - loading safetensors](https://github.com/darkshapes/nnll/blob/main/modules/nnll_04/src.py#L5)

> [nnll 05 - loading gguf](https://github.com/darkshapes/nnll/blob/main/modules/nnll_05/src.py#L2)

> [nnll 06 - dict crawler](https://github.com/darkshapes/nnll/blob/main/modules/nnll_06/src.py#L14)

> [nnll 07 - nn id system](https://github.com/darkshapes/nnll/blob/main/modules/nnll_07/src.py#L2)

> [nnll 08 - seed methods](https://github.com/darkshapes/nnll/blob/main/modules/nnll_08/src.py#L2)

> [nnll 09 - token encoder type 1](https://github.com/darkshapes/nnll/modules/nnll_09/src.py#L12)

> [nnll 10 - minimal generative inference (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_10/src.py#L15)

> [nnll 11 - pipe constructor](https://github.com/darkshapes/nnll/blob/main/modules/nnll_11/src.py#L93)

> [nnll 12 -  (incomplete) iterative text encoder initialization](https://github.com/darkshapes/nnll/blob/main/modules/nnll_12/src.py#L5)

> [nnll 13 - system capability agent (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_13/src.py#L1)

> [nnll 14 - iterative gpu check](https://github.com/darkshapes/nnll/blob/main/modules/nnll_14/src.py#L7)

> [nnll xx - token encoder type 2]

> [nnll xx - token encoder type 3]

> [nnll xx - prototype token sculptor revisiting nnll 08]

> [nnll xx - alternate methods of torch.no_grad inference]

> [nnll xx - modular variable autoencoder component]

> [nnll xx - output image formatting]

> [nnll xx - metadata encoding method 1/comparison]

> [nnll xx - self-embedding hash/snapshots]

> ...
</details>

<details><summary>
setup

</summary>

###### create virtual environment
> ```
> py -3.12 -m venv .venv_nnll
> ``` -->

###### activate (windows)
> ```
> Set-ExecutionPolicy Bypass -Scope Process -Force; .venv_nnll\Scripts\Activate.ps1
> ```

###### activate( linux | macos)
> ```
> .venv_nnll\bin\activate
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
> git clone https://github.com/darkshapes/mull.git
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
> git clone https://huggingface.co/exdysa/metadata nnll/metadata
> ```

</details>