# nnll

### neural network link library

### Approaching zero-setup, fault-tolerant generative AI. [darkshapes](https://github.com/darkshapes/)

> `modules` : experiments, prototypes, comparative work, studies

> `core` : future sub-modules highlighting apex methods from `modules`,


#### modules index :

<details><summary>

> `  nnll_00-19`
</summary>

> [nnll 00 - lambda-condensed nested dict traversal](https://github.com/darkshapes/nnll/blob/main/modules/nnll_00/src.py#L29)<br>
> [nnll 01 - lambda-condensed nested dict traversal:](https://github.com/darkshapes/nnll/blob/main/modules/nnll_01/src.py#L8)<br>
> [nnll 02 - recursive nested dict crawl](https://github.com/darkshapes/nnll/blob/main/modules/nnll_02/src.py#L76)<br>
> [nnll 03 - basic nested dict comparison](https://github.com/darkshapes/nnll/blob/main/modules/nnll_03/src.py#L19)<br>
> [nnll 04* - loading safetensors comparative analysis](https://github.com/darkshapes/nnll/blob/main/modules/nnll_04/src.py#L5)<br>
> [nnll 05* - loading gguf comparative analysis](https://github.com/darkshapes/nnll/blob/main/modules/nnll_05/src.py#L2)<br>
> [nnll 06 - dict crawler](https://github.com/darkshapes/nnll/blob/main/modules/nnll_06/src.py#L14)<br>
> [nnll 07 - nn class/type id system](https://github.com/darkshapes/nnll/blob/main/modules/nnll_07/src.py#L2)<br>
> [nnll 08 - seed methods](https://github.com/darkshapes/nnll/blob/main/modules/nnll_08/src.py#L2)<br>
> [nnll 09 - token encoder type 1](https://github.com/darkshapes/nnll/modules/nnll_09/src.py#L12)<br>
> [nnll 10 - minimal diffusers sdxl inference (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_10/src.py#L15)<br>
> [nnll 11 - pipe constructor](https://github.com/darkshapes/nnll/blob/main/modules/nnll_11/src.py#L93)<br>
> [nnll 12 - iterative text encoder initialization (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_12/src.py#L5)<br>
> [nnll 13 - system capability agent (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_13/src.py#L1)<br>
> [nnll 14 - iterative gpu counter](https://github.com/darkshapes/nnll/blob/main/modules/nnll_14/src.py#L7)<br>
> [nnll 15 - dynamic iterative gpu counter (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_15/src.py#L24)<br>
> [nnll 16 - scalable, modular gpu device class system (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_16/src.py#L6)<br>
> [nnll 17 - example device classes for nll_16 (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_17/src.py#L4)<br>
> [nnll 18 - alternate token encoder for extra prompt length](https://github.com/darkshapes/nnll/blob/main/modules/nnll_18/src.py#L5)<br>
> [nnll 19 - study of `spandrel`](https://github.com/darkshapes/nnll/blob/main/modules/nnll_19/study.py#L5)<br>

</details>

<details><summary>

> `  nnll_20-39`

</summary>

> [nnll 20 - minimal diffusers flux inference (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_20/src.py#L8)<br>
> [nnll 21 - minimal symlink routine](https://github.com/darkshapes/nnll/blob/main/modules/nnll_21/src.py#L5)<br>
> [nnll 22* - diffusers symlink routine(nnll_19, 21)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_22/src.py#L5)<br>
> [nnll 23* - dynamic module constructor (nnll 11,12,14,15,16)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_23/src.py#L5)<br>

</details>

<details><summary>

> `   nnll_xx (planned explorations)`

</summary>

> [nnll xx - token encoder type 3]<br>
> [nnll xx - prototype token sculptor revisiting nnll 08]<br>
> [nnll xx - alternate methods of torch.no_grad inference]<br>
> [nnll xx - modular variable autoencoder component]<br>
> [nnll xx - output image formatting]<br>
> [nnll xx - metadata encoding method 1/comparison]<br>
> [nnll xx - self-embedding hash/snapshots]<br>
> ...

</details>


<details><summary>

#### setup

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