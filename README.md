
# nnll
### neural network link library


### Approaching zero-setup, fault-tolerant generative AI inference.


`nnll` (or <em>null</em>) is a collection of versatile tools and components for generative AI applications. The routines are part of a process of rigorous experimentation, multiple iteration, careful study, and loving refinement. They may appeal to other AI developers, researchers and tinkerers, and are designed to individually import or effortlessly fold into the work of others' projects.

Goals:
- Compatability: Platform, framework, language agnostic
- Modularity   : Independently functioning, tested and ready-made components, avoiding dependencies
- Simplicity   : Streamline generative processing model chains
- Rapidity     : Granular and automated hyperparameter and resource optimization
- Reliability  : Failure-resistance on full-spectrum consumer hardware

> x/dev branch: updated **~=nightly**.<br>
> main branch : stable, devoted to **long-term support**.

 #### [[darkshapes](https://github.com/darkshapes/)]
<hr><br>

[![Python application](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml/badge.svg)](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml)
![GitHub repo size](https://img.shields.io/github/repo-size/darkshapes/nnll)
![Discord](https://img.shields.io/discord/1266757128249675867)<hr>

<h4>modules index :</h4><details><summary>

> `  nnll_00-19`</summary>

> [nnll 00 - lambda-condensed nested dict traversal](https://github.com/darkshapes/nnll/blob/main/modules/nnll_00/src.py#L29)<br>
> [nnll 01 - lambda-condensed nested dict traversal:](https://github.com/darkshapes/nnll/blob/main/modules/nnll_01/src.py#L8)<br>
> [nnll 04 - loading safetensors (based on previous performance analysis)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_04/src.py#L5)<br>
> [nnll 05 - loading gguf (based on previous performance analysis)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_05/src.py#L2)<br>
> [nnll 06 - basic recursive dict crawler](https://github.com/darkshapes/nnll/blob/main/modules/nnll_06/src.py#L14)<br>
> [nnll 07 - nn class/type id system](https://github.com/darkshapes/nnll/blob/main/modules/nnll_07/src.py#L2)<br>
> [nnll 08 - seed methods](https://github.com/darkshapes/nnll/blob/main/modules/nnll_08/src.py#L2)<br>
> [nnll 09 - token encoder type 1](https://github.com/darkshapes/nnll/blob/main/modules/nnll_09/src.py#L12)<br>
> [nnll 13 - system capability agent (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_13/src.py#L1)<br>
> [nnll 16 - scalable, modular gpu device class system (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_16/src.py#L6)<br>
> [nnll 17 - example device classes for nll_16 (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_17/src.py#L4)<br>
> [nnll 18 - alternate token encoder for extra prompt length (incomplete)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_18/src.py#L5)<br>

</details>

<details><summary>

> `  nnll_20-39`</summary>

> [nnll 22 - diffusers symlink routine(nnll 19, 21)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_22/src.py#L5)<br>
> [nnll 23 - dynamic module constructor (nnll 11, 12, 14, 15, 16)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_23/src.py#L5)<br>
> [nnll_24 - nested dictionary path return (nnll 02, 03, 06)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_24/src.py#L5)<br>
> [nnll_25 - tensor and hash data, regex comparison ](https://github.com/darkshapes/nnll/blob/main/modules/nnll_25/src.py#L9)<br>
> [nnll_26 - pytorch-specific seed and tensor routines](https://github.com/darkshapes/nnll/blob/main/modules/nnll_26/src.py)<br>
> [nnll_27 - pretty column-formatted console printer](https://github.com/darkshapes/nnll/blob/main/modules/nnll_27/src.py#L6)<br>
> [nnll_28 - torch pickletensor loader](https://github.com/darkshapes/nnll/blob/main/modules/nnll_28/src.py#L8)<br>
> [nnll_29 - cascading dict comparison filter](https://github.com/darkshapes/nnll/blob/main/modules/nnll_29/src.py#L11)<br>
> [nnll 30 - barebones json read/write](https://github.com/darkshapes/nnll/blob/main/modules/nnll_30/src.py#L5)<br>
> [nnll 31 - state dict layer name search](https://github.com/darkshapes/nnll/blob/main/modules/nnll_31/src.py#L18)<br>
> [nnll 32 - model header extractor (nnll_29)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_32/src.py#L7)<br>
> [nnll 33 - layer and tensor value comparison (nnll_24)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_33/src.py#L4)<br>
> [nnll 34 - model shard collector (inc)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_34/src.py#L4)<br>
> [nnll 35 - filename numeric fetch](https://github.com/darkshapes/nnll/blob/main/modules/nnll_35/src.py#L4)<br>
> [nnll 36 - state dict collector](https://github.com/darkshapes/nnll/blob/main/modules/nnll_36/src.py#L13)<br>
> [nnll_37 - model indexer(inc)](https://github.com/darkshapes/nnll/blob/main/modules/nnll_37/src.py#L11)<br>


</details>

<details><summary>

> `   nnll_xx (planned explorations)`</summary>

> [nnll xx - civitai model index/download]
> [nnll xx - token encoder type 3]<br>
> [nnll xx - prototype token sculptor revisiting nnll 08]<br>
> [nnll xx - alternate methods of torch.no_grad inference]<br>
> [nnll xx - modular variable autoencoder component]<br>
> [nnll xx - output image formatting]<br>
> [nnll xx - metadata encoding method 1/comparison]<br>
> [nnll xx - self-embedding hash/snapshots]<br>
> ...

</details><hr><h4> setup</h4>

<h5>clone repo</h5>

> ```
> git clone https://github.com/darkshapes/nnll.git
> ```

<details> <summary> <a>Next--></a></summary>

#####  create virtual environment
> ```
> py -m venv .venv_nnll
> ```

<details> <summary> <a>Next--></a></summary>

##### 3 activate --> (windows powershell)
> ```
> Set-ExecutionPolicy Bypass -Scope Process -Force; .venv_nnll\Scripts\Activate.ps1
> ```

##### 3 activate --> ( linux | macos)
> ```
> .venv_nnll/bin/activate
> ```

<details> <summary> <a>Next--></a></summary>

##### 4 install
> ```
> cd nnll
> pip install -e .
> ```

##### Done.
</details>
</details>
</details>
