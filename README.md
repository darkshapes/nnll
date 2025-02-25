

# nnll

`nnll` (or <em>null</em>) is a comprehensive AI toolkit for managing and processing Diffusion and Large Language Models (LLMs). The project is divided into highly modular, ready-to-use components, and may appeal to researchers or developers working individual experiments or in the general field of large-scale machine learning model deployment.

We currently support 🧨Diffusers, 🤗Transformers, 🦙Llama inference, and refined methods for tasks such as extracting and classifying metadata, pipeline preparation, GPU configuration, consumer-grade system optimization, and a variety of generative AI preparations.
<br>
<br>

[![Python application](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml/badge.svg)](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml)<br>
![GitHub repo size](https://img.shields.io/github/repo-size/darkshapes/nnll)<br>
![Discord](https://img.shields.io/discord/1266757128249675867)<br>
<br>

## use

`nnll-parse`   - Lookup model metadata headers and write out to .json file.<br>
`nnll-find`    - Scan .json files from `-parse` for a specific string pattern within layer contents.<br>
<!-- `nnll-index`   - Identify available models within a given path and create a database of their attributes<br> -->
`nnll-toc`     - (run from root folder only) Recreate the project table of contents, populating preview and navigation links for IDEs<br>
<br>

## imports
`import nnll_**.src`
<br><br>

## [modules table of contents](https://github.com/darkshapes/nnll/blob/main/modules/README.md)

> [!NOTE]
> x/dev branch: updated **~=nightly**.<br>
> main branch : stable, devoted to **long-term support**.
<br><br>

## development setup


##### clone repo

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
<br><br><br>
