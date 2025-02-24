

# nnll

## neural network link library
`nnll` (or <em>null</em>) is a comprehensive AI toolkit for managing and processing Diffusion and Large Language Models (LLMs). The project is divided into highly modular, ready-to-use components, and may appeal to researchers or developers working individual experiments or in the general field of large-scale machine learning model deployment.

We currently support 🧨Diffusers, 🤗Transformers, 🦙Llama inference, and refined methods for tasks such as extracting and classifying metadata, pipeline preparation, GPU configuration, consumer-grade system optimization, and a variety of generative AI preparations.
<br>

[![Python application](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml/badge.svg)](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml)<br>
![GitHub repo size](https://img.shields.io/github/repo-size/darkshapes/nnll)<br>
![Discord](https://img.shields.io/discord/1266757128249675867)<br>
<br>

## use

`nnll-parse`   - Process metadata headers from a model file or directory of models and write out to individual .json files.<br>
`nnll-find`    - Scan .json files from `-parse` for string patterns within tensor layer metadata and output matches to console.<br>
`nnll-index`   - Identify available models within a given path and create a .json database file of their attributes.<br>
`nnll-toc`     - (run from /nnll folder) Recreate the project table of contents, updating function names and populating navigation links for IDEs and GitHub<br>
<br>

## imports
`import nnll_**.src`
<br><br>

## [modules table of contents](https://github.com/darkshapes/nnll/blob/main/modules/README.md)

> [!NOTE]
> x/dev branch: updated **~=nightly**.<br>
> main branch : stable, devoted to **long-term support**.
<br><br>

## contributing

This project follows the principles of test-driven development (TDD) and continuous itegration/continuous delivery (CI/CD). Submissions will be expected to follow these guidelines.

## setup

##### clone repo

> ```
> git clone https://github.com/darkshapes/nnll.git
> ```

<details> <summary> <a>Next--></a></summary>

#####  create virtual environment
> ```
> python3 -m venv .venv_nnll
> ```

<details> <summary> <a>Next--></a></summary>

##### 3 (windows powershell) activate
> ```
> Set-ExecutionPolicy Bypass -Scope Process -Force; .venv_nnll\Scripts\Activate.ps1
> ```

##### 3 ( linux | macos) activate
> ```
> source .venv_nnll/bin/activate
> ```

<details> <summary> <a>Next--></a></summary>

##### 4 install
> ```
> pip install -e nnll
> ```
or
>
> pip install -e 'nnll\[dev\]'
>

##### Done.
</details>
</details>
</details>
<br><br><br>
