<div align="center">

![nnll75_transparent](https://github.com/user-attachments/assets/de8c1a49-4695-4c4b-b7c4-29fba483a65d)</div>
# nnll

## neural network link library
`nnll` (or <em>null</em>) is a comprehensive AI toolkit for managing and processing Diffusion and Large Language Models (LLMs). The project is divided into highly modular, ready-to-use components, and may appeal to researchers or developers working in the general field of machine learning.

Library compatibility includes 🧨Diffusers, 🤗Transformers, 🦙Ollama, and focuses on refining methods for tasks such as extracting and classifying metadata, pipeline preparation, GPU configuration, consumer-grade system optimization, and a variety of direct and indirect generative AI preparations.
<br>

# :shipit:

[![Python application](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml/badge.svg)](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml)<br>
![GitHub repo size](https://img.shields.io/github/repo-size/darkshapes/nnll)<br>
![Discord](https://img.shields.io/discord/1266757128249675867)<br>
<br>

## use
Some modules are full scripts and can be run from command line. These are written here:

`nnll-parse`   - Process metadata headers from a model file or directory of models and write out to individual .json files.<br>
`nnll-find`    - Scan .json files from `-parse` for string patterns within tensor layer metadata and output matches to console.<br>
<br>

## specifics

Each module contains 1-5 functions or 1-2 classes and its own test routines. There are multiple ways to integrate nnll into a project (sorted by level of involvement)

- *Recommended* : Install the project as a dependency via `nnll @ git+https://github.com/darkshapes/nnll`
- Install the entire project as a dependency via `nnll @ git+https://github.com/darkshapes/nnll`
- Basic clone or fork of the project
-  Use a [submodule](https://github.blog/open-source/git/working-with-submodules/)
- [Filter](https://github.com/newren/git-filter-repo/) a clone of the project to a single subfolder and include it in your own


`nnll` is a 'living' project. Like a spoken language, it evolves over time. For this reason, we prefer 'living' duplications of the repo. If you still want a static hard copy, you are welcome to copy and paste folders or code wherever you please.

<br><br>

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
