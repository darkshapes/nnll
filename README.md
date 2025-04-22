<div align="center">

![nnll75_transparent](https://github.com/user-attachments/assets/de8c1a49-4695-4c4b-b7c4-29fba483a65d)</div>
# nnll <br><sub>neural network link library</sub>

`nnll` (or <em>null</em>) is a project incubator and AI toolkit for managing and processing Diffusion and Large Language Models (LLMs). The project is divided into modular, ready-to-use components and may appeal to researchers or developers working in the general field of machine learning.

* Generative AI pipeline preparation & execution
* Extracting and classifying metadata from images/models
* Consumer-grade GPU/CPU inference optimization
* Misc UX/UI Experimentation
* 🧨Diffusers, 🤗Transformers, 🦙Ollama, 🍏MLX, 🌀DSPy, 🚅LiteLLM
* :shipit: <br><br>

[![Python application test status](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml/badge.svg)](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml) <br>
![commits per month](https://img.shields.io/github/commit-activity/m/darkshapes/nnll?color=indigo)<br>
![code size](https://img.shields.io/github/languages/code-size/darkshapes/nnll?color=navy)<br>
![Discord](https://img.shields.io/discord/1266757128249675867?color=black)<br><br>

## setup <br> <sub>`clone` -> `venv` -> `activate` -> `pip install "nnll[module]"`<sub>
#####  clone repo
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
> - install the bare minimum:
> ```
> pip install -e nnll
> ```
> - install select packages:
> ```
> pip install -e "nnll[nnll_33,nnll_56]"
> ```
> - install all packages :
> ```
> pip install -e "nnll[dev]"
>```
##### Done.
</details></details></details><br>

## use
Some modules are full scripts and can be run from command line. These are written here:

`zodiac`        - Experimental generative system<br>
`astra`        - Live diagnostic console<br>
`nnll-hash`    - Hash the layer metadata from models within a directory and write out to console.<br>
`nnll-parse`   - Process metadata headers from a model file or directory of models and write out to individual JSON files.<br>
`nnll-find`    - Search a local directory of model layer files (HuggingFace🤗 index.json, JSON from `nnll-parse`)<br>
<br>

Each module contains 1-5 functions or 1-2 classes and its own test routines. There are multiple ways to integrate nnll into a project (sorted by level of involvement)

- *Recommended* : Add the project as a dependency including only modules that are needed with `"nnll[nnll_04,nnll_16]" @ git+https://github.com/darkshapes/nnll`
- Install the entire project as a dependency via `nnll @ git+https://github.com/darkshapes/nnll`
- Basic clone or fork of the project
-  Use a [submodule](https://github.blog/open-source/git/working-with-submodules/)
- [Filter](https://github.com/newren/git-filter-repo/) a clone of the project to a single subfolder and include it in your own


`nnll` is a 'living' project. Like a spoken language, it evolves over time. For this reason, we prefer 'living' duplications of the repo. If you still want a static hard copy, you are welcome to copy and paste folders or code wherever you please.<br><br>
## contributing
```
* Environment  : uv
* Testing      : pytest -vv tests/*.py
* Formatting   : ruff/better align
* Linting      : ruff/pylint
* Type Checking: pylance/pyright
* Spelling     : typos vsc
* Docstrings   : sphinx
```
<br>

![Alt](https://repobeats.axiom.co/api/embed/13fd2c53953a777ae8583f620fa8bd014baadef1.svg "Repobeats analytics image")