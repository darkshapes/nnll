

<h1 style="border: 0;">nnll</h1>

## neural network link library

<div style="float: right; padding:10px; display: flex;">

[![Python application](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml/badge.svg)](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml)<br>
![GitHub repo size](https://img.shields.io/github/repo-size/darkshapes/nnll)<br>
![Discord](https://img.shields.io/discord/1266757128249675867)<br>

</div><div style="width: 75%; display: flex;">

`nnll` (or <em>null</em>) is a collection of versatile tools and components for tensor apps. The modular routines may appeal to other AI developers, researchers and tinkerers, and are designed to individually import or effortlessly fold into the work of others' projects.
</div><br>

## terminal commands

`nnll-parse`   - Lookup model metadata headers and write out to .json file.<br>
`nnll-find`    - Scan models for a specific string pattern within their state dict layer contents.<br>
`nnll-index`   - Identify available models within a given path and create a database of their attributes<br>
`nnll-toc`     - (run from root folder only) Recreate the project table of contents, populating preview and navigation links for IDEs<br>
<br><br>

## importing modules
`from nnll import modules.nnll_**.src`

## [modules table of contents](https://github.com/darkshapes/nnll/blob/main/modules/toc.md)


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
