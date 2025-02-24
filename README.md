

# nnll

## neural network link library
`nnll` (or <em>null</em>) is a set of tensor packaging utilities and computer vision experiments. The modules may appeal to other AI developers, researchers and tinkerers, and are designed to effortlessly fold into others' projects.
<br>
<br>

[![Python application](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml/badge.svg)](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml)<br>
![GitHub repo size](https://img.shields.io/github/repo-size/darkshapes/nnll)<br>
![Discord](https://img.shields.io/discord/1266757128249675867)<br>
<br>

## use

`nnll-parse`   - Lookup model metadata headers and write out to .json file.<br>
`nnll-find`    - Scan .json files from `-parse` for a specific string pattern within layer contents.<br>
`nnll-index`   - Identify available models within a given path and create a database of their attributes<br>
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
