
# nnll
### neural network link library

`nnll` (or <em>null</em>) is a collection of versatile tools and components for generative AI applications. The routines are part of a process of rigorous experimentation,  iteration and study, all lovingly refined. The modular routines may appeal to other AI developers, researchers and tinkerers, and are designed to individually import or effortlessly fold into the work of others' projects.

> x/dev branch: updated **~=nightly**.<br>
> main branch : stable, devoted to **long-term support**.

 #### [[darkshapes](https://github.com/darkshapes/)]
<hr><br>

[![Python application](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml/badge.svg)](https://github.com/darkshapes/nnll/actions/workflows/python-app.yml)<br>
![GitHub repo size](https://img.shields.io/github/repo-size/darkshapes/nnll)<br>
![Discord](https://img.shields.io/discord/1266757128249675867)<hr>

## terminal commands

> [!NOTE] these commands you will want to run from the root folder<br>
`nnll-contents`- Recreate the modules table of contents, populating preview links for IDEs.<br>
`nnll-parse`   - Lookup model metadata headers and write out to .json file.<br>
`nnll-find`    - Scan models for a specific string pattern within their state dict layer contents.<br>
<!--/`nnll-tag`- Manually construct a tag for a model
`nnll-index`- Identify models from specific files or folders and build an index of their tags and metadata.
 --> <br>
##

## importing modules

<!-- `from nnll import folder`  -->
<h2>[modules table of contents](https://github.com/darkshapes/nnll/blob/main/modules/toc.md)</h2><details><summary>

<hr>

## development setup

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
<br><br><br>
