---
language:
  - en
library_name: nnll
license_name: MPL-2.0 + Commons Clause 1.0
---

<div align="center">

![nnll75_transparent](https://raw.githubusercontent.com/darkshapes/entity-statement/refs/heads/main/png/nnll/nnll150_transparent.png)</div>

# nnll <br><sub>neural network link library: Flexible code for multimodal AI apps</sub>

`nnll` (or <em>null</em>)is a toolkit for unified inference pipelines. It provides modular, reusable, and efficient components as a foundation to simplify the process of building and managing complex generative systems.

- Generative AI pipeline preparation & execution
- Writing metadata from images
- Consumer-grade GPU/CPU inference optimization
- Misc UX/UI Experimentation
- 🧨Diffusers, 🤗Transformers
- :shipit: <br><br>

[![Python application test status](https://github.com/darkshapes/nnll/actions/workflows/nnll.yml/badge.svg)](nnll/actions/workflows/nnll.yml) <br>
![commits per month](https://img.shields.io/github/commit-activity/m/darkshapes/nnll?color=indigo)<br>
![code size](https://img.shields.io/github/languages/code-size/darkshapes/nnll?color=navy)<br>
[<img src="https://img.shields.io/discord/1266757128249675867?color=5865F2">](https://discord.gg/VVn9Ku74Dk)<br>
[<img src="https://img.shields.io/badge/me-__?logo=kofi&logoColor=white&logoSize=auto&label=feed&labelColor=maroon&color=grey&link=https%3A%2F%2Fko-fi.com%2Fdarkshapes">](https://ko-fi.com/darkshapes)<br>
<br>

## Quick Guide

### Install

Install [uv](https://github.com/astral-sh/uv#installation), then run these terminal commands

- > ```
  > git clone https://github.com/darkshapes/nnll
  > cd nnll
  > uv sync --group dev
  > ```

### Use

Enter a terminal and activate the python environment in

- > Linux/Macos:
  >
  > ```
  > source .venv/bin/activate
  > ```

  > Windows Powershell:
  >
  > ```
  > Set-ExecutionPolicy Bypass -Scope Process -Force; .venv\Scripts\Activate.ps1
  > ```

nnll stores dependency versions in branches. In order to prevent failures, the repository should _NEVER_ be rebased.

> [!IMPORTANT]
>
> ## Classes, Methods & Constants :
>
> ```
> ReadModelTag
>           `---------------------------------Universal model tag reader
> ExtensionType
>           `---------------------------------Model extension constants
> JSONCache
>        `------------------------------------Json read operations
> ```
>
> ## Available terminal commands:<br>
>
> - <A href="#nnll-autocard">`nnll-autocard`</a>
> - <A href="#nnll-info">`nnll-info`</a>

#### nnll-autocard

```
usage: nnll-autocard black-forest-labs/FLUX.1-Krea-dev -u exdysa -f FLUX.1-Krea-dev-MLX -l mlx -q 8

Create a new HuggingFace RepoCard.

    Retrieve HuggingFace repository data, fill out missing metadata,create a model card.
    Optionally download and quantize repo to a desired folder that will be ready for upload.
    Online function.

positional arguments:
  repo                  Relative path to HF repository

options:
  -h, --help            show this help message and exit
  -l, --library {gguf,schnell,dev,mlx}
                        Output model type [gguf,mlx,dev,schnell] (optional, default: 'mlx') NOTE: dev/schnell use MFLUX.
  -q, --quantization {8,6,4,3,2}
                        Set quantization level (optional, default: None)
  -d, --dry_run         Perform a dry run, reading and generating a repo card without converting the model (optional, default: False)
  -u, --user USER       User for generated repo card (optional)
  -f, --folder FOLDER   Folder path for downloading (optional, default: /Users/unauthorized/Downloads)
  -p, --prompt PROMPT   A prompt for the code example (optional, default: 'Test Prompt')

**Valid pipeline tags**:

         text-classification, token-classification, table-question-answering, question-answering, zero-shot-classification, translation, summarization, feature-extraction, text-generation, text2text-generation, fill-mask, sentence-similarity, text-to-speech, text-to-audio, automatic-speech-recognition, audio-to-audio, audio-classification, audio-text-to-text, voice-activity-detection, depth-estimation, image-classification, object-detection, image-segmentation, text-to-image, image-to-text, image-to-image, image-to-video, unconditional-image-generation, video-classification, reinforcement-learning, robotics, tabular-classification, tabular-regression, tabular-to-text, table-to-text, multiple-choice, text-ranking, text-retrieval, time-series-forecasting, text-to-video, image-text-to-text, visual-question-answering, document-question-answering, zero-shot-image-classification, graph-ml, mask-generation, zero-shot-object-detection, text-to-3d, image-to-3d, image-feature-extraction, video-text-to-text, keypoint-detection, visual-document-retrieval, any-to-any, other
```

[A link to example output of the `nnll-autocard` command](https://huggingface.co/exdysa/shuttle-3.1-aesthetic-MLX-Q8/blob/main/README.md)

#### nnll-info

Immediate diagnostic system status information

```
usage: nnll-info
```

### [Detailed instructions for all version branches:](https://github.com/darkshapes/sdbx/wiki/Develop)

Discussion topics, issue requests, reviews, and code updates are encouraged. Build with us! Talk to us in [our Discord](https://discord.gg/VVn9Ku74Dk)!

<br>
<!--
![Alt](https://repobeats.axiom.co/api/embed/13fd2c53953a777ae8583f620fa8bd014baadef1.svg "Repobeats analytics image") -->
