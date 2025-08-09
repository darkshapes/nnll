---
language:
- en
library_name: nnll
license_name: MPL-2.0 + Commons Clause 1.0
---

---
language:
- en
library_name: nnll
license_name: MPL-2.0 + Commons Clause 1.0
---

<div align="center">

![nnll75_transparent](https://github.com/user-attachments/assets/de8c1a49-4695-4c4b-b7c4-29fba483a65d)</div>
# nnll <br><sub>neural network link library: Flexible code for multimodal AI apps</sub>

`nnll` (or <em>null</em>)is a toolkit for researchers and developers working with AI models like Diffusion and Large Language Models (LLMs).  It provides modular, reusable, and efficient components as a foundation to simplify the process of building and managing these complex systems.

* Generative AI pipeline preparation & execution
* Extracting and classifying metadata from images/models
* Consumer-grade GPU/CPU inference optimization
* Misc UX/UI Experimentation
* 🧨Diffusers, 🤗Transformers, 🦙Ollama, 🍏MLX, 🌀DSPy, 🚅LiteLLM
* :shipit: <br><br>

[![Python application test status](https://github.com/darkshapes/nnll/actions/workflows/nnll.yml/badge.svg)](nnll/actions/workflows/nnll.yml) <br>
![commits per month](https://img.shields.io/github/commit-activity/m/darkshapes/nnll?color=indigo)<br>
![code size](https://img.shields.io/github/languages/code-size/darkshapes/nnll?color=navy)<br>
[<img src="https://img.shields.io/discord/1266757128249675867?color=5865F2">](https://discord.gg/VVn9Ku74Dk)<br>
[<img src="https://img.shields.io/badge/me-__?logo=kofi&logoColor=white&logoSize=auto&label=feed&labelColor=maroon&color=grey&link=https%3A%2F%2Fko-fi.com%2Fdarkshapes">](https://ko-fi.com/darkshapes)<br>
<br>

## Quick Guide

### Install

Install [uv](https://github.com/astral-sh/uv#installation), then run these terminal commands
- >
  >```
  > git clone https://github.com/darkshapes/nnll
  > cd nnll
  > uv sync --group dev
  > ```

### Use

Enter a terminal and activate the python environment in
- >
  > Linux/Macos:
  > ```
  > source .venv/bin/activate
  > ```

  > Windows Powershell:
  > ```
  > Set-ExecutionPolicy Bypass -Scope Process -Force; .venv\Scripts\Activate.ps1
  > ```

Available terminal commands:<br>
- <A href="#mir-add">`mir-add`</a>
- <A href="#mir-maid">`mir-maid`</a>
- <A href="#mir-tasks">`mir-tasks`</a>
- <A href="#mir-pipe">`mir-pipe`</a>
- <A href="#nnll-autocard">`nnll-autocard`</a>
- <A href="#nnll-autohash">`nnll-autohash`</a>
- <A href="#nnll-hash">`nnll-hash`</a>
- <A href="#nnll-layer">`nnll-layer`</a>
- <A href="#nnll-meta">`nnll-meta`</a>

#### mir-add
```
usage: mir-add --domain info --arch lora --series slam --compatibility sd1_series \
        -k {'repo':'alimama-creative/slam-sd1.5', 'pkg':{0: {'diffusers': 'load_lora_weights'}}}

Manually add entries to MIR database.
Offline function.

options:
  -h, --help            show this help message and exit
  -d, --domain DOMAIN   Broad name of the type of data (model/ops/info/dev)
  -a, --arch ARCH       Common name of the neural network structure being referenced
  -s, --series SERIES   Specific release title or technique
  -c, --compatibility COMPATIBILITY
                        Details about purpose, tasks
  -k, --kwargs KWARGS   Keyword arguments to pass to function constructors (default: NOne)

MIR Class attributes:
         Domain: ['dev', 'info', 'model', 'ops']
         Ops: ['pkg', 'repo']
         Info: ['repo', 'pkg', 'file_256', 'layer_256', 'file_b3', 'layer_b3', 'identifier']
         Dev: ['stage', 'dtype', 'dep_pkg', 'gen_kwargs', 'lora_kwargs', 'module_alt', 'module_path', 'repo_pkg', 'requires', 'scheduler_alt', 'scheduler_kwargs_alt', 'scheduler_kwargs', 'scheduler', 'tasks', 'weight_map']
```
[A link to example output of the `mir-add` command](nnll/mir/config/mir.json)

#### mir-maid
```
usage: mir-maid

Build a custom MIR model database from the currently installed system environment.
Offline function.

options:
  -h, --help        show this help message and exit
  -r, --remake_off  Don't erase and remake the MIR database (default: False)

Includes `mir-task` and `mir-pipe` if ran using `python -m nnll.mir.maid` . Output:
            2025-08-03 14:22:47 INFO     ('Available torch devices: mps',)
            2025-08-03 14:22:47 INFO     ('Wrote #### lines to MIR database file.',)
```
[A link to example output of the `mir-maid` command](nnll/mir/config/mir.json)

#### mir-tasks
```
usage: mir-tasks

Scrape the task classes from currently installed libraries and attach them to an existing MIR database.
Offline function.

options:
  -h, --help  show this help message and exit

Can be run automatically with 'mir-maid' Should only be used after `mir-maid`.

Output:
    INFO     ('Wrote #### lines to MIR database file.',)
```
[A link to example output of the `mir-tasks` command](nnll/mir/config/mir.json)

#### mir-pipe

```
usage: mir-pipe

Infer pipe components from Diffusers library and attach them to an existing MIR database.
Offline function.

options:
  -h, --help  show this help message and exit

Can be run automatically with 'mir-maid' Should only be used after `mir-maid`.

Output:
    INFO     ('Wrote #### lines to MIR database file.',)
```
[A link to example output of the `mir-pipe` command](nnll/mir/config/mir.json)

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

#### nnll-autohash
```
usage: nnll-autohash [-h] repo

Generate hashes for files or state dicts located remotely or in a cached repo

positional arguments:
  repo        Relative path to repository

options:
  -h, --help  show this help message and exit
```
[A link to example output of the `nnll-autohash` command](https://huggingface.co/darkshapes/MIR/blob/main/layer_hash/b3_layerPhi-4-mini-instruct_20250727154025.json)

#### nnll-hash
```
usage: nnll-hash '~/Downloads/models/'

Output hashes of each model file state dict in [path] to console and .JSON
 Offline function.

positional arguments:
  path                  Path to the directory where files should be analyzed. (default '.'')

options:
  -h, --help            show this help message and exit
  -f, --file            Change mode to calculate hash for the whole file instead of state dict layers (default: False)
  -s, --sha             Change algorithm from BLAKE3 to SHA256 (default: False)
  -d, --describe, --describe-process
                        Include processing metadata in the output (default: True)
  -u, --unsafe          Try to hash non-standard type model files. MAY INCLUDE NON-MODEL FILES. (default: False)
```
[A link to an output of the `nnll-hash` command](https://huggingface.co/darkshapes/MIR/blob/main/layer_hash/b3_layer_0725.json)

#### nnll-layer
```
usage: nnll-layer adaln

Recursively search for layer name metadata in state dict .JSON files of the current folder.
Print filenames with matching layers to console along with the first matching layer's corresponding shape, and tensor counts.
Offline function.

positional arguments:
  pattern     Pattern to search for

options:
  -h, --help  show this help message and exit

Output:
2025-08-03 14:57:10 INFO     ('./Pixart-Sigma-XL-2-2k-ms.diffusers.safetensors.json', {'shape': [1152], 'tensors': 604})                             console.py:84
                    INFO     ('./PixartXL-2-1024-ms.diffusers.safetensors.json', {'shape': [384], 'tensors': 613})                                   console.py:84
                    INFO     ('./flash-pixart-a.safetensors.json', {'shape': [64, 256], 'tensors': 587})
```

#### nnll-meta
```
usage: nnll-meta ~/Downloads/models/images -s ~Downloads/models/metadata

Scan the state dict metadata from a folder of files at [path] to the console, then write to a json file at [save]
Offline function.

positional arguments:
  path                  Path to directory where files should be analyzed. (default .)

options:
  -h, --help            show this help message and exit
  -s, --save_to_folder_path SAVE_TO_FOLDER_PATH
                        Path where output should be stored. (default: '.')
  -d, --separate_desc   Ignore the metadata from the header. (default: False)
  -u, --unsafe          Try to read non-standard type model files. MAY INCLUDE NON-MODEL FILES. (default: False)

Valid input formats: ['.pth', '.ckpt', '.onnx', '.pt', '.pickletensor', '.safetensors', '.sft', '.gguf']
```
[A link to example output of the `nnll-meta` command](https://huggingface.co/darkshapes/MIR/blob/main/layer_data/chroma-unlocked-v46.safetensors.json)

### [Detailed instructions :](https://github.com/darkshapes/sdbx/wiki/Develop)

Discussion topics, issue requests, reviews, and code updates are encouraged. Build with us! Talk to us in [our Discord](https://discord.gg/VVn9Ku74Dk)!

<br>
<!--
![Alt](https://repobeats.axiom.co/api/embed/13fd2c53953a777ae8583f620fa8bd014baadef1.svg "Repobeats analytics image") -->