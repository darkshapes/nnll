# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""為全人類製作模型卡！！"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from nnll.integrity import ensure_path
from nnll.metadata.helpers import ask_multi_input
from nnll.monitor.console import nfo
from nnll.monitor.file import dbug


def index_model_card(repo_path) -> Optional[Dict[str, Any]]:
    """Fetch repo modelcard\n
    :param repo_path: Path to repo for modelcard
    :return: The model card tags as dictionary
    """
    from huggingface_hub import constants, repocard
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    from requests import HTTPError

    constants.HF_HUB_OFFLINE = 0
    constants.HF_XET_HIGH_PERFORMANCE = 1
    constants.HF_HUB_ENABLE_HF_TRANSFER = 1
    try:
        model_metadata = repocard.RepoCard.load(repo_path).data
        nfo(f"Metadata acquired from {repo_path}")

    except (EntryNotFoundError, LocalEntryNotFoundError, HTTPError, ValueError) as error_log:
        dbug(error_log)
        return None
    return model_metadata


def write_card(folder_path, model_card):
    """card"""

    readme = Path(os.path.join(folder_path, "README.md"))
    if os.path.exists(readme):
        os.remove(readme)
    readme.touch(exist_ok=False)
    with open(readme, "w", encoding="UTF-8") as md_file:
        md_file.write(model_card)
        return readme
    return None


def generate_model_card(conditions: tuple, full_tags: dict) -> None:
    """Assemble the gathered metadata for the card\n
    :param conditions: Generation details
    :param full_tags: The repocard with user-augmented fields
    :param dry_run: Whether or not to generate model card only or run conversion
    """
    tag_block = ["---"]
    model_field = [f"name: {conditions['model_name']}"]
    link_block = ["> [!IMPORTANT]", f"> Original Model Link : [https://huggingface.co/{conditions['repo']}](https://huggingface.co/{conditions['repo']})", "> \n"]
    code_block = ["```"]
    content = []
    for tag, value in full_tags.items():
        if value is not None:
            if isinstance(value, list) and len(value) > 1:
                new_value = []
                for data in value:
                    new_value.append(f" - {data}")
                new_value.insert(0, f"{tag} :")
                value = new_value
            elif isinstance(value, str):
                value = [f"{tag}: {value}"]
            else:
                value = [f"{tag}: {value[0]}"]
            content.extend(value)
    blurb = ask_multi_input(tag="a short headline")
    mlx_blurb = [
        f"""

# {conditions["model_name"]}
{conditions["model_name"]} {next(iter(blurb), "")}

> [!WARNING]
> MLX is a framework for METAL graphics supported by Apple computers with ARM M-series processors (M1/M2/M3/M4)

> [!NOTE]
> Generation using uv https://docs.astral.sh/uv/**:
> ```
> uv{conditions["example"]}
>```

> [!NOTE]
> Generation using pip:
> ```
> pip{conditions["example"]}
> ```
    """
    ]
    model_card_elements = tag_block + model_field + content + tag_block + link_block + code_block + model_field + content + code_block + mlx_blurb
    model_card = "\n".join(model_card_elements)
    validate = input(f"""Generated_card:\n\n{model_card}\n\n  saving to {conditions["folder_path_named"]} Correct? (y and enter to submit, any other entry and enter to quit) """)

    if "y" in validate.lower() or "\n" in validate.lower():
        return model_card
    return


def autocard(conditions: tuple) -> None:  # pylint:disable=redefined-outer-name
    """Generate a model card then convert the downloaded repo\n
    :param conditions: Generation details
    """

    required_tags = {
        "base_model": None,
        "license": None,
        "pipeline_tag": None,
        "tasks": None,
        "language": None,
    }
    optional_tags = {
        "license_link": None,
        "datasets": None,
        "paper_url": None,
        "funded_by": None,
        "hardware_type": None,
        "cloud_region": None,
        "cloud_provider": None,
        "hours_used": None,
        "co2_emitted": None,
        "thumbnail": None,
    }
    if not conditions["library"] == "gguf":
        optional_tags.setdefault(
            "get_started_code",
            f"uv{conditions['example']} --prompt 'Test prompt'",
        )
    model_metadata = index_model_card(conditions["repo"])
    full_tags = required_tags | optional_tags
    for tag, _ in full_tags.items():
        if hasattr(model_metadata, tag):
            full_tags[tag] = getattr(model_metadata, tag)
        if not full_tags[tag]:
            full_tags[tag] = ask_multi_input(tag, required=tag in required_tags)
    return generate_model_card(conditions, full_tags=full_tags)


def main():
    """
    Build model cards for all humanity!!
    全人類のためのモデルカードを作る!!
    """
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Create a new HuggingFace RepoCard.\n
    Retrieve HuggingFace repository data, fill out missing metadata,create a model card.
    Optionally download and quantize repo to a desired folder that will be ready for upload.
    Online function. """,
        usage="nnll-autocard black-forest-labs/FLUX.1-Krea-dev -u exdysa -f FLUX.1-Krea-dev-MLX -l mlx -q 8",
        epilog="""**Valid pipeline tags**:

         text-classification, token-classification, table-question-answering, question-answering, zero-shot-classification, translation, summarization, feature-extraction, text-generation, text2text-generation, fill-mask, sentence-similarity, text-to-speech, text-to-audio, automatic-speech-recognition, audio-to-audio, audio-classification, audio-text-to-text, voice-activity-detection, depth-estimation, image-classification, object-detection, image-segmentation, text-to-image, image-to-text, image-to-image, image-to-video, unconditional-image-generation, video-classification, reinforcement-learning, robotics, tabular-classification, tabular-regression, tabular-to-text, table-to-text, multiple-choice, text-ranking, text-retrieval, time-series-forecasting, text-to-video, image-text-to-text, visual-question-answering, document-question-answering, zero-shot-image-classification, graph-ml, mask-generation, zero-shot-object-detection, text-to-3d, image-to-3d, image-feature-extraction, video-text-to-text, keypoint-detection, visual-document-retrieval, any-to-any, other""",
    )
    example = {
        "gguf": "",
        "schnell": "",
        "dev": "",
        "mlx": "",
    }
    parser.add_argument("repo", type=str, help="Relative path to HF repository")
    parser.add_argument("-l", "--library", type=str, choices=list(example), default="mlx", help="Output model type [gguf,mlx,dev,schnell] (optional, default: 'mlx') NOTE: dev/schnell use MFLUX.")
    parser.add_argument("-q", "--quantization", type=int, choices=[8, 6, 4, 3, 2], required=False, help="Set quantization level (optional, default: None)")
    parser.add_argument("-d", "--dry_run", action="store_true", help="Perform a dry run, reading and generating a repo card without converting the model (optional, default: False)")
    parser.add_argument("-u", "--user", type=str, default="darkshapes", help="User for generated repo card (optional)")
    parser.add_argument("-f", "--folder", type=str, default=os.path.join(str(Path.home()), "Downloads"), help=f"Folder path for downloading (optional, default: {os.path.join(str(Path.home()), 'Downloads')})")
    parser.add_argument("-p", "--prompt", type=str, default="Test Prompt", help="A prompt for the code example (optional, default: 'Test Prompt')")

    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    conditions = {
        "repo": args.repo,
        "library": args.library,
        "model_name": os.path.basename(f"{args.repo}-{'GGUF' if args.library == 'gguf' else 'MLX'}"),
        "quantization": args.quantization,
    }
    if args.quantization:
        conditions["model_name"] += f"-Q{conditions['quantization']}"
    flux_arguments = "--seed 10 --width 1024 --height 1024"
    if conditions["quantization"]:
        flux_arguments += f" -q {conditions['quantization']}"
    universal_arguments = f" --model {args.user}/{conditions['model_name']} --prompt '{args.prompt}'"
    example.update({"schnell": f"x --from mflux mflux-generate {universal_arguments} --base-model schnell --steps 4 {flux_arguments} "})
    example.update({"dev": f"x --from mflux mflux-generate {universal_arguments} --base-model dev --steps 50 --guidance 4.0 {flux_arguments} "})
    example.update({"mlx": f"x --from mlx-lm mlx_lm.generate --model {universal_arguments}"})
    conditions.setdefault("example", example.get(args.library))
    conditions.setdefault("folder_path_named", os.path.join(args.folder, conditions["model_name"]))

    nfo(
        f"""Processing repo: {conditions["repo"]} to {conditions["folder_path_named"]} : extra arguments (
        {conditions["library"]}
        {conditions["quantization"]}
        {conditions["example"]}"""
    )
    folder_path_named = None
    readme = write_card(".", autocard(conditions=conditions))
    if readme and not args.dry_run:
        from nnll.tensor_pipe.autoquant import convert_repo

        folder_path_named = convert_repo(conditions)
    elif readme:
        folder_path_named = ensure_path(conditions["folder_path_named"])
    if folder_path_named:
        shutil.move(readme, os.path.join(folder_path_named, "README.md"))


if __name__ == "__main__":
    main()
