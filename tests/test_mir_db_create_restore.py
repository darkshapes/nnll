# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from pathlib import Path
from nnll.mir.json_cache import MIR_PATH_NAMED


def test_mir_creation():
    from nnll.mir.mir import mir_entry
    from pprint import pprint

    os.remove(MIR_PATH_NAMED)
    Path().touch()

    entry = mir_entry(
        domain="info",
        arch="unet",
        series="stable-diffusion-xl",
        comp="base",
        repo="stabilityai/stable-diffusion-xl",
        pkg={
            0: {
                "diffusers": "class_name",
                "generation": {"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False},
            }
        },
    )
    entry.update(
        mir_entry(
            domain="model",
            arch="unet",
            series="stable-diffusion-xl",
            comp="base",
            file_path="/Users/nyan/Documents/models",
        ),
    )
    entry.update(
        mir_entry(
            domain="ops",
            arch="scheduler",
            series="align-your-steps",
            comp="stable-diffusion-xl",
            pkg={
                0: {
                    "diffusers.schedulers.scheduling_utils": {
                        "AysSchedules": {"num_inference_steps": 10, "timesteps": "StableDiffusionXLTimesteps"},
                    }
                }
            },
        )
    )
    entry.update(
        mir_entry(
            domain="ops",
            arch="patch",
            series="hidiffusion",
            comp="stable-diffusion-xl",
            pkg={0: {"hidiffusion": {"apply_hidiffusion": {"generation": {"height": 2048, "width": 2048, "eta": 1.0, "guidance_scale": 7.5}}}}},
        )
    )
    pprint(entry)


def test_mir_maid():
    import json
    import os
    from nnll.mir.mir import mir_entry
    from nnll.integrity import ensure_path

    entry = mir_entry(
        domain="info",
        arch="unet",
        series="stable-diffusion-xl",
        comp="base",
        repo="stabilityai/stable-diffusion-xl",
        pkg={
            0: {
                "diffusers": "class_name",
                "generation": {"num_inference_steps": 40, "denoising_end": 0.8, "output_type": "latent", "safety_checker": False},
            }
        },
    )
    try:
        os.remove(MIR_PATH_NAMED)
    except FileNotFoundError:
        pass
    folder_path_named = ensure_path(os.path.dirname(MIR_PATH_NAMED), os.path.basename(MIR_PATH_NAMED))
    folder_path_named = os.path.dirname(folder_path_named)
    from nnll.mir.maid import MIRDatabase

    mir_db = MIRDatabase()
    mir_db.add(entry)
    mir_db.write_to_disk()
    print(mir_db.database)
    with open(MIR_PATH_NAMED, "r", encoding="UTF-8") as f:
        result = json.load(f)
    expected = {
        "info.unet.stable-diffusion-xl": {
            "base": {
                "pkg": {
                    "0": {
                        "diffusers": "class_name",
                        "generation": {
                            "denoising_end": 0.8,
                            "num_inference_steps": 40,
                            "output_type": "latent",
                            "safety_checker": False,
                        },
                    },
                },
                "repo": "stabilityai/stable-diffusion-xl",
            },
        },
    }

    assert mir_db.database == expected
    assert result == expected


def test_restore_mir():
    import json
    import os

    from nnll.integrity import ensure_path
    from nnll.metadata.json_io import write_json_file
    from nnll.mir.json_cache import MIR_PATH_NAMED
    from nnll.mir.maid import MIRDatabase, main

    database = {"expecting": "data"}
    try:
        os.remove(MIR_PATH_NAMED)
    except FileNotFoundError:
        pass
    folder_path_named = ensure_path(os.path.dirname(MIR_PATH_NAMED), os.path.basename(MIR_PATH_NAMED))
    folder_path_named = os.path.dirname(folder_path_named)
    write_json_file(folder_path_named, file_name="mir.json", data=database, mode="w")
    database.pop("expecting", {})
    mir_db = MIRDatabase()
    mir_db.database.pop("empty", {})
    main(mir_db)
    with open(MIR_PATH_NAMED, "r", encoding="UTF-8") as f:
        result = json.load(f)
    mir_db = MIRDatabase()
    expected = mir_db.database
    for tag, compatibility in result.items():
        for comp, field in compatibility.items():
            for header, definition in field.items():
                if isinstance(definition, dict):
                    for key in definition:
                        if len(key) > 1:
                            assert field[header][key] == expected[tag][comp][header][key]
                        # else:
                        # assert field[header][key] == expected[tag][comp][header][key]
                else:
                    assert field[header] == expected[tag][comp][header]

    print(mir_db.database)


if __name__ == "__main__":
    test_mir_creation()
