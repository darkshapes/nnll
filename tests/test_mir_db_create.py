# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import os
from pathlib import Path
from nnll.mir.json_cache import MIR_PATH_NAMED


os.remove(MIR_PATH_NAMED)
Path(MIR_PATH_NAMED).touch()


def test_mir_creation():
    from nnll.mir.mir import mir_entry
    from pprint import pprint

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


if __name__ == "__main__":
    test_mir_creation()
