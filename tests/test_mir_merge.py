# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# test_merge_data.py
import pytest

from nnll.mir.automata import assimilate


class MIRDatabase:
    def __init__(self):
        self.database = {
            "info.unet.stable-diffusion-xl": {
                "base": {
                    "repo": "stabilityai/stable-diffusion-xl-base-1.0",
                    "pkg": {0: {"diffusers": "StableDiffusionXLPipeline"}},
                    "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
                },
            }
        }


def test_merge_data_simple_case():
    mir_db = MIRDatabase()
    mir_db.database["arch1.series1"] = {"component1": {}}

    data_tuple = [("arch1", "series1", {"component1": {"field1": {"key1": "value1"}}})]

    assimilate(mir_db, data_tuple)
    assert mir_db.database["arch1.series1"]["component1"]["field1"]["key1"] == "value1"


# Test case
@pytest.fixture
def mock_mir_db():
    return MIRDatabase()


def test_merge_data(mock_mir_db):
    """TEST DATAAAAA 測試資料
    Call the function to test & Check if the data was merged correctly"""
    from pprint import pprint

    data_tuple = [
        (
            "info.unet",
            "stable-diffusion-xl",
            {
                "base": {
                    "pkg": {
                        0: {
                            "generation": {
                                "denoising_end": 0.8,
                                "output_type": "latent",
                                "safety_checker": False,
                                "width": 1024,
                                "height": 1024,
                            },
                        },
                        1: {"diffusers": "DiffusionPipeline"},
                    },
                    "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
                }
            },
        ),
    ]

    assimilate(mock_mir_db, data_tuple)
    expected_result = {
        "base": {
            "repo": "stabilityai/stable-diffusion-xl-base-1.0",
            "pkg": {
                0: {
                    "diffusers": "StableDiffusionXLPipeline",
                    "generation": {
                        "denoising_end": 0.8,
                        "output_type": "latent",
                        "safety_checker": False,
                        "width": 1024,
                        "height": 1024,
                    },
                },
                1: {"diffusers": "DiffusionPipeline"},
            },
            "layer_256": ["62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef"],
        }
    }
    pprint(mock_mir_db.database)
    assert mock_mir_db.database["info.unet.stable-diffusion-xl"] == expected_result


def test_merge_data_nested_case():
    mir_db = MIRDatabase()
    mir_db.database = {"arch2.series2": {"base": {"pkg": {0: {"module": {}}}}}}
    print(mir_db.database)
    assert mir_db.database["arch2.series2"]["base"]["pkg"][0] == {"module": {}}
    data_tuple = [("arch2", "series2", {"base": {"pkg": {0: {"extra": {"x": {"key2": "value2"}}}}}})]
    assimilate(mir_db, data_tuple)
    print(mir_db.database)

    assert mir_db.database["arch2.series2"]["base"]["pkg"][0]["module"] == {}
    assert mir_db.database["arch2.series2"]["base"]["pkg"][0]["extra"] == {"x": {"key2": "value2"}}


def test_merge_data_multiple_levels():
    mir_db = MIRDatabase()
    mir_db.database["arch3.series3"] = {"component3": {"field3": {"definition3": {"sub_def3": {}}}}}

    data_tuple = [("arch3", "series3", {"component3": {"field3": {"definition3": {"sub_def3": {"key3": "value3"}}}}})]

    assimilate(mir_db, data_tuple)
    assert mir_db.database["arch3.series3"]["component3"]["field3"]["definition3"]["sub_def3"]["key3"] == "value3"


def test_merge_data_type_error():
    mir_db = MIRDatabase()
    mir_db.database["arch4.series4"] = {"component4": {}}

    data_tuple = [("arch4", "series4", {"component4": "not a dict"})]

    with pytest.raises(TypeError):
        assimilate(mir_db, data_tuple)
