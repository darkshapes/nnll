# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pytest


@pytest.fixture
def mock_test_database():
    from nnll.mir.maid import MIRDatabase, main

    mir_db = MIRDatabase()
    main(mir_db)
    return mir_db


# @pytest.fixture
# def mock_test_database():
#     from nnll.mir.maid import MIRDatabase

#     mir_db = MIRDatabase()
#     return mir_db


def test_grade_maybes_fail(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="table-cascade")
    assert result is None


def test_grade_similar_fail_again(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="able-cascade-")
    assert result is None


def test_grade_cascade_decoder_match(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="stabilityai/stable-cascade")
    assert result == ["info.unet.stable-cascade", "decoder"]


def test_grade_cascade_match(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="stabilityai/stable-cascade")
    assert result == ["info.unet.stable-cascade", "decoder"]


def test_grade_field_change(mock_test_database):
    result = mock_test_database.find_path(field="pkg", sub_field=0, target="parler_tts")
    assert result == ["info.art.parler-tts-v1", "*"]


def test_grade_letter_case_change(mock_test_database):
    result = mock_test_database.find_path(field="pkg", target="AuDiOCrAfT")
    assert result == ["info.art.audiogen", "*"]


def test_repo_case_change(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="outeAI/OuteTTS-0.3-1b")
    assert result == ["info.art.outetts-0", "*"]


def test_sub_module_detection(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="PixArt-alpha/PixArt-Sigma-XL-2-1024-Ms")
    assert result == ["info.dit.pixart-sigma-xl-2-1024-ms", "*"]


def test_find_path_truncated(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="UsefulSenso")
    assert result is None


def test_find_path_truncated_2(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="UsefulSensors")
    assert result is None


def test_find_path_truncated_4(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="UsefulSensors/moon")
    print(result)
    assert result is None


def test_find_path_decent(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="UsefulSensors/moonshine")
    print(result)
    assert result == ["info.stst.moonshine", "*"]


def test_find_path_truncated_6(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="UsefulSensors/moonshine-")
    assert result == ["info.stst.moonshine", "*"]


def test_find_qwen(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="Qwen/Qwen2-VL-7B-Instruct")
    assert result == ["info.vit.qwen2-vl", "*"]


# def test_find_qwen_32(mock_test_database):
#     result = mock_test_database.find_path(field="repo", target="Qwen/Qwen2-VL-nstruct".lower())
#     assert result == ["info.vit.qwen2-vl", "*"]


7
