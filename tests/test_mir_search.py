# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pytest


@pytest.fixture
def mock_test_database():
    from nnll.mir.maid import MIRDatabase  # , main

    mir_db = MIRDatabase()
    # main(mir_db)
    return mir_db


def test_grade_maybes_fail(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="table-cascade")
    assert result is None


def test_grade_similar_fail_again(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="able-cascade-")
    assert result is None


def test_grade_cascade_decoder_match(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="stabilityai/stable-cascade")
    assert result == ["info.unet.stable-cascade", "decoder"]


def test_grade_cascade_match(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="stabilityai/stable-cascade", domain="info.unet")
    assert result == ["info.unet.stable-cascade", "decoder"]


def test_grade_field_change(mock_test_database):
    result = mock_test_database.find_tag(field="pkg", target="parler_tts", domain="info.")
    assert result == ["info.art.parler-tts-v1", "*"]


def test_grade_letter_case_change(mock_test_database):
    result = mock_test_database.find_tag(field="pkg", target="AuDiOCrAfT")
    assert result == ["info.art.audiogen", "*"]


def test_repo_case_change(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="outeAI/OuteTTS-0.3-1b")
    assert result == ["info.art.outetts-0", "*"]


def test_sub_module_detection(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="PixArt-alpha/PixArt-Sigma-XL-2-1024-Ms")
    assert result == ["info.dit.pixart-sigma-xl-2-1024-ms", "*"]


def test_find_tag_truncated(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="UsefulSenso")
    assert result is None


def test_find_tag_truncated_2(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="UsefulSensors")
    assert result is None


def test_find_tag_truncated_4(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="UsefulSensors/moon")
    assert result is None


def test_find_tag_decent(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="UsefulSensors/moonshine")
    assert result == ["info.stst.moonshine", "*"]


def test_find_tag_truncated_6(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="UsefulSensors/moonshine-")
    assert result == ["info.stst.moonshine", "*"]


def test_find_qwen_2_vl(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="Qwen/Qwen2-VL-7B-Instruct", domain="info.vit")
    assert result == ["info.vit.qwen2-vl", "*"]


def test_find_qwen_2_vl_2(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="Qwen/Qwen2-VL-Instruct".lower(), domain="info.vit")
    assert result == ["info.vit.qwen2-vl", "*"]


def test_grade_similar_fail_again(mock_test_database):
    result = mock_test_database.find_tag(field="task", target="UMT5EncoderModel")
    assert result is None


def test_find_gpt_oss(mock_test_database):
    result = mock_test_database.find_tag(field="repo", target="openai/gpt-oss-120b".lower(), domain="info.moe")
    assert result == ["info.moe.gpt-oss", "*"]
