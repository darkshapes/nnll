# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pytest


def test_mir_maid():
    import json

    from nnll.mir.json_cache import MIR_PATH_NAMED  #
    from nnll.mir.maid import MIRDatabase

    expected = {"empty": "101010101010101010"}
    mir_db = MIRDatabase()
    mir_db.database = expected
    mir_db.write_to_disk()
    with open(MIR_PATH_NAMED, "r", encoding="UTF-8") as f:
        result = json.load(f)

    assert result == expected


def test_restore_mir():
    import json

    from nnll.mir.json_cache import MIR_PATH_NAMED
    from nnll.mir.maid import MIRDatabase, main

    mir_db = MIRDatabase()
    mir_db.database.pop("empty")
    main(mir_db)
    expected = mir_db.database
    with open(MIR_PATH_NAMED, "r", encoding="UTF-8") as f:
        result = json.load(f)
    for tag, compatibility in result.items():
        for comp, field in compatibility.items():
            for header, definition in field.items():
                if isinstance(definition, dict):
                    for key in definition:
                        if len(key) > 1:
                            assert field[header][key] == expected[tag][comp][header][key]
                        else:
                            assert field[header][key] == expected[tag][comp][header][int(key)]
                else:
                    assert field[header] == expected[tag][comp][header]

    print(mir_db.database)


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


def test_grade_cascade_prior_match(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="stabilityai/stable-cascade-prior")
    assert result == ["info.unet.stable-cascade-prior", "*"]


def test_grade_cascade_match(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="stabilityai/stable-cascade")
    assert result == ["info.unet.stable-cascade", "decoder"]


def test_grade_field_change(mock_test_database):
    result = mock_test_database.find_path(field="pkg", target="parler_tts")
    assert result == ["info.artm.parler-tts", "tiny-v1"]


def test_grade_letter_case_change(mock_test_database):
    result = mock_test_database.find_path(field="pkg", sub_field=0, target="AuDiOCrAfT.MoDeLs")
    assert result == ["info.artm.audiogen", "medium-1-5b"]


def test_repo_case_change(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="outeAI/OuteTTS-0.3-1b")
    assert result == ["info.artm.outetts-0-3", "1b"]


def test_sub_module_detection(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="PixArt-alpha/PixArt-Sigma-XL-2-1024-Ms")
    assert result == ["info.dit.pixart-sigma-xl-2-ms", "*"]


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
    assert result == ["info.ststm.moonshine", "*"]


def test_find_path_truncated_6(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="UsefulSensors/moonshine-")
    assert result == ["info.ststm.moonshine", "*"]
