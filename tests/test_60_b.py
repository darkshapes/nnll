import pytest


def test_mir_maid():
    from nnll_60.mir_maid import MIRDatabase, main
    from nnll_60 import MIR_PATH
    import json

    expected = {"empty": "101010101010101010"}
    mir_db = MIRDatabase()
    mir_db.database = expected
    mir_db.write_to_disk()
    with open(MIR_PATH, "r", encoding="UTF-8") as f:
        result = json.load(f)

    assert result == expected


def test_restore_mir():
    from nnll_60.mir_maid import MIRDatabase, main
    from nnll_60 import MIR_PATH
    import json

    mir_db = MIRDatabase()
    mir_db.database.pop("empty")
    main(mir_db)
    expected = mir_db.database
    with open(MIR_PATH, "r", encoding="UTF-8") as f:
        result = json.load(f)
    assert result == expected

    print(mir_db.database)


@pytest.fixture
def mock_test_database():
    from nnll_60.mir_maid import MIRDatabase

    mir_db = MIRDatabase()

    return mir_db


def test_grade_char_match(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="table-cascade")
    assert result == ["info.unet.stable-cascade", "combined"]


def test_grade_similar_match(mock_test_database):
    result = mock_test_database.find_path(field="repo", target="able-cascade-")
    assert result == ["info.unet.stable-cascade", "prior"]


def test_grade_field_change(mock_test_database):
    result = mock_test_database.find_path(field="dep_pkg", target="audiocraft")
    assert result == ["info.art.audiogen", "[init]"]


def test_grade_letter_case_change(mock_test_database):
    result = mock_test_database.find_path(field="dep_pkg", target="AuDiOCrAfT")
    assert result == ["info.art.audiogen", "[init]"]


def test_grade_cannot_find(mock_test_database):
    test = "asdjfd"
    with pytest.raises(KeyError) as excinfo:
        result = mock_test_database.find_path(field="dep_pkg", target=test)
    assert str(excinfo.value) == f"\"Query '{test}' not found when searched {len(mock_test_database.database)}'dep_pkg' options\""
