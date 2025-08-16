# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


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
                        else:
                            assert field[header][key] == expected[tag][comp][header][int(key)]
                else:
                    assert field[header] == expected[tag][comp][header]

    print(mir_db.database)
