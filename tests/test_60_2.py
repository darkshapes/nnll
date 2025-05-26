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
