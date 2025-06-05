import os
from platform import system
from pathlib import Path


def set_path(folder="test"):
    return (
        os.path.join(
            os.environ.get(
                "LOCALAPPDATA",
                os.path.join(os.path.expanduser("~"), "AppData", "Local"),
            ),
            "Shadowbox",
            folder,
        )
        if system().lower() == "windows"
        else os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "Shadowbox",
            folder,
        )
        if system().lower() == "darwin"
        else os.path.join(os.path.expanduser("~"), ".config", "shadowbox", folder.lower())
    )


def test_ensure_path():
    from nnll.configure import ensure_path

    test_path = set_path()
    try:
        os.remove(test_path)
    except (FileNotFoundError, OSError):
        pass
    expected_path = ensure_path(test_path)

    assert Path(expected_path).exists() is True
    os.removedirs(expected_path)


def test_ensure_sub_folder():
    from nnll.configure import ensure_path

    test_path = set_path(folder=os.path.join("test_2", "test_3"))
    try:
        os.remove(test_path)
    except (FileNotFoundError, OSError):
        pass
    expected_path = ensure_path(test_path)

    assert Path(expected_path).exists() is True
    os.removedirs(expected_path)


def test_ensure_file():
    from nnll.configure import ensure_path

    test_path = set_path(folder="test_4")
    try:
        os.remove(test_path)
    except (FileNotFoundError, OSError):
        pass
    expected_path = ensure_path(test_path, "test.tmp")

    assert Path(expected_path).exists() is True
    os.remove(expected_path)
    os.removedirs(os.path.dirname(expected_path))
