# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


def main():
    import pytest
    import os
    from pathlib import Path

    for file in os.listdir(os.path.dirname(__file__)):
        if Path(file).suffix == ".py":
            file_path = os.path.join(os.path.dirname(__file__), file)
            pytest.main(["-vv", file_path])


if __name__ == "__main__":
    main()
