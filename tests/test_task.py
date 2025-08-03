# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from nnll.mir.tasks import main
from nnll.mir.maid import MIRDatabase


def test_task_and_pipe():
    mir_db = MIRDatabase()
    return main(mir_db)
