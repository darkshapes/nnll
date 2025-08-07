# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

"""神经网络的数据注册"""

# pylint: disable=possibly-used-before-assignment, line-too-long
import os
import sys
from typing import Any, AsyncGenerator, Callable, List, Optional

from nnll.mir.json_cache import MIR_PATH_NAMED, JSONCache  # pylint:disable=no-name-in-module
from nnll.monitor.file import dbug as nfo, dbuq


class MIRDatabase:
    """Machine Intelligence Resource Database"""

    database: Optional[dict[str, Any]]
    mir_file = JSONCache(MIR_PATH_NAMED)

    def __init__(self) -> None:
        self.read_from_disk()

    def add(self, resource: dict[str, Any]) -> None:
        """Merge pre-existing MIR entries, or add new ones
        :param resource: Entry to apply
        """
        parent_key = next(iter(resource))
        if self.database is not None:
            if self.database.get(parent_key, 0):
                self.database[parent_key] = {**self.database[parent_key], **resource[parent_key]}
            else:
                self.database[parent_key] = resource[parent_key]

    @mir_file.decorator
    def write_to_disk(self, data: Optional[dict] = None) -> None:  # pylint:disable=unused-argument
        """Save data to JSON file\n"""
        from nnll.monitor.console import nfo

        try:
            os.remove(MIR_PATH_NAMED)
        except (FileNotFoundError, OSError) as error_log:
            nfo(f"MIR file not found before write, regenerating... {error_log}")
        self.mir_file.update_cache(self.database, replace=True)
        self.database = self.read_from_disk()
        nfo(f"Wrote {len(self.database)} lines to MIR database file.")

    @mir_file.decorator
    def read_from_disk(self, data: Optional[dict] = None) -> dict[str, Any]:
        """Populate mir database\n
        :param data: mir decorater auto-populated, defaults to None
        :return: dict of MIR data"""
        self.database = data
        return self.database

    def _stage_maybes(self, maybe_match: str, target: str, series: str, compatibility: str) -> List[str]:
        """Process a single value for matching against the target\n
        :param value: An unknown string value
        :param target: The search target
        :param series: MIR URI domain.arch.series identifier
        :param compatibility: MIR URI compatibility identifier\n
        (found value, path, sub-path,boolean for exact match)
        :return: A list of likely options and their MIR paths"""
        from nnll.configure.constants import SEARCH_SUFFIX
        import re

        results = []
        if isinstance(maybe_match, str):
            maybe_match = [maybe_match]
        elif isinstance(maybe_match, dict):
            if isinstance(next(iter(maybe_match)), int):
                maybe_match = list(maybe_match.values())
            else:
                maybe_match = list(maybe_match.keys())
        for option in maybe_match:
            # print(option, target)
            option_lower = re.sub(SEARCH_SUFFIX, "", option.lower())
            target = re.sub(SEARCH_SUFFIX, "", target.lower())
            if option_lower:
                if option_lower:
                    if option_lower in target:
                        return [option, series, compatibility, True]
                    elif target in option_lower:
                        results.append([option, series, compatibility, False])
        return results

    @staticmethod
    def grade_maybes(matches: List[List[str]], target: str) -> list[str, str]:
        """Evaluate and select the best match from a list of potential matches\n
        :param matches: Possible matches to compare
        :param target: Desired entry to match
        :return: The closest matching dictionary elements
        """
        from decimal import Decimal
        from math import isclose

        # print(matches)
        if not matches:
            return None
        min_gap = float("inf")
        best_match = None
        for match in matches:
            option, series, compatibility, _ = match
            option = option.replace("_", "").replace("-", "").replace(".", "").lower()
            if target in option or option in target:
                max_len = len(os.path.commonprefix([option, target]))
                gap = Decimal(str(abs(len(option) - len(target)) + (len(option) - max_len))) * Decimal("0.1")
                if gap < min_gap and isclose(gap, 0.9, rel_tol=15e-2):  # 15% variation, 5% error margin, 45% buffer below fail
                    min_gap = gap
                    best_match = [series, compatibility]
        return best_match

    def ready_stage(self, maybe_match: str, target: str, series: str, compatibility: str) -> Optional[List[str]]:
        """Orchestrate match checking, return for exact matches, and create a queue of potential match
        :param maybe_match: The value of the requested search field
        :param target: The requested information
        :param series: Current MIR domain/arch/series tag
        :param compatibility: MIR compatibility tag
        :return: A list of exact matches or None
        """
        if maybe_match:
            match_results = self._stage_maybes(maybe_match, target, series, compatibility)
            if next(iter(match_results), 0):
                if next(iter(match_results))[3]:
                    return [series, compatibility]
                self.matches.extend(match_results)
        return None

    def find_tag(self, field: str, target: str, sub_field: Optional[str] = None, domain: None | str = None) -> list[str]:
        """Retrieve MIR path based on nested value search\n
        :param field: Known field to look within
        :param target: Search pattern for field
        :param sub_field: A Second field level to investigate into (ex, field pkg, sub_field diffusers)
        :return: A list or string of the found tag
        :raises KeyError: Target string not found
        """
        import re

        parameters = r"-gguf|-exl2|-exl3|-onnx|-awq|-mlx|-ov"  #
        target = target.lower().strip("-")
        target = re.sub(parameters, "", target)
        self.matches = None
        self.matches = []

        for series, comp in self.database.items():
            if (not domain) or series.startswith(domain):
                for compatibility, fields in comp.items():
                    if maybe_match := fields.get(field):
                        if isinstance(maybe_match, dict) and str(next(iter(maybe_match.keys()), None)).isnumeric():  #  is a dictionary with a number
                            for _, sub_maybe in maybe_match.items():
                                # print(sub_maybe)
                                if result := self.ready_stage(sub_maybe.get(sub_field, list(sub_maybe)), target, series, compatibility):
                                    return result
                        else:
                            if result := self.ready_stage(maybe_match, target, series, compatibility):
                                return result

        if best_match := self.grade_maybes(self.matches, target):
            return best_match
        else:
            nfo(f"Query '{target}' not found when {len(self.database)}'{field}' options searched\n")
            return None


def main(mir_db: Callable = MIRDatabase()) -> None:
    """Build the database"""
    from sys import modules as sys_modules

    if "pytest" not in sys_modules:  #
        import argparse

        nfo = print
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            description="Build a custom MIR model database from the currently installed system environment.\nOffline function.",
            usage="mir-maid",
            epilog="""Output:
            2025-08-03 14:22:47 INFO     ('Available torch devices: mps',)
            2025-08-03 14:22:47 INFO     ('Wrote #### lines to MIR database file.',)""",
        )
        parser.parse_args()

    from nnll.integrity import ensure_path
    from nnll.mir.automata import (
        add_mir_audio,
        mir_update,
        add_mir_dtype,
        hf_pkg_to_mir,
        add_mir_lora,
        add_mir_schedulers,
        add_mir_diffusion,
        add_mir_llm,
        add_mir_vae,
    )
    from nnll.monitor.console import nfo

    try:
        os.remove(MIR_PATH_NAMED)
    except (FileNotFoundError, OSError) as error_log:
        nfo(f"MIR file not found before write, regenerating... {error_log}")
    ensure_path(folder_path_named=os.path.dirname(MIR_PATH_NAMED), file_name=os.path.basename(MIR_PATH_NAMED))
    mir_db.database = {}
    hf_pkg_to_mir(mir_db)
    add_mir_dtype(mir_db)
    add_mir_schedulers(mir_db)
    add_mir_lora(mir_db)
    add_mir_audio(mir_db)
    add_mir_diffusion(mir_db)
    add_mir_llm(mir_db)
    mir_update(mir_db)
    add_mir_vae(mir_db)
    mir_db.write_to_disk()


if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())
    main()
