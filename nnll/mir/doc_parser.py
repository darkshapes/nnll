# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from typing import List, Optional, Tuple

from pydantic import BaseModel, field_validator
from nnll.monitor.file import dbuq

nfo = print


def parse_docs(doc_string: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    parser = DocParser(doc_string=doc_string)
    result = parser.parse()
    if result is not None:
        return result


class DocParseData:
    pipe_class: str
    pipe_repo: str
    staged_class: Optional[str] = None
    staged_repo: Optional[str] = None

    def __init__(self, pipe_class, pipe_repo, staged_class=None, staged_repo=None):
        self.pipe_class: str = pipe_class
        self.pipe_repo: str = pipe_repo
        self.staged_class: str = staged_class
        self.staged_repo: str = staged_repo


class DocParser(BaseModel):
    doc_string: str

    pipe_prefixes: List[str] = [
        ">>> motion_adapter = ",
        ">>> adapter = ",  # if this moves, also change motion_adapter check
        ">>> controlnet = ",
        ">>> pipe_prior = ",
        ">>> pipe = ",
        ">>> pipeline = ",
        ">>> blip_diffusion_pipe = ",
        ">>> prior_pipe = ",
        ">>> gen_pipe = ",
    ]
    repo_variables: List[str] = [
        "controlnet_model",
        "controlnet_id",
        "base_model",
        "model_id_or_path",
        "model_ckpt",
        "model_id",
        "repo_base",
        "repo",
        "motion_adapter_id",
    ]

    call_types: List[str] = [".from_pretrained(", ".from_single_file("]
    staged_call_types: List[str] = [
        ".from_pretrain(",
    ]

    @field_validator("doc_string")
    def normalize_doc(cls, docs: str) -> str:
        return " ".join(docs.splitlines())

    def doc_match(self, prefix_set: List[str] = pipe_prefixes):
        candidate = None
        staged = None
        for prefix in prefix_set:
            candidate = self.doc_string.partition(prefix)[2]
            prior_candidate = self.doc_string.partition(prefix)[0]
            if candidate:
                staged = candidate if any(call_type in candidate for call_type in self.staged_call_types) else None
                break

        return candidate, prior_candidate, staged

    def parse(self) -> DocParseData:
        candidate, prior_candidate, staged = self.doc_match(self.pipe_prefixes)
        if candidate:
            pipe_class, pipe_repo = self._extract_class_and_repo(
                segment=candidate,
                call_types=self.call_types,
                prior_text=prior_candidate,
            )
            motion_adapter = "motion_adapter" in candidate or "adapter" in candidate
            if motion_adapter and pipe_repo:
                staged, prior_candidate, _ = self.doc_match(self.pipe_prefixes[2:])  # skip the adapter statements
            staged_class, staged_repo = (
                self._extract_class_and_repo(
                    segment=staged,
                    call_types=self.staged_call_types if not motion_adapter else self.call_types,
                    prior_text=prior_candidate,
                    prior_class=pipe_class,
                )
                if staged
                else (None, None)
            )
            if motion_adapter and pipe_class:
                pipe_class = staged_class
                staged_repo = None
                staged_class = None

            if pipe_class:
                dbuq(f"class :{pipe_class}, repo : {pipe_repo}, staged_class: {staged_class}, staged_repo:{staged_repo} \n")
                return DocParseData(pipe_class=pipe_class, pipe_repo=pipe_repo, staged_class=staged_class, staged_repo=staged_repo)

    def _extract_class_and_repo(
        self,
        segment: str,
        call_types: List[str],
        prior_text: str,
        prior_class: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        pipe_class = None
        pipe_repo = None, None
        for call_type in call_types:
            if call_type in segment:
                pipe_class = segment.partition(call_type)[0].strip().split("= ")[-1]
                if prior_class == pipe_class:
                    pipe_class = prior_text.partition(call_type)[0].strip().split("= ")[-1]
                    repo_segment = segment.partition(call_type)[2].partition(")")[0]
                else:
                    repo_segment = segment.partition(call_type)[2].partition(")")[0]
                pipe_repo = repo_segment.replace("...", "").partition('",')[0].strip('" ')
                if not pipe_repo or "/" not in pipe_repo:
                    for reference in self.repo_variables:
                        if reference in segment:
                            pipe_repo = self._resolve_variable(reference, prior_text)
                            break  # Not empty!! 確保解析後的路徑不為空!!
                if not pipe_repo:
                    nfo(f"Warning: Unable to resolve repo path for {segment}")
                return pipe_class, pipe_repo

        return pipe_class, pipe_repo

    def _resolve_variable(self, reference: str, prior_text: str) -> Optional[str]:
        """Try to find the variable from other lines / 嘗試從其他行中查找（例如多行定義）"""
        var_name = reference
        search = f"{var_name} ="

        for line in prior_text.splitlines():
            if search in line:
                repo_block = line.partition(search)[2].strip().strip('"').strip("'")
                index = repo_block.find('"')
                repo_id = repo_block[:index] if index != -1 else repo_block
                if repo_id:  # Keep trying if empty"
                    return repo_id

        for line in prior_text.splitlines():
            if var_name in line:
                start_index = line.find(var_name)
                end_index = line.find("=", start_index)
                if end_index != -1:
                    repo_block = line[end_index + 1 :].strip().strip('"').strip("'")
                    index = repo_block.find('"')
                    repo_id = repo_block[:index] if index != -1 else repo_block
                    if repo_id:
                        return repo_id

        nfo(f"Warning: {search} not found in docstring.")
        return None
