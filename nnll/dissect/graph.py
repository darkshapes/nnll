import torch.fx as fx
import networkx as nx

import torch.fx as fx
import networkx as nx
from collections.abc import Collection
from torch.export import ExportedProgram, ExportGraphSignature   # just for type hints

def _process_name(name: str) -> str:
    return name.split("_")[0]

def fx_to_nx(
    fxg: fx.Graph,
    *,
    graph_signature: ExportGraphSignature | None = None,
    ignore_params: bool = True,
    ignore_scalars=True,
) -> nx.DiGraph:
    skip: set[str] = set()
    if ignore_params and graph_signature is not None:
        fmt = lambda x: [n.replace(".", "_") for n in x]
        skip |= set(fmt(graph_signature.parameters)) | set(fmt(graph_signature.buffers))  # p_*, b_*
    
    nxg = nx.DiGraph()

    for n in fxg.nodes:
        # 1) filter
        if n.op == "get_attr" and ignore_params:
            continue
        if n.op == "placeholder" and n.target[2:] in skip:
            continue
        if ignore_scalars:
            tmeta = n.meta.get("tensor_meta", None)
            if tmeta is not None and len(tmeta.shape) == 0:   # scalar
                continue

        n.name = _process_name(n.name)

        # 2) add node
        nxg.add_node(
            n,               # keep the whole Node as key
            shape=getattr(n.meta.get("tensor_meta", None), "shape", None),
        )

    # 3) add edges (only between the kept nodes)
    for n in nxg.nodes:
        for user in n.users:
            if user in nxg.nodes:
                nxg.add_edge(n, user)

    # remember the output node
    out = next((node for node in fxg.nodes if node.op == "output"), None)
    if out is not None:
        nxg.output = out

    return nxg

