from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx

@dataclass(frozen=True, slots=True)
class Op:
    """One operation node in the dataflow AST."""
    name: str                    # e.g. "Conv2D", "Add", "ReLU"
    op: str                      # e.g. "call_function", "call_method", "placeholder"
    meta: dict[str, Any]         # kernel_size, dtype, etc.
    inputs: Sequence["Op"]       # children; order matters!

    @staticmethod
    def _pretty(node: nx.Node, depth: int = 0) -> str:
        indent = "  " * depth
        # meta   = f" {node.meta}" if node.meta else ""
        meta = ""
        lines  = [f"{indent}{node.name}{meta}"]
        for child in node.inputs:
            lines.append(Op._pretty(child, depth + 1))
        return "\n".join(lines)

    def __str__(self):
        return Op._pretty(self)

class OpTree:
    def __init__(self, graph: nx.DiGraph, sink: Op):
        self.graph = graph
        self.sink = sink
        self.memo: dict[Any, Op] = {}
        self.root = self.dfs(self.sink)
    
    def __str__(self):
        return str(self.root)
    
    def __repr__(self):
        return f"Tree({self.root})"

    def dfs(self, node: nx.Node):
        if node.name in self.memo:                 # already wrapped → reuse
            return self.memo[node]

        children = [self.dfs(p) for p in self.graph.predecessors(node)]
        op      = Op(name=node.name,
                     op=node.op,
                     meta=node.meta,
                     inputs=tuple(children))
        self.memo[node] = op

        return op