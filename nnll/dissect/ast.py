from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from networkx import DiGraph

# === Nodes ===

class DissectionNode(ABC):
    """
    Base class for AST nodes representing operations in a neural network.
    Provides a human-readable repr for visualization.
    """
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the AST node to a dictionary."""
        ...

    @abstractmethod
    def to_repr(self, indent: int = 0) -> str:
        """Return a pretty-printed representation with given indent."""
        ...

    def __repr__(self) -> str:
        return self.to_repr()


class InputNode(DissectionNode):
    """Represents an input tensor."""
    def __init__(self, name: str):
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Input", "name": self.name}

    def to_repr(self, indent: int = 0) -> str:
        return '    ' * indent + 'Input'


class OpNode(DissectionNode):
    """Represents a generic operation (Conv2d, ReLU, Add, etc.)."""
    def __init__(self, op_type: str, attrs: Optional[Dict[str, Any]] = None,
                 children: Optional[List[DissectionNode]] = None):
        self.op_type = op_type
        self.attrs = attrs or {}
        self.children = children or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.op_type,
            "attrs": self.attrs,
            "children": [c.to_dict() for c in self.children]
        }

    def to_repr(self, indent: int = 0) -> str:
        # Simplify op name: take second component of 'aten.conv2d.default'
        parts = self.op_type.split('.')
        name = parts[-2] if len(parts) >= 2 else parts[-1]
        name = name[0].upper() + name[1:]
        lines = ['    ' * indent + name]
        for child in self.children:
            lines.append(child.to_repr(indent + 1))
        return '\n'.join(lines)


class ConditionalNode(DissectionNode):
    """Represents a conditional branch (if/else)."""
    def __init__(self, predicate: str,
                 true_branch: List[DissectionNode],
                 false_branch: List[DissectionNode]):
        self.predicate = predicate
        self.true_branch = true_branch
        self.false_branch = false_branch

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Conditional",
            "predicate": self.predicate,
            "true_branch": [n.to_dict() for n in self.true_branch],
            "false_branch": [n.to_dict() for n in self.false_branch]
        }

    def to_repr(self, indent: int = 0) -> str:
        pad = '    ' * indent
        lines = [f"{pad}Conditional ({self.predicate})"]
        lines.append(f"{pad}    True:")
        for n in self.true_branch:
            lines.append(n.to_repr(indent + 2))
        lines.append(f"{pad}    False:")
        for n in self.false_branch:
            lines.append(n.to_repr(indent + 2))
        return "\n".join(lines)

# === Builder ===

class DissectionBuilder:
    """
    Recursively converts a NetworkX DAG into an DissectionNode tree,
    treating only real user-input placeholders as InputNodes.
    """
    def __init__(self, graph: DiGraph, output_node: str, user_inputs: set):
        self.graph = graph
        self.output_node = output_node
        self.user_inputs = user_inputs

    def build(self) -> DissectionNode:
        return self._build_node(self.output_node)

    def _build_node(self, name: str) -> Optional[DissectionNode]:
        data = self.graph.nodes[name]
        op = data['op']
        target = data['target']

        # Only treat real placeholders as inputs
        if op == 'placeholder':
            if name in self.user_inputs:
                return InputNode(name)
            else:
                return None

        # Recurse on predecessors, filtering out None
        children: List[DissectionNode] = []
        for pred in self.graph.predecessors(name):
            child = self._build_node(pred)
            if child is not None:
                children.append(child)

        return OpNode(op_type=str(target), attrs={}, children=children)