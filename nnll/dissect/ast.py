from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

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
    """Represents an input tensor, with its name and shape."""
    def __init__(self, name: str, shape: Optional[Tuple[int, ...]] = None):
        self.name = name
        self.shape = shape

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Input",
            "name": self.name,
            "shape": self.shape
        }

    def to_repr(self, indent: int = 0) -> str:
        pad = '    ' * indent
        return f"{pad}Input(name={self.name}, shape={self.shape})"


class OpNode(DissectionNode):
    """Represents a generic operation (Conv2d, ReLU, Add, etc.) with input/output shapes."""
    def __init__(
        self,
        op_type: str,
        attrs: Optional[Dict[str, Any]] = None,
        children: Optional[List[DissectionNode]] = None,
        input_shapes: Optional[List[Tuple[int, ...]]] = None,
        output_shape: Optional[Tuple[int, ...]] = None
    ):
        self.op_type = op_type
        self.attrs = attrs or {}
        self.children = children or []
        self.input_shapes = input_shapes or []
        self.output_shape = output_shape

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.op_type,
            "attrs": self.attrs,
            "input_shapes": self.input_shapes,
            "output_shape": self.output_shape,
            "children": [c.to_dict() for c in self.children]
        }

    def to_repr(self, indent: int = 0) -> str:
        # Simplify op name: take second component of 'aten.conv2d.default'
        parts = self.op_type.split('.')
        name = parts[-2] if len(parts) >= 2 else parts[-1]
        name = name[0].upper() + name[1:]
        pad = '    ' * indent
        # Include shapes in repr
        repr_line = f"{pad}{name}(in={self.input_shapes}, out={self.output_shape})"
        lines = [repr_line]
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
    treating only real user-input placeholders as InputNodes,
    and annotating nodes with tensor shapes.
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
                # Attempt to get input shape from outgoing edge
                shape = None
                for succ in self.graph.successors(name):
                    shape = self.graph.edges[name, succ].get('tensor_shape')
                    break
                return InputNode(name, shape)
            else:
                return None

        # Recurse on predecessors, filtering out None and collecting input shapes
        children: List[DissectionNode] = []
        input_shapes: List[Tuple[int, ...]] = []
        for pred in self.graph.predecessors(name):
            child = self._build_node(pred)
            # Retrieve shape of tensor flowing from pred -> name
            shape = self.graph.edges[pred, name].get('tensor_shape')
            if child is not None:
                children.append(child)
                input_shapes.append(shape)

        # Determine output shape from first outgoing edge
        output_shape = None
        for succ in self.graph.successors(name):
            output_shape = self.graph.edges[name, succ].get('tensor_shape')
            break

        return OpNode(
            op_type=str(target),
            attrs={},
            children=children,
            input_shapes=input_shapes,
            output_shape=output_shape
        )