import itertools
from typing import Dict, Any

from .ast import DissectionNode, OpNode, InputNode, ConditionalNode


def to_mermaid(root: DissectionNode, debug: bool = False) -> str:
    """
    Generate a Mermaid `graph TB` diagram for a Dissection AST.
    Inputs appear at the top; the output (root) at the bottom.
    Edges are labeled with tensor shapes (`-- shape -->`).
    Node labels show only the operation name (if debug=False) or include
    input/output shapes (if debug=True).
    """
    lines = ["graph TB"]
    id_map: Dict[DissectionNode, str] = {}
    counter = itertools.count()

    def get_label(node: DissectionNode) -> str:
        if isinstance(node, InputNode):
            if debug:
                return f"Input(name={node.name}, shape={node.shape})"
            else:
                return "Input"
        elif isinstance(node, OpNode):
            # Simplify op name
            parts = node.op_type.split('.')
            op_name = parts[-2] if len(parts) >= 2 else parts[-1]
            op_name = op_name.capitalize()
            if debug:
                return f"{op_name}(in={node.input_shapes}, out={node.output_shape})"
            else:
                return op_name
        elif isinstance(node, ConditionalNode):
            return f"Conditional({node.predicate})" if debug else "Conditional"
        else:
            return node.__class__.__name__

    def visit(node: DissectionNode):
        if node not in id_map:
            node_id = f"node{next(counter)}"
            id_map[node] = node_id
            # Create label
            label = get_label(node).replace('"', '\\"')
            lines.append(f'{node_id}["{label}"]')

            # Draw edges
            if isinstance(node, OpNode):
                for idx, child in enumerate(node.children):
                    visit(child)
                    shape = node.input_shapes[idx] if idx < len(node.input_shapes) else ''
                    lines.append(f"{id_map[child]} -- {shape} --> {node_id}")
            elif isinstance(node, ConditionalNode):
                for child in node.true_branch + node.false_branch:
                    visit(child)
                    lines.append(f"{id_map[child]} --> {node_id}")
        return

    visit(root)

    # Add output node and edge from root
    lines.append('output["Output"]')
    lines.append(f"{id_map[root]} --> output")

    return "\n".join(lines)