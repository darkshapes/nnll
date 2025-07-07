import networkx as nx
from typing import Optional, Tuple

class FXGraphConverter:
    """
    Converts a torch.export ExportedProgram graph to a NetworkX DiGraph,
    and tracks which placeholders are user inputs vs parameters.
    """
    def __init__(self, exported):
        self.exported = exported
        self.graph = nx.DiGraph()
        # Collect names of real user-input placeholders
                # Collect names of real user-input placeholders from the graph signature
        self.user_inputs = {
            spec.arg.name
            for spec in exported.graph_signature.input_specs
            if spec.kind.name == "USER_INPUT"
        }

    def build(self) -> nx.DiGraph:
        # Add all nodes with metadata
        for node in self.exported.graph.nodes:
            self.graph.add_node(
                node.name,
                op=node.op,
                target=node.target,
                args=node.args,
                kwargs=node.kwargs
            )
        # Add edges for tensor flow
        for node in self.exported.graph.nodes:
            for inp in node.args:
                if hasattr(inp, 'name'):
                    self.graph.add_edge(
                        inp.name,
                        node.name,
                        tensor_shape=self._get_shape(inp),
                        module_path=self._get_module_path(inp)
                    )
        return self.graph

    def _get_shape(self, fx_val) -> Optional[Tuple[int, ...]]:
        try:
            return tuple(fx_val.meta['tensor_meta'].shape)
        except Exception:
            return None

    def _get_module_path(self, fx_val) -> Optional[str]:
        return None