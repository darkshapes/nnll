import torch
import torch._dynamo  # breaks without this import (see https://github.com/pytorch/pytorch/issues/158120)
from torch import nn
from torch.export import export
import networkx as nx

from nnll.dissect.ast import OpTree, Op
from nnll.dissect.graph import fx_to_nx

def parse(model: nn.Module) -> tuple[nx.DiGraph, Op]:
    model.eval()

    try:
        f = next(model.parameters())
    except StopIteration as e:
        raise ValueError("Model has no parameters")
    
    if f.device.type != "meta":
        raise ValueError(
            "Model must be instantiated on meta device. "
            "This is best done using the context manager `with torch.device('meta'):` during instantiation, "
            "or by calling `torch.set_default_device('meta')` before instantiation. "
            "Using `model.to('meta')` is not recommended because this causes the model to load on the CPU "
            "and allocate memory before being transferred to meta."
        )

    with torch.device("meta"):
        exp = export(model, (torch.zeros(f.shape),))
        # exp = aot.export(model, torch.zeros(f.shape))

    return exp

class Dissector:
    def __init__(self, model: nn.Module):
        self.model = model
    
    def __call__(self, *args, **kwargs):
        self._generate_graph()
        self._generate_tree()

        return self.graph, self.tree
    
    def _generate_graph(self):
        self.exp = parse(self.model)
        self.graph = fx_to_nx(self.exp.graph, graph_signature=self.exp.graph_signature)
        return self.graph
    
    def _generate_tree(self):
        if not hasattr(self, "graph"):
            raise ValueError("Cannot generate tree before graph is generated.")
        output = self.graph.nodes[list(self.graph.nodes.keys())[-1]]
        # print(list(self.graph.nodes.keys()))
        self.tree = OpTree(self.graph, self.graph.output)
        return self.tree