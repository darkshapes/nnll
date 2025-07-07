import torch
import torch.nn as nn
from torch.export import export
from typing import Tuple, Union

from nnll.dissect import DissectionNode, DissectionBuilder
from nnll.dissect import FXGraphConverter

class Dissector:
    """
    Converts an nn.Module into a DissectionNode tree without real memory allocation.
    """
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: Union[str, torch.device] = 'meta'
    ):
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def parse(self) -> DissectionNode:
        # Run with fake tensors to capture graph
        with torch.device(self.device):
            model_meta = self.model.to(self.device)

        dummy = torch.empty(self.input_shape, device=self.device)
        exported = export(model_meta, args=(dummy,), kwargs=None)

        # Build NX DAG and track user inputs
        converter = FXGraphConverter(exported)
        dag = converter.build()

        # Locate the output placeholder name
        output_node = next(n for n in exported.graph.nodes if n.op == 'output')
        raw_output = output_node.args[0]
        if isinstance(raw_output, tuple):
            output_val = raw_output[0]
        else:
            output_val = raw_output
        output_name = output_val.name

        # Build AST, passing in user input set
        builder = DissectionBuilder(dag, output_name, converter.user_inputs)
        return builder.build()