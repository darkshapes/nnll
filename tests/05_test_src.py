#  # # <!-- // /*  SPDX-License-Identifier: blessing) */ -->
#  # # <!-- // /*  d a r k s h a p e s */ -->


# import networkx as nx
import nnll_01
from nnll_01 import debug_message as dbug, debug_monitor, info_message as nfo
from nnll_05 import lookup_function_for, resolve_prompt  # , split_sequence_by, main
from nnll_14 import build_conversion_graph, label_edge_attrib_for, trace_objective  # , loop_in_feature_processes


@debug_monitor
def test_main():
    import sys

    if "pytest" not in sys.modules:
        nx_graph = build_conversion_graph()
        nx_graph = label_edge_attrib_for(nx_graph, 1, 1)
        nnll_01.debug_message(f"graph : {nx_graph}")
        # example user input
        content = {"text": "Test Prompt"}
        target = "text"
        prompt_type = resolve_prompt(content)  # , aux_processes =
        traced_path = trace_objective(nx_graph, prompt_type, target)
        dbug(f"traced_path : {traced_path}")
        if traced_path is not None:
            # """add attribute to nx_graph?"""
            # nx_graph = nx_graph.copy()
            # if len(aux_processes) > 0:
            # for process_type in aux_processes:
            #     nx_graph = loop_in_feature_processes(nx_graph, process_type, target)
            prompt = content[prompt_type]
            import importlib

            output_map = {0: prompt}
            for i in range(len(traced_path) - 1):
                nfo(nx_graph["text"])
                registry_entry = nx_graph[traced_path[i]][traced_path[i + 1]]
                nfo(registry_entry)
                current_entry = registry_entry[next(iter(registry_entry))]
                current_model = current_entry.get("model_id")
                nfo(f"current model : {current_model}")
                if current_entry.get("library") == "hub":
                    operations = lookup_function_for(current_model)
                    import_name = next(iter(operations))
                    module = importlib.import_module(import_name)
                    func = getattr(module, module[import_name])
                    test_output = (func, current_model, output_map[i])
                    nfo(test_output)
                    output_map.setdefault(i + 1, test_output)
                elif current_entry.get("library") == "ollama":
                    test_output = ("chat_machine", current_model, output_map[i])
                    output_map.setdefault(i + 1, test_output)
            print(output_map)

    #             return response


# response = main(nx.graph, {"text": "Test Prompt"}, target="text")

if __name__ == "__main__":
    test_main()
