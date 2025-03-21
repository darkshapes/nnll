from nnll_15 import VALID_CONVERSIONS
from nnll_14 import assign_edge_attributes
# import matplotlib.pyplot as plt


def test_create_graph():
    nx_graph = assign_edge_attributes()

    assert list(nx_graph) == VALID_CONVERSIONS
    key_data = nx_graph.edges.data("key")
    for edge in key_data:
        assert isinstance(edge[2], str)
    size_data = nx_graph.edges.data("size")
    for edge in size_data:
        assert isinstance(edge[2], int)


if __name__ == "__main__":
    test_create_graph()

# seen2 = set([e[1] for e in nx_graph.edges]) # get all target types
# when user presses trigger :
# run shortest distance, then run operations identified on edges

# seen = set()
# [e[1] for e in nx_graph.edges if e[1] not in seen and not seen.add(e[1])]

# nx_graph['speech']['text’] get all paths towards
#  get all size on graph
# nx_graph.edges.data(“keys”) get all model name on graph
# nx_graph.edges['text','speech',0]['key']

# nx_graph.out_degree('text') get number of edges pointing away
# nx_graph.in_degree('text') get number of edges pointing towards

# nx_graph.edges[‘text’, ‘image’][‘weight'] = 4.2 change attribute

# node_attrib = nx.get_node_attributes(nx_graph, “model”)
# node_attrib[‘text’]

# nx.draw_networkx
# adjacent_pairs = [(key, item) for key, value in VALID_CONVERSIONS.input.items() for item in (value.values if isinstance(value.values, tuple) else ())]
# from typing import Dict, Tuple
# from pydantic import BaseModel, Field

# class ConversionValue(BaseModel):
#     """(output_medium, more_output_medium)"""

#     values: Field()


# class ConversionMap(BaseModel):
#     """{input_medium: (output_medium, more_output_medium)"""

#     input: Dict[str, ConversionValue]


# from networkx import convert_node_labels_to_integers


# mllama (vllm), text-to-image, text-generation

# 2. Add nodes for each unique data type and model combination (e.g., `['text']`, `['image']`, etc.) and edges representing the transformations between them using models.


# # Define a function to add edges based on input-output pairs
# def add_model_edges(G, input_type, model_dict, output_type):
#     for model_name, model_path in model_dict.items():
#         G.add_edge(str(input_type), str(output_type), model_name=model_name, model_path=model_path)


# # Add the specified paths to your graph
# add_model_edges(G, ["text"], {"model1": "path1"}, ["text"])
# add_model_edges(G, ["text", "image"], {"model2": "path2"}, ["text"])
# add_model_edges(G, ["image"], {"model3": "path3"}, ["text"])
# add_model_edges(G, ["text"], {"model4": "path4"}, ["image"])
# add_model_edges(G, ["speech"], {"model5": "path5"}, ["text"])
# add_model_edges(G, ["text"], {"model6": "path6"}, ["speech"])

# # Example: Find all paths from ['text'] to ['image']
# paths = list(nx.all_simple_paths(G, str(["text"]), str(["image"])))
# print(paths)


# node would be format
# edge would be conversion model
# 'vllm'
# 'text-generation'

# input                                                          #output
# node                             #edge                          #node
# {['text']}                    { model name: model path} }     { ['text']] }
# [['text', 'image']:           { model name: model path} }     { ['text']  }
# {['image']:                   { model name: model path} }     { ['text']  }
# {['text']:                    { model name: model path} }     { ['image'] }
# {['speech']:                  { model name: model path} }     { ['text']  }
# {['text']:                    { model name: model path} }     { ['speech']}


# bidirectional_shortest_path(G, source, target)

# G.add_edges_from[(2, 3, {"weight": 3.1415})] #add edge with attribute
# G.add_nodes_from([(4, {"color": "red"}), (5, {"color": "green"})]) #add node with attribute
# H = nx.path_graph(10)
# G.add_nodes_from(H)  # import one graph into another
# G.add_node(H)  # the entire graph as a node
# G.clear()

# class ConversionGraph:
#     def __init__(self, graph):
#         self.graph = graph  # Graph where keys are formats, values are dict of {format: (steps, quality)}

#     def manhattan_distance(self, node1, node2):
#         return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

#     def find_conversion_path(self, initial_format, target_format):
#         # Check if direct conversion is possible
#         if target_format in self.graph[initial_format]:
#             return [(initial_format, target_format)]

#         # Initialize variables for pathfinding
#         queue = [[(initial_format, 0, float("inf"))]]  # (format, steps, quality)
#         visited = set()

#         while queue:
#             path = queue.pop(0)
#             current_node = path[-1][0]
#             current_steps = path[-1][1]
#             current_quality = path[-1][2]

#             if current_node == target_format:
#                 return path

#             for neighbor, (steps, quality) in self.graph[current_node].items():
#                 # Avoid backtracking and only move forward
#                 if neighbor not in visited:
#                     new_steps = current_steps + steps
#                     new_quality = min(current_quality, quality)
#                     distance_to_goal = self.manhattan_distance((new_steps, new_quality), (0, float("inf")))

#                     # Prioritize paths with fewer steps but consider higher quality nodes
#                     queue.append(path + [(neighbor, new_steps, new_quality)])

#             visited.add(current_node)
#             queue.sort(key=lambda p: (len(p), -p[-1][2]))  # Sort by path length and quality

#         return None


# # (steps, quality)
# graph = {
#     "FormatA": {"FormatB": (1, 8), "FormatC": (2, 9)},
#     "FormatB": {"FormatD": (1, 7)},
#     "FormatC": {"FormatD": (3, 6), "FormatE": (2, 10)},
#     "FormatD": {"TargetFormat": (1, 5)},
#     "FormatE": {"TargetFormat": (1, 9)},
# }

# if __name__ == "__main__":
#     converter = ConversionGraph(graph)
#     path = converter.find_conversion_path("FormatA", "TargetFormat")
#     print("Conversion Path:", path)


#    for model, details in ollama_models.items():
#        nx_graph.add_edges_from(details.available_tasks, key=model, weight=details.size)
#    for model, details in hub_models.items():
#        nx_graph.add_edges_from(details.available_tasks, key=model, weight=details.size)
#     nx_graph = build_conversion_graph()
#     ollama_models = from_ollama_cache()
#     hub_models = from_hf_hub_cache()
#     for model, details in ollama_models.items():
#         nx_graph.add_edges_from(details.available_tasks, label=model, weight=details.size)
#     for model, details in hub_models.items():
#         nx_graph.add_edges_from(details.available_tasks, label=model, weight=details.size)
