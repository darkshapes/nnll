# # import networkx as nx
# import networkx as nx
# from typing import Dict, Tuple
# # from pydantic import BaseModel, Field


# # class ConversionValue(BaseModel):
# #     """(output_medium, more_output_medium)"""

# #     values: Field()


# # class ConversionMap(BaseModel):
# #     """{input_medium: (output_medium, more_output_medium)"""

# #     input: Dict[str, ConversionValue]


# VALID_CONVERSIONS = {
#     "text": ("text", "speech", "image", "music"),
#     "image": ("text", "image", "upscale_image", "latent_image"),
#     "music": ("text", "latent_music"),
#     "speech": ("text"),
#     "upscale_image": (),  # No reverse from upscale_image to anything
#     "latent_image": ("image"),
#     "latent_music": ("music",),
# }


# def create_graph():
#     """create coordinate pair from valid conversions then deploy as a graph"""
#     # adjacent_pairs = [(key, item) for key, value in VALID_CONVERSIONS.input.items() for item in (value.values if isinstance(value.values, tuple) else ())]
#     adjacent_pairs = [(key, item) for key, value in VALID_CONVERSIONS.items() for item in (value if isinstance(value, tuple) else ())]
#     nx_graph = nx.DiGraph(adjacent_pairs)
#     return nx_graph


# NX_GRAPH = create_graph()


# # from networkx import convert_node_labels_to_integers
# # from textual.app import App, ComposeResult
# # from textual_plot import HiResMode, PlotWidget


# # mllama (vllm), text-to-image, text-generation

# # 2. Add nodes for each unique data type and model combination (e.g., `['text']`, `['image']`, etc.) and edges representing the transformations between them using models.


# # # Define a function to add edges based on input-output pairs
# # def add_model_edges(G, input_type, model_dict, output_type):
# #     for model_name, model_path in model_dict.items():
# #         G.add_edge(str(input_type), str(output_type), model_name=model_name, model_path=model_path)


# # # Add the specified paths to your graph
# # add_model_edges(G, ["text"], {"model1": "path1"}, ["text"])
# # add_model_edges(G, ["text", "image"], {"model2": "path2"}, ["text"])
# # add_model_edges(G, ["image"], {"model3": "path3"}, ["text"])
# # add_model_edges(G, ["text"], {"model4": "path4"}, ["image"])
# # add_model_edges(G, ["speech"], {"model5": "path5"}, ["text"])
# # add_model_edges(G, ["text"], {"model6": "path6"}, ["speech"])

# # # Example: Find all paths from ['text'] to ['image']
# # paths = list(nx.all_simple_paths(G, str(["text"]), str(["image"])))
# # print(paths)

# # # Access model details along the path
# # for i in range(len(paths[0]) - 1):
# #     edge_data = G.get_edge_data(paths[0][i], paths[0][i + 1])
# #     print(f"Model: {edge_data['model_name']}, Path: {edge_data['model_path']}")


# # class MinimalApp(App[None]):
# #     def compose(self) -> ComposeResult:
# #         yield PlotWidget()

# #     def on_mount(self) -> None:
# #         plot = self.query_one(PlotWidget)
# #         plot.set_xticks([])
# #         plot.set_yticks([])
# #         plot.set_xlabel("")
# #         plot.set_ylabel("")
# #         plot.set_styles("background: #1f1f1f;")
# #         plot.scatter(*X, hires_mode=HiResMode.BRAILLE, marker_style="purple")
# #         G = nx.Graph
# #         G = nx.Graph(day="Friday")
# #         G.graph["day"] = "Monday"
# #         G.add_node(1, time="5pm")
# #         G.add_nodes_from([3], time="2pm")
# #         G.nodes[1]["room"] = 714
# #         X = convert_node_labels_to_integers(G)
# #         plot.set_xticks([])
# #         plot.set_yticks([])
# #         plot.set_xlabel("")
# #         plot.set_ylabel("")
# #         plot.set_styles("background: #1f1f1f;")
# #         plot.scatter(*X, hires_mode=HiResMode.BRAILLE, marker_style="purple")


# # MinimalApp().run()


# # node would be format
# # edge would be conversion model
# # 'vllm'
# # 'text-generation'

# # input                                                          #output
# # node                             #edge                          #node
# # {['text']}                    { model name: model path} }     { ['text']] }
# # [['text', 'image']:           { model name: model path} }     { ['text']  }
# # {['image']:                   { model name: model path} }     { ['text']  }
# # {['text']:                    { model name: model path} }     { ['image'] }
# # {['speech']:                  { model name: model path} }     { ['text']  }
# # {['text']:                    { model name: model path} }     { ['speech']}


# # bidirectional_shortest_path(G, source, target)

# # G.add_edges_from[(2, 3, {"weight": 3.1415})] #add edge with attribute
# # G.add_nodes_from([(4, {"color": "red"}), (5, {"color": "green"})]) #add node with attribute
# # H = nx.path_graph(10)
# # G.add_nodes_from(H)  # import one graph into another
# # G.add_node(H)  # the entire graph as a node
# # G.clear()

# # class ConversionGraph:
# #     def __init__(self, graph):
# #         self.graph = graph  # Graph where keys are formats, values are dict of {format: (steps, quality)}

# #     def manhattan_distance(self, node1, node2):
# #         return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

# #     def find_conversion_path(self, initial_format, target_format):
# #         # Check if direct conversion is possible
# #         if target_format in self.graph[initial_format]:
# #             return [(initial_format, target_format)]

# #         # Initialize variables for pathfinding
# #         queue = [[(initial_format, 0, float("inf"))]]  # (format, steps, quality)
# #         visited = set()

# #         while queue:
# #             path = queue.pop(0)
# #             current_node = path[-1][0]
# #             current_steps = path[-1][1]
# #             current_quality = path[-1][2]

# #             if current_node == target_format:
# #                 return path

# #             for neighbor, (steps, quality) in self.graph[current_node].items():
# #                 # Avoid backtracking and only move forward
# #                 if neighbor not in visited:
# #                     new_steps = current_steps + steps
# #                     new_quality = min(current_quality, quality)
# #                     distance_to_goal = self.manhattan_distance((new_steps, new_quality), (0, float("inf")))

# #                     # Prioritize paths with fewer steps but consider higher quality nodes
# #                     queue.append(path + [(neighbor, new_steps, new_quality)])

# #             visited.add(current_node)
# #             queue.sort(key=lambda p: (len(p), -p[-1][2]))  # Sort by path length and quality

# #         return None


# # # (steps, quality)
# # graph = {
# #     "FormatA": {"FormatB": (1, 8), "FormatC": (2, 9)},
# #     "FormatB": {"FormatD": (1, 7)},
# #     "FormatC": {"FormatD": (3, 6), "FormatE": (2, 10)},
# #     "FormatD": {"TargetFormat": (1, 5)},
# #     "FormatE": {"TargetFormat": (1, 9)},
# # }

# # if __name__ == "__main__":
# #     converter = ConversionGraph(graph)
# #     path = converter.find_conversion_path("FormatA", "TargetFormat")
# #     print("Conversion Path:", path)
