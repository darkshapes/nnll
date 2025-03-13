# import networkx as nx
# G = nx.Graph(


class ConversionGraph:
    def __init__(self, graph):
        self.graph = graph  # Graph where keys are formats, values are dict of {format: (steps, quality)}

    def manhattan_distance(self, node1, node2):
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    def find_conversion_path(self, initial_format, target_format):
        # Check if direct conversion is possible
        if target_format in self.graph[initial_format]:
            return [(initial_format, target_format)]

        # Initialize variables for pathfinding
        queue = [[(initial_format, 0, float("inf"))]]  # (format, steps, quality)
        visited = set()

        while queue:
            path = queue.pop(0)
            current_node = path[-1][0]
            current_steps = path[-1][1]
            current_quality = path[-1][2]

            if current_node == target_format:
                return path

            for neighbor, (steps, quality) in self.graph[current_node].items():
                # Avoid backtracking and only move forward
                if neighbor not in visited:
                    new_steps = current_steps + steps
                    new_quality = min(current_quality, quality)
                    distance_to_goal = self.manhattan_distance((new_steps, new_quality), (0, float("inf")))

                    # Prioritize paths with fewer steps but consider higher quality nodes
                    queue.append(path + [(neighbor, new_steps, new_quality)])

            visited.add(current_node)
            queue.sort(key=lambda p: (len(p), -p[-1][2]))  # Sort by path length and quality

        return None


# (steps, quality)
graph = {
    "FormatA": {"FormatB": (1, 8), "FormatC": (2, 9)},
    "FormatB": {"FormatD": (1, 7)},
    "FormatC": {"FormatD": (3, 6), "FormatE": (2, 10)},
    "FormatD": {"TargetFormat": (1, 5)},
    "FormatE": {"TargetFormat": (1, 9)},
}

if __name__ == "__main__":
    converter = ConversionGraph(graph)
    path = converter.find_conversion_path("FormatA", "TargetFormat")
    print("Conversion Path:", path)
