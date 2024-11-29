

# from functools import reduce

# # Example nested dictionary and target values
# reference_map = {
#     'z': {
#         'a': {
#             'x': {
#                 'b1': {'c': 1, 'd': 2},
#                 'b2': {'c': 2, 'd': 2},
#             },
#             'y': {
#                 'b': {'c': 2, 'd': 1}
#             }
#         }
#     }
# }

# file_tags = {'c': 2, 'd': 1}


# def get_nested(nest: dict, keys: list) -> list | int | str:
#     """
#     Fetch values within nested dictionary\n
#     :param nest: `dict` nested dictionary
#     :param key: `list` the path through a nested dictionary
#     :return: the value assigned to the keys indicated in key
#     :rtype: `list`, `int`, or `str`
#     """
#     return reduce(lambda nest, key: nest.get(key, None) if isinstance(nest, dict) else None, keys, nest)


# def get_all_keys_paths(reference_map: dict) -> list:
#     """
#     Construct a path/breadcrump trail through a dictionary\n
#     :param reference_map: `dict` the structure to traverse
#     :rtype: `list`
#     :return: the keys leading up to the next group of values\n
#     note that 'current_layer' could be either `dict` or the contents of a value (`int`, `str`, `list`)
#     """
#     def helper(current_layer, path: list = []) -> list:
#         paths = []
#         for key, value in current_layer.items():
#             new_path = path + [key]
#             if isinstance(value, dict):
#                 paths.extend(helper(value, new_path))
#             else:
#                 paths.append(new_path)
#         return paths

#     return helper(reference_map)


# def flatten(nested_ref_list: list) -> list:
#     """
#     Convert nested list created by `get_all_keys_paths` form expected by `get_nested` and identifier classes\n
#     :param nested_list: `list` nested list to convert\n
#     :return: a flat list of keys pointing to a set of values
#     :rtype: `list`
#     """
#     return reduce(lambda acc, path:
#                   acc + (flatten(path) if isinstance(path, list) else [path]),
#                   nested_ref_list, [])


# def find_value_path(reference_map: dict, file_tags: dict) -> list | None:
#     """
#     Function to find the path with the matching value\n
#     :param reference_map: \n
#     :return: a flat list of keys pointing to a set of values
#     :rtype: `list`
#     """
#     all_paths = get_all_keys_paths(reference_map)

#     for path in all_paths:
#         flattened_path = flatten(path[::-1])
#         value = get_nested(reference_map, flattened_path[:-1])  # Exclude the last key since we're checking the dictionary

#         if isinstance(value, dict) and all(value[key] == value.get(key) for key in value):
#             return flattened_path[:-1] + [flattened_path[-1]]  # Include the last key to complete the path

#     return None


# # Find the matching path
# matching_path = find_value_path(reference_map, file_tags)

# print("Matching path:", matching_path)


# from functools import reduce

# # Example nested dictionary and target values
# nested_dict = {
#     'z': {
#         'a': {
#             'x': {
#                 'b1': {'c': 1, 'd': 2},
#                 'b2': {'c': 2, 'd': 2},
#             },
#             'y': {
#                 'b': {'c': 2, 'd': 1}
#             }
#         }
#     }
# }

# target_values = {'c': 2, 'd': 1}

# # Fetch values within nested dictionary
# def get_nested(nest, keys):
#     return reduce(lambda nest, key: nest.get(key, None) if isinstance(nest, dict) else None, keys, nest)

# # Construct a breadcrumb trail to create a path in the dictionary that can be called with get_nested
# def get_all_keys_paths(nest):
#     def helper(d, path=[]):
#         paths = []
#         for k, v in d.items():
#             new_path = path + [k]
#             if isinstance(v, dict):
#                 paths.extend(helper(v, new_path))
#             else:
#                 paths.append(new_path)
#         return paths
#     return helper(nest)

# def flatten(nested_list: list) -> list:
#     """
#     Convert nested list created by `get_all_keys_paths` form expected by `get_nested` and identifier classes\n
#     :param nested_list: the list to iterate from
#     :return: list of keys leading up to a set of values
#     """
#     return reduce(lambda acc, path:
#                   acc + (flatten(path) if isinstance(path, list) else [path]),
#                   nested_list, [])

# # Function to find the path with the matching value
# def find_value_path(d, target_values):
#     all_paths = get_all_keys_paths(d)

#     for path in all_paths:
#         flattened_path = flatten(path[::-1])
#         value = get_nested(d, flattened_path[:-1])  # Exclude the last key since we're checking the dictionary

#         if isinstance(value, dict) and value == target_values:
#             return flattened_path[:-1] + [flattened_path[-1]]  # Include the last key to complete the path

#     return None

# # Find the matching path
# matching_path = find_value_path(nested_dict, target_values)

# print("Matching path:", matching_path)

# # Fetch values within nested dictionary


# def get_nested(nest, keys): return reduce(lambda nest, key: nest.get(key, None) if isinstance(nest, dict) else None, keys, nest)

# # Construct a breadcrumb trail to create a path in the dictionary that can be called with get_nested


# def get_all_keys_paths(nest):
#     return reduce(lambda acc, key:
#                   acc + ([key] if not isinstance(nest[key], dict) else [[key]] + get_all_keys_paths(nest[key])),
#                   nest.keys(),
#                   [])


# def flatten(nested_list: list) -> list:
#     """
#     Convert nested list created by `get_all_keys_paths` form expected by `get_nested` and identifier classes\n
#     :param nested_list: the list to iterate from
#     :return: list of keys leading up to a set of values
#     """
#     # Convert nested list of keys into a form which can be fed into get_nested
#     # acc is the accumulator
#     # path is our current
#     return reduce(lambda acc, path:
#                   acc + (flatten(path) if isinstance(path, list) else [path]),
#                   nested_list, [])

# # Function to find the path with the matching value


# def find_value_path(d, target_value):
#     all_paths = get_all_keys_paths(d)

#     for path in all_paths:
#         flattened_path = flatten(path[::-1])
#         value = get_nested(d, flattened_path)

#         if value == target_value:
#             return flattened_path

#     return None


# # Known value to find
# known_value = 2
# found_path = find_value_path(nested_dict, known_value)

# print("Found path:", found_path)


# def find_matching_path(nested_dict, target_values, path=[]):
#     # Check if the current dictionary matches the target values
#     if nested_dict == target_values:
#         return path

#     # Iterate through each key in the current dictionary
#     for key, value in nested_dict.items():
#         new_path = path + [key]

#         # If the value is a dictionary, recursively search it
#         if isinstance(value, dict):
#             result = find_matching_path(value, target_values, new_path)

#             # If a match was found, return the path
#             if result:
#                 return result

#     # No match found in this branch, return None
#     return None


# # Example nested dictionary and target values
# nested_dict = {
#     'z': {
#         'a': {
#             'x': {
#                 'b1': {'c': 1, 'd': 2},
#                 'b2': {'c': 2, 'd': 2},
#             },
#             'y': {
#                 'b': {'c': 2, 'd': 1}
#             }
#         }
#     }
# }

# target_values = {'c': 2, 'd': 1}

# # Find the matching path
# matching_path = find_matching_path(nested_dict, target_values)

# print("Matching path:", matching_path)


# # fetch values within nested dictionary
# get_nested = lambda d, keys: reduce(lambda d, key: d.get(key, None) if isinstance(d, dict) else None, keys, d)

# # construct a pathway that can be called with get_nested
# get_all_keys_paths = (lambda d:
#                     reduce(lambda acc, key:
#                         acc + ([key] if not isinstance(d[key], dict) else get_all_keys_paths(d[key]) + [[key]]),
#                         d.keys(),
#                         [])
#                      )

# # convert nested list of keys into a form which can be fed into get_nested or id constructor
# flatten = (lambda nested_list:
#                     reduce(lambda acc, x:
#                         acc + (flatten(x) if isinstance(x, list) else [x]), nested_list, []))

# all_keys_paths = get_all_keys_paths(nested_dict)
# flattened_paths = [flatten(path[::-1]) for path in all_keys_paths]
# print(flattened_paths)


# # Get all key paths

# # Flatten the result to get a list of key paths


# # Example nested dictionary

# # However, our purpose is to match the `2` Value, so we will need to then back track out of `b : { c` and into `y: { c`


# # Define the recursive lambda function using reduce


# all_keys = get_all_keys(nested_dict)
# print(all_keys)  # Output: ['a', 'b', 'c']


def backtrack_depth(depth): return depth[:-1] if depth and depth[-1] in ["block_names", "tensors", "shape", "file_size", "hash"] else depth

# depth - nested level inside the dict
# category - level 1
# classification - level 2
# criteria - level 3


#     def extract_tensor_data(self, source_data_item, id_values):
#         """
#         Extracts shape and key data from the source data.
#         This would extract whatever additional information is needed when a match is found.
#         """
#         TENSOR_TOLERANCE = 4e-2
#         search_items = ["dtype", "shape"]
#         for field_name in search_items:
#             field_value = source_data_item.get(field_name)
#             if field_value:
#                 if isinstance(field_value, list):
#                     field_value = str(field_value)  # We only need the first two numbers of 'shape'
#                 if field_value not in id_values.get(field_name,""):
#                     id_values[field_name] = " ".join([id_values.get(field_name, ""),field_value]).lstrip() # Prevent data duplication

#         return {
#             "tensors": id_values.get("tensors", 0),
#             'shape': id_values.get('shape', 0),
#         }

#     def find_matching_metadata(self, known_values, source_data, id_values, depth=[]):
#         """
#         Recursively traverse the criteria and source_data to check for matches.
#         Track comparisons with id_values, using a list to track the recursion depth.

#         known_values: the original hierarchy of known values from a .json file
#         source_data: the original model state dict
#         id_values: information matching our needs extracted from the source data
#         depth: current level inside the dict
#         """

#         id_values = id_values

#         # Get the dict position indicated in depth
#         get_nested = lambda d, keys: reduce(lambda d, key: d.get(key, None) if isinstance(d, dict) else None, keys, d)
#         # Return the previous position indicated in depth
#         backtrack_depth = lambda depth: depth[:-1] if depth and depth[-1] in ["block_names", "tensors", "shape", "file_size", "hash"] else depth

#         def advance_depth(depth: list, lateral: bool = False) -> list:
#             """
#             Attempts to advance through the tuning dict laterally (to the next key at the same level),
#             failing which retraces vertically (first to the parent level, then the next key at parent level).
#             """
#             if not depth:
#                 return None  # Stop if we've reached the root or there's no further depth
#             parent_dict = get_nested(known_values, depth[:-1]) # Prior depth

#             if not isinstance(parent_dict, dict):    # We look for dicts, and no other types
#                 return None  # Invalid state if we can't get the parent dict

#             parent_keys = list(parent_dict.keys())  #  Keys from above
#             previous_depth = depth[-1] # Current level

#             if previous_depth in parent_keys: # Lateral movement check
#                 current_index = parent_keys.index(previous_depth)

#                 if current_index + 1 < len(parent_keys): #Lateral/next movement, same level
#                     new_depth = depth[:-1]  # Get the parent depth
#                     new_depth.append(parent_keys[current_index + 1])  # Add the next key in sequence
#                     return new_depth

#             if len(depth) > 1: # If no lateral movement is possible, try  vertical/backtracking if there is more than one level
#                 return advance_depth(depth[:-1])  # Move to parent and retry

#             return None  # Traversal complete

#         criteria = get_nested(known_values, depth)
#         if criteria is None:  # Cannot advance, stop
#             return id_values

#         if isinstance(criteria, str): criteria = [criteria]
#         if isinstance(criteria, dict):
#             for name in criteria: # Descend dictionary structure
#                 depth.append(name) # Append the current name to depth list
#                 self.find_matching_metadata(known_values, source_data, id_values, depth)
#                 if depth is None:  # Cannot advance, stop
#                     return id_values
#                 else:
#                     depth = backtrack_depth(depth)
#                     current_depth = get_nested(known_values, depth)
#                     if current_depth[-1] ==
#                         if len(current_depth) == id_values.get(depth[-1],0):
#                             id_values.get("type", set()).add(depth[-1])

#                         known_values[next(iter(known_values), "nn")].pop(depth[-1])
#                     advance_depth(depth)
#                     self.find_matching_metadata(known_values, source_data, id_values, depth)
#             return id_values

#         elif isinstance(criteria, list): # when fed correct datatype, we check for matches
#             for checklist in criteria:
#                 if not isinstance(checklist, list): checklist = [checklist]  # normalize scalar to list
#                 for list_entry in checklist: # the entries to match
#                     if depth[-1] == "hash":
#                          id_values["hash"] = hashlib.sha256(open(id_values["file_name"],'rb').read()).hexdigest()
#                     list_entry = str(list_entry)
#                     if list_entry.startswith("r'"): # Regex conversion
#                         expression = (list_entry
#                             .replace("d+", r"\d+")  # Replace 'd+' with '\d+' for digits
#                             .replace(".", r"\.")    # Escape literal dots with '\.'
#                             .strip("r'")            # Strip the 'r' and quotes from the string
#                         )
#                         regex_entry = re.compile(expression)
#                         match = next((regex_entry.search(k) for k in source_data), False)
#                     else:
#                         match = next((k for k in source_data if list_entry in k), False)
#                     if match: # Found a match, based on keys
#                         previous_depth = depth[-1]
#                         depth = backtrack_depth(depth)
#                         found = depth[-1] if depth else "unknown"    # if theres no header or other circumstances
#                         id_values[found] = id_values.get(found, 0) + 1

#                         shape_key_data = self.extract_tensor_data(source_data[match], id_values)
#                         id_values.update(shape_key_data)

#                         depth.append(previous_depth) #if length depth = 2
#                         depth = advance_depth(depth, lateral=True)
#                         if depth is None:  # Cannot advance, stop
#                             return id_values
#                         self.find_matching_metadata(known_values, source_data, id_values, depth)  # Recurse


#             return id_values
