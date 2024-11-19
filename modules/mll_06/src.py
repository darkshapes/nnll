
# def find_keys(d, val):
#     return list(key for key, value in d.items() if value == val)


# numeric = {
#     'File1': [10, 1],
#     'File2': 11,
#     'File3': {10, 1}
# }
# print(find_keys(numeric, {10, 1}))

def compare_dicts(known, metadata):
    for k, v in known.items():
        if isinstance(v, dict):
            if not compare_dicts(v, metadata.get(k, {})):
                return False
        elif metadata.get(k) != v:
            return False
    return True
