
def find_matching_model(file_attributes, data_dict):
    """
    Basic dictionary comparison method using conditionals
    `file_attributes` metadata from file under examination
    `data_dict` precollected and labeled values of file metadata
    """
    for key, value in data_dict.items():
        if key == file_attributes.get("neuralnet"):
            if "type_1" in file_attributes:
                if file_attributes["type_1"] in data_dict:
                    for model_name, model_data in data_dict[file_attributes["type_1"]].items():
                        if all(attr in model_data.values() for attr in file_attributes.values()):
                            return True, model_name  # Return both True and the matching model name.
    return False, None
