


from unittest import TestCase
def find_matching_model(file_attributes, data_dict):
    for key, value in data_dict.items():
        if 'neuralnet' == file_attributes['type'] and key != '\x1b[0m':
            return True, ""


def test():
    test = TestCase()
    assert find_matching_model("")
    test.assertEqual(True, False)

test()