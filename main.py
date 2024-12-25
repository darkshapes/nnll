#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s


from setuptools import setup

def main():
    # setup()
    from index import trace_catalog_file_structure, populate_module_index, write_index_to_file
    active_directories = trace_catalog_file_structure()
    module_index       = populate_module_index(active_directories)
    write_index_to_file(module_index)

    from _version import version
    print(f"{version=}")

if __name__ == "__main__":
    main()