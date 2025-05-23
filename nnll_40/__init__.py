### <!-- // /*  SPDX-License-Identifier: MPL-2.0  */ -->
### <!-- // /*  d a r k s h a p e s */ -->


# pylint: disable=import-outside-toplevel

# parse metadata into tag?
from nnll_01 import debug_monitor


@debug_monitor
def create_model_tag(pulled_keys: dict) -> dict:
    from nnll_07 import Domain, Architecture, Implementation, Compatibility

    domain_ml = Domain("model")  # create the domain only when we know its a model
    if "unknown" in pulled_keys:
        domain_ml = Domain("dev")  # For unrecognized models,

    arch_found = Architecture(pulled_keys.get("architecture"))
    model_type = pulled_keys["model_type"]
    pulled_keys.pop("architecture")
    pulled_keys.pop("model_type")
    impl_sdxl = Implementation(model_type)
    compat = Compatibility(model_type, **pulled_keys)
    impl_sdxl.add_compat(compat.component_name, compat)
    arch_found.add_impl(impl_sdxl.implementation, impl_sdxl)
    domain_ml.add_arch(arch_found.architecture, arch_found)
    index_tag = domain_ml.to_dict()

    label = domain_ml.to_dict()
    print(label)


# def create_model_tag(model_header,metadata_dict):
#         parse_file = parse_model_header(model_header)
#         reconstructed_file_path = os.path.join(disk_path,each_file)
#         attribute_dict = metadata_dict | {"disk_path": reconstructed_file_path}
#         file_metadata = parse_file | attribute_dict
#         index_tag = create_model_tag(file_metadata)
#
