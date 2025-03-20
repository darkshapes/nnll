

import PIL


def upscale_image(image_file:str) -> PIL:
    from tensor_device import first_available
    from spandrel import ImageModelDescriptor, ModelLoader
    import torch

    device = first_available()
    model = ModelLoader().load_from_file(r"path/to/model.pth")
    assert isinstance(model, ImageModelDescriptor)
    model.to(device).eval()

    def process(image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(image)

    process(image_file)