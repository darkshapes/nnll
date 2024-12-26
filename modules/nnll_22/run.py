
#// SPDX-License-Identifier: blessing
#// d a r k s h a p e s

import sys
from modules.nnll_22.src import TextEncoderLink, UNetLink, AutoencoderLink


text_link = TextEncoderLink()
unet_link = UNetLink()
autoencoder_link = AutoencoderLink()

target_path_text = sys.argv[0]
shards_text = ["/Users/unauthorized/Downloads/models/text/t5xxl.flux1-dev.diffusers.1of2safetensors.safetensors", "/Users/unauthorized/Downloads/models/text/t5xxl.flux1-dev.diffusers.2of2safetensors.safetensors"]

print(text_link.create_symlink(model_type="clip-l", target_path=target_path_text))
print(text_link.create_symlink(model_type="t5-xxl", target_path=shards_text))
