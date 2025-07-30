# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->
import pytest

from nnll.model_detect.identity import ModelIdentity

if __name__ == "main":

    @pytest.mark.asyncio
    async def test_model_id_other():
        model_id = ModelIdentity()
        data = await model_id.label_model_repo("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
        assert data == ["info.dit.pixart-sigma-xl-2-1024-ms", "*"]

    @pytest.mark.asyncio
    async def test_model_id_flux():
        model_id = ModelIdentity()
        data = await model_id.label_model_repo("black-forest-labs/FLUX.1-dev")
        assert data == ["info.dit.flux1-dev", "*"]

    @pytest.mark.asyncio
    async def test_model_id_chroma():
        model_id = ModelIdentity()
        data = await model_id.label_model_repo("jack813liu/mlx-chroma")
        assert data == ["info.dit.chroma", "chroma"]

    @pytest.mark.asyncio
    async def test_model_id_jaguar():
        model_id = ModelIdentity()
        data = await model_id.label_model_repo("exdysa/shuttle-jaguar-MLX-Q8")
        assert data is None

    @pytest.mark.asyncio
    async def test_model_id_orsta():
        model_id = ModelIdentity()
        data = await model_id.label_model_repo("exdysa/Orsta-32B-0326-Q4_K_M-GGUF")
        assert data is None

    @pytest.mark.asyncio
    async def test_model_hash_jaguar():
        model_id = ModelIdentity()
        data = await model_id.label_model_layers("exdysa/shuttle-jaguar-MLX-Q8")
        assert data == [["info.stst.t5-v1-1-xxl", "*"], ["info.dit.flux1-schnell", "shuttle-jaguar"], ["info.vit.clip-vit-patch14", "*"], ["info.vae.flux1-schnell", "shuttle-jaguar"]]
