### <!-- // /*  SPDX-License-Identifier: blessing) */ -->
### <!-- // /*  d a r k s h a p e s */ -->


from nnll_01 import debug_monitor, info_message as nfo


def hash_layers(path: str):
    import os
    from nnll_04 import ModelTool
    from nnll_44 import compute_hash_for
    from pathlib import Path

    model_tool = ModelTool()
    nfo(path)
    # nfo("{")
    for each in os.listdir(os.path.normpath(Path(path))):
        if Path(each).suffix.lower() == ".safetensors":
            state_dict = model_tool.read_metadata_from(os.path.join(path, each))
            hash_value = compute_hash_for(text_stream=str(state_dict))
            nfo(f"'{hash_value}' : '{each}'")
    # nfo("}")


@debug_monitor
def main():
    """Parse arguments to feed to dict header reader"""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Output the hash of a state dict from all model files at [path] to the console", epilog="Example: nnll-hash '~/Downloads/models/images'")
    parser.add_argument("path", help="Path to directory where files should be analyzed. (default .)", default=".")
    args = parser.parse_args()

    hash_layers(args.path)


if __name__ == "__main__":
    main()

# 0f742c03f5ec009baa8a1548834f24bb9e859b9261856cad6848b6f4ee1a3d7b, artium_v20.safetensors
# 0f742c03f5ec009baa8a1548834f24bb9e859b9261856cad6848b6f4ee1a3d7b, brixlAMustInYour_v5EndOfTheLine.safetensors
# 20d47474da0714979e543b6f21bd12be5b5f721119c4277f364a29e329e931b9, Fluximus20primeV10.kvyR.safetensors
# 20d47474da0714979e543b6f21bd12be5b5f721119c4277f364a29e329e931b9, ichWillMeinSteilFLUX.IMvF.safetensors
# 31164c11db41b007f15c94651a8b1fa4d24097c18782d20fabe13c59ee07aa3a, animagineXL40_v40.safetensors
# 31164c11db41b007f15c94651a8b1fa4d24097c18782d20fabe13c59ee07aa3a, luma20A320PDXL20VAE.5COZ.safetensors
# c4a8d365e7fe07c6dbdd52be922aa6dc23215142342e3e7f8f967f1a123a6982, cozylustrij.OBVE.safetensors
# c4a8d365e7fe07c6dbdd52be922aa6dc23215142342e3e7f8f967f1a123a6982, illustreijlv2.m3sq.safetensors
# d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16, 2dnPony_v2.safetensors
# d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16, cashmoneyAnime_niji.safetensors
# d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16, ponyFaetality_v11.safetensors


# 176f01c1e240fd1510752aaba85c914f2a71e4f557ef75be42fe7518a4cbf890, RealHybridPonyXL.safetensors
# 20143e5445fcbc34c4f3e9608c0ed5b89f2ccc607f1e1bc95cb3d34f3bca99ff, artiusSd35LargeTurbo.grVP.safetensors
# 20d79b91c9190ead70f110f23aa7aaa23eefb4b0dec5bc8e3cc55f1d310c0483, midgardponyv32bf16.IZ78.safetensors
# 24fa0e9bb4994e7f9b262a152fc2665492a097b15576c32fbf6ccf87ebc3f513, moonmixAnimeEdition_v10Pruned.safetensors
# 291238d76c575e06aad6fcaf7d905887fa5e79c723cc5506786519d118e28058, shuttle-3-diffusion.safetensors
# 34dff8d98898baa0f10e71943e56b588cc114253b0d2f1051f3ce7a8a45fee0b, playground-v2.5.diffusers.unet.safetensors
# 36bb43a1e4904994a226d9ff64a561e42d5ed90bb2856d8a26313c6f11000c60, sd35FusionV1Fp8AIO.26bs.safetensors
# 385695bb5b49c52f45818901ee0c095cd6035ed44b0904ea80e789658c932f46, d35Fusion8StepsMergeFull_v1UNET.safetensors
# 45c56e663cf535d2ac2fdb4a12561be365646324f3c4674e770c38ecf5e40050, serenityFP8SD35LargeInclClips_v10.safetensors
# 4a1f2b8234fa4336e263842e042d42e8d64d8a4d3941d9c0c78366b50303950c, hunyuan-1.2.diffusers.safetensors
# 55c56c46ee8817a322aa20fd6dd0c90c69c1ddc6415b64da9ce1f6c32a1f6f5e, ichWilllMEINSTEIL_v10.safetensors
# 56b1ccd89b0d6ab658048aa34d659788b6ed663f13ef566f4b11bccef590b9da, playground-v2.5.diffusers.unet.fp16.safetensors
# 585555ceb76ce58efb7650fd613f8d0e648c15ed193fc23d012aec82a3ba540c, poltergeistIllustrious.lI8i.safetensors
# 7813ebe7cbaf33ea5222cd1d826902b5b4c726d4c24e66c314de29240f75008f, openflux1-v0.1.0-fp8.safetensors
# 79d2bfe93a2ac037cdc59ccb5576e32d00d75d4741fba49fc7e82b9724928216, flux.1.vae.diffusers.safetensors
# 9c2722241a368683554a22782bb9b74900da9cf31b9d2b439f390fba8a395af2, hellaineMixPDXL_v45.safetensors
# a1673b090421fecc6bdfdd485e9fa643cb4963902b2f1eab5b0cf4c95863f441, cosxl.safetensors
# ad8763121f98e28bc4a3d5a8b494c1e8f385f14abe92fc0ca5e4ab3191f3a881, flux1-dev.safetensors
# bff32fdf327b28dddc32e113aa4f2ce65f7a6a3c1c25dc6c3a8b326e92e66e4d, lumina_2.safetensors
# c0ca51fdea051fcd042bf4b56d32e1e8bb9525a921f2e197f370f101e90527f0, lumina-next-sft-diffusers.safetensors
# cb99fe4d9c2bc89062066e799494d0d8f2bbd20861ec3ebeacc5b3c7e177a707, aZovyaPhotoreal_v1Ultra.safetensors
# ced0e5c6ce95c4bbe38fe074c630cf1fa237ee98d00c1cdc4895bba45e8bd959, ponyRealism_v22MainVAE.safetensors
# d3990941477cde17033e454eefdc7282aea2efe99d51173d23bddcccb5f793fe, CounterfeitV30_v30.safetensors
# d4813e9f984aa76cb4ac9bf0972d55442923292d276e97e95cb2f49a57227843, playground-v2.5-1024px-aesthetic.fp16.safetensors
# e4d1f327a83c372276d99861a37511af313d9e7335f710f58342eb122bb04f4b, mystic-fp8.safetensors
# ef5c9cd1ebe6e3be5e8b1347eca0a6f0b138986c71220a7f1c2c14f29d01beed, flux1-schnell.safetensors
# f15aa739d3e4ee000e83d21cab019ccc52d2953ca97f022dc350f5f245480a12, hybrid-sdxl-700m.safetensors
# fe2e9edf7e3923a80e64c2552139d8bae926cc3b028ca4773573a6ba60e67c20, playground-v2.5-1024px-aesthetic.safetensors


# 62a5ab1b5fdfa4fedb32323841298c6effe1af25be94a8583350b0a7641503ef xl
# 31164c11db41b007f15c94651a8b1fa4d24097c18782d20fabe13c59ee07aa3a animage
# d4fc7682a4ea9f2dfa0133fafb068f03fdb479158a58260dcaa24dcf33608c16 pony
# c4a8d365e7fe07c6dbdd52be922aa6dc23215142342e3e7f8f967f1a123a6982 il

# ad8763121f98e28bc4a3d5a8b494c1e8f385f14abe92fc0ca5e4ab3191f3a881 flux dev
# 20d47474da0714979e543b6f21bd12be5b5f721119c4277f364a29e329e931b9

# 8c2e5bc99bc89290254142469411db66cb2ca2b89b129cd2f6982b30e26bd465, sd3.5 large
# 8c2e5bc99bc89290254142469411db66cb2ca2b89b129cd2f6982b30e26bd465


# 14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443 flux hybrid
# 14d0e1b573023deb5a4feaddf85ebca10ab2abf3452c433e2e3ae93acb216443

# 117225c0e91423746114b23d3e409708ad55c90ff52b21fa7a1c5105d2e935a5, PixartXL-2-1024-ms.diffusers.safetensors
# 987f3c2ff5d399191e5fd7dd7b1f1f285c197dc8124ad77f05cde7f2fb677a3c, Pixart-Sigma-XL-2-2k-ms.diffusers.safetensors


# 2240ae134a3b983abf45200c198f07e3d8068012fbbd2f658bbaa1fd6a0629c0, lumina-next-sft-diffusers.vae.safetensors
# 2240ae134a3b983abf45200c198f07e3d8068012fbbd2f658bbaa1fd6a0629c0, playground-v2.5.diffusers.vae.safetensors
# 35641f65ad7ea600cb931dcab556f7503279f1d8d99eda170fe7976d48502a2a, auraflow.vae.diffusers.fp16.safetensors
# 35641f65ad7ea600cb931dcab556f7503279f1d8d99eda170fe7976d48502a2a, playground-v2.5.diffusers.vae.fp16.safetensors
