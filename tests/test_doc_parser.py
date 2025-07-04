import unittest
from nnll.mir.doc_parser import parse_docs


class TestDocParser(unittest.TestCase):
    def test_parse_simple_case(self):
        doc_string = """
            >>> pipe = MyPipeline.from_pretrained("model/repo")
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "MyPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "model/repo")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_with_variable_resolution(self):
        doc_string = """
            model_id = "custom/model"
            >>> pipe = MyPipeline.from_pretrained(model_id)
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "MyPipeline")
        self.assertEqual(result.pipe_repo, "custom/model")

    def test_parse_staged_case(self):
        doc_string = """
            >>> pipe = MyPipeline.from_pretrained("model/repo")
            >>> prior_pipe = PriorPipeline.from_pretrain("prior/repo")
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "MyPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "model/repo")  # repo_path
        self.assertEqual(result.staged_class, "PriorPipeline")  # staged_class
        self.assertEqual(result.staged_repo, "prior/repo")  # staged_repo

    def test_parse_no_match(self):
        doc_string = """
            >>> something_else = SomeClass.do_something()
        """
        result = parse_docs(doc_string)
        self.assertIsNone(result)  # pipe_class

    def test_parse_multiline_doc(self):
        doc_string = """
            # model_id_or_path = "another/repo"
            >>> pipe_prior = PriorPipeline.from_pretrain(model_id_or_path)
            >>> pipeline = MyPipeline.from_pretrained("repo/path")
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "MyPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "repo/path")  # repo_path
        self.assertEqual(result.staged_class, "PriorPipeline")  # staged_class
        self.assertEqual(result.staged_repo, "another/repo")  # staged_repo

    def test_parse_blip(self):
        doc_string = """
        Examples:
            ```py
            >>> from diffusers.pipelines import BlipDiffusionPipeline
            >>> from diffusers.utils import load_image
            >>> import torch

            >>> blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(
            ...     "Salesforce/blipdiffusion", torch_dtype=torch.float16
            ... ).to("cuda")


            >>> cond_subject = "dog"
            >>> tgt_subject = "dog"
            >>> text_prompt_input = "swimming underwater"

            >>> cond_image = load_image(
            ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"
            ... )
            >>> guidance_scale = 7.5
            >>> num_inference_steps = 25
            >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"


            >>> output = blip_diffusion_pipe(
            ...     text_prompt_input,
            ...     cond_image,
            ...     cond_subject,
            ...     tgt_subject,
            ...     guidance_scale=guidance_scale,
            ...     num_inference_steps=num_inference_steps,
            ...     neg_prompt=negative_prompt,
            ...     height=512,
            ...     width=512,
            ... ).images
            >>> output[0].save("image.png")
            ```
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "BlipDiffusionPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "Salesforce/blipdiffusion")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_pia(self):
        doc_string = """
            Examples:
                ```py
                >>> import torch
                >>> from diffusers import EulerDiscreteScheduler, MotionAdapter, PIAPipeline
                >>> from diffusers.utils import export_to_gif, load_image

                >>> adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter")
                >>> pipe = PIAPipeline.from_pretrained(
                ...     "SG161222/Realistic_Vision_V6.0_B1_noVAE", motion_adapter=adapter, torch_dtype=torch.float16
                ... )

                >>> pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
                >>> image = load_image(
                ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
                ... )
                >>> image = image.resize((512, 512))
                >>> prompt = "cat in a hat"
                >>> negative_prompt = "wrong white balance, dark, sketches, worst quality, low quality, deformed, distorted"
                >>> generator = torch.Generator("cpu").manual_seed(0)
                >>> output = pipe(image=image, prompt=prompt, negative_prompt=negative_prompt, generator=generator)
                >>> frames = output.frames[0]
                >>> export_to_gif(frames, "pia-animation.gif")
                ```
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "PIAPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "openmmlab/PIA-condition-adapter")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_animatediff_xl(self):
        doc_string = """
        Examples:
            ```py
            >>> import torch
            >>> from diffusers.models import MotionAdapter
            >>> from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
            >>> from diffusers.utils import export_to_gif

            >>> adapter = MotionAdapter.from_pretrained(
            ...     "a-r-r-o-w/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16
            ... )

            >>> model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            >>> scheduler = DDIMScheduler.from_pretrained(
            ...     model_id,
            ...     subfolder="scheduler",
            ...     clip_sample=False,
            ...     timestep_spacing="linspace",
            ...     beta_schedule="linear",
            ...     steps_offset=1,
            ... )
            >>> pipe = AnimateDiffSDXLPipeline.from_pretrained(
            ...     model_id,
            ...     motion_adapter=adapter,
            ...     scheduler=scheduler,
            ...     torch_dtype=torch.float16,
            ...     variant="fp16",
            ... ).to("cuda")

            >>> # enable memory savings
            >>> pipe.enable_vae_slicing()
            >>> pipe.enable_vae_tiling()

            >>> output = pipe(
            ...     prompt="a panda surfing in the ocean, realistic, high quality",
            ...     negative_prompt="low quality, worst quality",
            ...     num_inference_steps=20,
            ...     guidance_scale=8,
            ...     width=1024,
            ...     height=1024,
            ...     num_frames=16,
            ... )

            >>> frames = output.frames[0]
            >>> export_to_gif(frames, "animation.gif")
            ```
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "AnimateDiffSDXLPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "a-r-r-o-w/animatediff-motion-adapter-sdxl-beta")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_consistency(self):
        doc_string = """
            Examples:
                ```py
                >>> import torch

                >>> from diffusers import ConsistencyModelPipeline

                >>> device = "cuda"
                >>> # Load the cd_imagenet64_l2 checkpoint.
                >>> model_id_or_path = "openai/diffusers-cd_imagenet64_l2"
                >>> pipe = ConsistencyModelPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
                >>> pipe.to(device)

                >>> # Onestep Sampling
                >>> image = pipe(num_inference_steps=1).images[0]
                >>> image.save("cd_imagenet64_l2_onestep_sample.png")

                >>> # Onestep sampling, class-conditional image generation
                >>> # ImageNet-64 class label 145 corresponds to king penguins
                >>> image = pipe(num_inference_steps=1, class_labels=145).images[0]
                >>> image.save("cd_imagenet64_l2_onestep_sample_penguin.png")

                >>> # Multistep sampling, class-conditional image generation
                >>> # Timesteps can be explicitly specified; the particular timesteps below are from the original GitHub repo:
                >>> # https://github.com/openai/consistency_models/blob/main/scripts/launch.sh#L77
                >>> image = pipe(num_inference_steps=None, timesteps=[22, 0], class_labels=145).images[0]
                >>> image.save("cd_imagenet64_l2_multistep_sample_penguin.png")
                ```
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "ConsistencyModelPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "openai/diffusers-cd_imagenet64_l2")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_pixart_sigma(self):
        doc_string = """
            Examples:
                ```py
                >>> import torch
                >>> from diffusers import PixArtSigmaPipeline

                >>> # You can replace the checkpoint id with "PixArt-alpha/PixArt-Sigma-XL-2-512-MS" too.
                >>> pipe = PixArtSigmaPipeline.from_pretrained(
                ...     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
                ... )
                >>> # Enable memory optimizations.
                >>> # pipe.enable_model_cpu_offload()

                >>> prompt = "A small cactus with a happy face in the Sahara desert."
                >>> image = pipe(prompt).images[0]
                ```
        """
        result = parse_docs(doc_string)
        self.assertEqual(result.pipe_class, "PixArtSigmaPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_cascade(self):
        doc_string = """
            Examples:
                ```py
                >>> import torch
                >>> from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline

                >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(
                ...     "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16
                ... ).to("cuda")
                >>> gen_pipe = StableCascadeDecoderPipeline.from_pretrain(
                ...     "stabilityai/stable-cascade", torch_dtype=torch.float16
                ... ).to("cuda")

                >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
                >>> prior_output = pipe(prompt)
                >>> images = gen_pipe(prior_output.image_embeddings, prompt=prompt)
                ```
        """
        result = parse_docs(doc_string)
        # self.assertEqual(result.pipe_class, "StableCascadeDecoderPipeline")  # pipe_class
        # self.assertEqual(result.pipe_repo, "stabilityai/stable-cascade")  # repo_path
        # self.assertEqual(result.staged_class, "StableCascadePriorPipeline")  # staged_class
        # self.assertEqual(result.staged_repo, "stabilityai/stable-cascade-prior")  # staged_repo

        self.assertEqual(result.pipe_class, "StableCascadePriorPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "stabilityai/stable-cascade-prior")  # repo_path
        self.assertEqual(result.staged_class, "StableCascadeDecoderPipeline")  # staged_class
        self.assertEqual(result.staged_repo, "stabilityai/stable-cascade")  # staged_repo

    def test_parse_xl(self):
        doc_strings = [
            """
            Examples:
                ```py
                >>> import torch
                >>> from diffusers import StableDiffusionXLPipeline

                >>> pipe = StableDiffusionXLPipeline.from_pretrained(
                ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
                ... )
                >>> pipe = pipe.to("cuda")

                >>> prompt = "a photo of an astronaut riding a horse on mars"
                >>> image = pipe(prompt).images[0]
                ```
        """,
            """
            Examples:
                ```py
                >>> import torch
                >>> from diffusers import StableDiffusionXLInpaintPipeline
                >>> from diffusers.utils import load_image

                >>> pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                ...     "stabilityai/stable-diffusion-xl-base-1.0",
                ...     torch_dtype=torch.float16,
                ...     variant="fp16",
                ...     use_safetensors=True,
                ... )
                >>> pipe.to("cuda")

                >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
                >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

                >>> init_image = load_image(img_url).convert("RGB")
                >>> mask_image = load_image(mask_url).convert("RGB")

                >>> prompt = "A majestic tiger sitting on a bench"
                >>> image = pipe(
                ...     prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80
                ... ).images[0]
                ```
        """,
        ]
        result = []
        for doc in doc_strings:
            result.append(parse_docs(doc))

        self.assertEqual(result[0].pipe_class, "StableDiffusionXLPipeline")  # pipe_class
        self.assertEqual(result[0].pipe_repo, "stabilityai/stable-diffusion-xl-base-1.0")  # repo_path
        self.assertIsNone(result[0].staged_class)  # staged_class
        self.assertIsNone(result[0].staged_repo)  # staged_repo
        self.assertEqual(result[1].pipe_class, "StableDiffusionXLInpaintPipeline")  # pipe_class
        self.assertEqual(result[1].pipe_repo, "stabilityai/stable-diffusion-xl-base-1.0")  # repo_path
        self.assertIsNone(result[1].staged_class)  # staged_class
        self.assertIsNone(result[1].staged_repo)  # staged_repo


if __name__ == "__main__":
    unittest.main()
