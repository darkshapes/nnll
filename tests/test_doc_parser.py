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
        from diffusers.pipelines.blip_diffusion.pipeline_blip_diffusion import EXAMPLE_DOC_STRING

        result = parse_docs(EXAMPLE_DOC_STRING)
        self.assertEqual(result.pipe_class, "BlipDiffusionPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "Salesforce/blipdiffusion")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_pia(self):
        from diffusers.pipelines.pia.pipeline_pia import EXAMPLE_DOC_STRING

        result = parse_docs(EXAMPLE_DOC_STRING)
        self.assertEqual(result.pipe_class, "PIAPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "openmmlab/PIA-condition-adapter")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_animatediff_xl(self):
        from diffusers.pipelines.animatediff.pipeline_animatediff_sdxl import EXAMPLE_DOC_STRING

        result = parse_docs(EXAMPLE_DOC_STRING)
        self.assertEqual(result.pipe_class, "AnimateDiffSDXLPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "a-r-r-o-w/animatediff-motion-adapter-sdxl-beta")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_animatediff_controlnet(self):
        from diffusers.pipelines.animatediff.pipeline_animatediff_controlnet import EXAMPLE_DOC_STRING

        result = parse_docs(EXAMPLE_DOC_STRING)
        # TODO : This ought to return control net data but its missing in the docstring

        # self.assertEqual(result.pipe_class, "ControlNetModel")  # pipe_class
        # self.assertEqual(result.pipe_repo, "lllyasviel/ControlNet-v1-1")  # repo_path
        # self.assertIsNone(result.staged_class)  # staged_class
        # self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_consistency(self):
        from diffusers.pipelines.consistency_models.pipeline_consistency_models import EXAMPLE_DOC_STRING

        result = parse_docs(EXAMPLE_DOC_STRING)
        self.assertEqual(result.pipe_class, "ConsistencyModelPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "openai/diffusers-cd_imagenet64_l2")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_pixart_sigma(self):
        from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import EXAMPLE_DOC_STRING

        result = parse_docs(EXAMPLE_DOC_STRING)
        self.assertEqual(result.pipe_class, "PixArtSigmaPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")  # repo_path
        self.assertIsNone(result.staged_class)  # staged_class
        self.assertIsNone(result.staged_repo)  # staged_repo

    def test_parse_cascade(self):
        from diffusers.pipelines.stable_cascade.pipeline_stable_cascade import EXAMPLE_DOC_STRING

        result = parse_docs(EXAMPLE_DOC_STRING)
        self.assertEqual(result.pipe_class, "StableCascadePriorPipeline")  # pipe_class
        self.assertEqual(result.pipe_repo, "stabilityai/stable-cascade-prior")  # repo_path
        self.assertEqual(result.staged_class, "StableCascadeDecoderPipeline")  # staged_class
        self.assertEqual(result.staged_repo, "stabilityai/stable-cascade")  # staged_repo

    def test_parse_xl(self):
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import EXAMPLE_DOC_STRING
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import EXAMPLE_DOC_STRING as EXAMPLE_DOC_STRING_INPAINT

        doc_strings = [
            EXAMPLE_DOC_STRING,
            EXAMPLE_DOC_STRING_INPAINT,
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
