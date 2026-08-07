"""
Unit tests for LawSLM PyTorch Vision Encoder, OCR layout analyzer, and VLM Pipeline.
"""

import unittest
import torch
from PIL import Image
import io
import base64

from slm.vision.encoder import VisionEncoder, VisionPatchEmbedding
from slm.vision.ocr import DocumentVisualAnalyzer
from slm.vision.vlm import LawSLMVisionPipeline
from slm.model.transformer_lm import SLMForCausalLM
from slm.config.model_config import ModelConfig
from slm.tokenizer.bpe import BPETokenizer


class TestVisionPipeline(unittest.TestCase):

    def setUp(self):
        self.config = ModelConfig(vocab_size=200, d_model=64, n_heads=2, n_layers=1)
        self.tokenizer = BPETokenizer()
        self.tokenizer.train_on_texts(["Hello world! Vision test."], vocab_size=200)
        self.config.vocab_size = len(self.tokenizer.vocab)
        self.model = SLMForCausalLM(self.config)

    def test_vision_patch_embedding(self):
        patch_embed = VisionPatchEmbedding(image_size=224, patch_size=16, embed_dim=64)
        dummy_img = torch.randn(2, 3, 224, 224)
        out = patch_embed(dummy_img)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[2], 64)

    def test_vision_encoder_forward(self):
        encoder = VisionEncoder(image_size=224, patch_size=16, d_model=64, n_layers=1, n_heads=2)
        dummy_img = torch.randn(1, 3, 224, 224)
        tokens = encoder(dummy_img)
        self.assertEqual(tokens.shape[0], 1)
        self.assertEqual(tokens.shape[2], 64)

    def test_ocr_and_vlm_pipeline(self):
        analyzer = DocumentVisualAnalyzer()
        img = Image.new("RGB", (300, 400), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        pipeline = LawSLMVisionPipeline(self.model, self.tokenizer)
        res = pipeline.analyze_image_question(image_base64=b64, question="What legal document is this?")

        self.assertIn("question", res)
        self.assertIn("answer", res)
        self.assertIn("ocr", res)
        self.assertEqual(res["ocr"]["dimensions"], "300x400")


if __name__ == "__main__":
    unittest.main()
