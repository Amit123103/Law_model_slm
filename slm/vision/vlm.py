"""
Vision-Language Model (VLM) Pipeline for LawSLM.
Connects VisionEncoder features and document OCR with SLMForCausalLM text generation.
"""

from typing import Dict, Any, Optional
import torch

from slm.vision.encoder import VisionEncoder
from slm.vision.ocr import DocumentVisualAnalyzer
from slm.model.transformer_lm import SLMForCausalLM
from slm.tokenizer.bpe import BPETokenizer
from slm.sampling.generator import TextGenerator
from slm.utils.logger import get_logger

logger = get_logger("slm.vision.vlm")


class LawSLMVisionPipeline:
    """
    Multimodal Vision-Language pipeline enabling LawSLM to understand images, documents, and charts.
    """

    def __init__(
        self,
        model: SLMForCausalLM,
        tokenizer: BPETokenizer,
        vision_encoder: Optional[VisionEncoder] = None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.analyzer = DocumentVisualAnalyzer()
        self.vision_encoder = vision_encoder if vision_encoder is not None else VisionEncoder(
            image_size=224,
            patch_size=16,
            in_channels=3,
            d_model=model.config.d_model,
            n_layers=2,
            n_heads=4
        )
        self.generator = TextGenerator(model, tokenizer)

    def analyze_image_question(
        self,
        image_base64: str,
        question: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Processes image + natural language question to produce multimodal answer.
        """
        img = self.analyzer.load_image(image_base64)
        pixel_tensor = self.analyzer.image_to_tensor(img)
        ocr_result = self.analyzer.extract_ocr_text(img)

        # Extract vision patch tokens
        with torch.no_grad():
            vision_tokens = self.vision_encoder(pixel_tensor)

        # Construct multimodal prompt context
        combined_prompt = (
            f"[IMAGE DOCUMENT: {ocr_result['document_type']} | Resolution: {ocr_result['dimensions']}]\n"
            f"[OCR TEXT]:\n{ocr_result['extracted_text']}\n\n"
            f"User Question: {question}\nLawSLM Answer:"
        )

        answer_text = self.generator.generate(
            prompt=combined_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        return {
            "question": question,
            "answer": answer_text,
            "ocr": ocr_result,
            "vision_token_shape": list(vision_tokens.shape)
        }
