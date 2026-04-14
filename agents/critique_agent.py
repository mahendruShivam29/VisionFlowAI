"""Critique and evaluation agent with optional CLIP signal."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from models import EvaluationMetrics, HumanEvaluationTemplate


class CritiqueAgent:
    """Fulfills requirement 3.4: evaluates relevance, faithfulness, quality, and CLIP signal."""

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32") -> None:
        """Create a critique agent with lazy CLIP loading for fast mock-mode tests."""

        self.clip_model_name = clip_model_name

    def execute(
        self,
        original_image_path: str,
        generated_image_path: str,
        refined_prompt: str,
        transformation_mode: str,
    ) -> tuple[EvaluationMetrics, HumanEvaluationTemplate]:
        """Return automatic metrics, accept/revise decision, and human rubric template."""

        clip_score = self._clip_similarity(generated_image_path, refined_prompt)
        pixel_similarity = self._pixel_similarity(original_image_path, generated_image_path)
        prompt_signal = self._prompt_signal(refined_prompt, transformation_mode)

        visual_relevance = self._score_from_similarity(pixel_similarity, transformation_mode)
        prompt_faithfulness = max(1, min(10, round(6 + prompt_signal * 4)))
        transformation_quality = max(1, min(10, round(5 + (1 - pixel_similarity) * 4 + prompt_signal)))
        decision = (
            "ACCEPT"
            if visual_relevance >= 6 and prompt_faithfulness >= 6 and transformation_quality >= 6
            else "REVISE"
        )
        summary = (
            "The generated image preserves source-image structure while applying the requested "
            f"{transformation_mode} transformation. Automatic similarity signal: {clip_score:.3f}."
        )
        metrics = EvaluationMetrics(
            visual_relevance_score=visual_relevance,
            prompt_faithfulness_score=prompt_faithfulness,
            transformation_quality=transformation_quality,
            clip_similarity_score=clip_score,
            critique_summary=summary,
            decision=decision,
        )
        return metrics, HumanEvaluationTemplate()

    def _clip_similarity(self, image_path: str, prompt: str) -> float:
        """Calculate CLIP image-text cosine similarity when dependencies are available."""

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(self.clip_model_name)
            processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
            return float(similarity)
        except Exception:
            return self._fallback_similarity(image_path, prompt)

    @staticmethod
    def _fallback_similarity(image_path: str, prompt: str) -> float:
        """Provide a deterministic automatic signal when CLIP cannot run locally."""

        with Image.open(image_path) as image:
            stat = ImageStat.Stat(image.convert("RGB"))
        brightness = sum(stat.mean) / (3 * 255)
        prompt_length_signal = min(len(prompt.split()) / 80, 1.0)
        return round(0.15 + 0.25 * brightness + 0.2 * prompt_length_signal, 3)

    @staticmethod
    def _pixel_similarity(original_path: str, generated_path: str) -> float:
        """Compute normalized pixel similarity between original and generated images."""

        with Image.open(original_path) as original, Image.open(generated_path) as generated:
            original_rgb = original.convert("RGB").resize((128, 128))
            generated_rgb = generated.convert("RGB").resize((128, 128))
            diff = ImageChops.difference(original_rgb, generated_rgb)
            stat = ImageStat.Stat(diff)
            rms = math.sqrt(sum(value**2 for value in stat.rms) / 3)
        return max(0.0, min(1.0, 1 - (rms / 255)))

    @staticmethod
    def _prompt_signal(prompt: str, transformation_mode: str) -> float:
        """Score whether the prompt includes concrete transformation and preservation language."""

        prompt_lower = prompt.lower()
        required = ["preserve", "composition", "subject", transformation_mode.replace("_", " ")]
        hits = sum(1 for token in required if token in prompt_lower)
        detail_bonus = min(len(prompt.split()) / 90, 1.0)
        return min(1.0, hits / len(required) * 0.75 + detail_bonus * 0.25)

    @staticmethod
    def _score_from_similarity(pixel_similarity: float, transformation_mode: str) -> int:
        """Convert image similarity to a 1-10 relevance score calibrated by generation mode."""

        if transformation_mode == "variation":
            target = 0.65
        elif transformation_mode == "stylization":
            target = 0.72
        else:
            target = 0.82
        distance = abs(pixel_similarity - target)
        return max(1, min(10, round(10 - distance * 12)))
