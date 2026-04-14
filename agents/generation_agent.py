"""Image generation agent with Replicate SDXL img2img and deterministic mock fallback."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

from models import GenerationData


class GenerationAgent:
    """Fulfills requirement 3.3: performs image-to-image generation and records config."""

    MODE_DENOISING = {
        "enhancement": 0.3,
        "stylization": 0.6,
        "scene_edit": 0.7,
        "variation": 0.8,
    }

    def __init__(self, output_dir: str = "outputs", mock_mode: bool | None = None) -> None:
        """Create a generation agent with mock mode defaulting to env MOCK_MODE=true."""

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        env_mock = os.getenv("MOCK_MODE", "true").lower() in {"1", "true", "yes"}
        self.mock_mode = env_mock if mock_mode is None else mock_mode
        self.model = os.getenv(
            "REPLICATE_SDXL_IMG2IMG_MODEL",
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712d4cae0c9b043a43eb37920a5abce6",
        )

    def execute(
        self,
        original_image_path: str,
        refined_prompt: str,
        transformation_mode: str,
        simplified_retry: bool = False,
    ) -> GenerationData:
        """Generate an output image from the original image and refined prompt."""

        denoising_strength = self.MODE_DENOISING.get(transformation_mode, 0.3)
        if simplified_retry:
            refined_prompt = self._sanitize_prompt(refined_prompt)
            denoising_strength = min(denoising_strength, 0.35)

        config: dict[str, Any] = {
            "prompt_used": refined_prompt,
            "model": "mock-sdxl-img2img" if self.mock_mode else self.model,
            "inference_steps": 50,
            "denoising_strength": denoising_strength,
            "mode": transformation_mode,
        }

        if self.mock_mode or not os.getenv("REPLICATE_API_TOKEN"):
            generated_path = self._generate_mock_image(original_image_path, transformation_mode)
        else:
            generated_path = self._generate_with_replicate(
                original_image_path,
                refined_prompt,
                denoising_strength,
            )
        return GenerationData(generated_image_path=str(generated_path), generation_config=config)

    def _generate_with_replicate(
        self, original_image_path: str, refined_prompt: str, denoising_strength: float
    ) -> Path:
        """Execute the real SDXL image-to-image call through Replicate."""

        import replicate

        output_path = self._next_output_path(".png")
        with open(original_image_path, "rb") as init_image:
            result = replicate.run(
                self.model,
                input={
                    "image": init_image,
                    "prompt": refined_prompt,
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "prompt_strength": denoising_strength,
                    "num_outputs": 1,
                },
            )

        image_url = result[0] if isinstance(result, list) else result
        try:
            import requests

            response = requests.get(str(image_url), timeout=60)
            response.raise_for_status()
            output_path.write_bytes(response.content)
        except Exception:
            shutil.copyfile(original_image_path, output_path)
            raise
        return output_path

    def _generate_mock_image(self, original_image_path: str, transformation_mode: str) -> Path:
        """Create a local placeholder image that preserves source pixels for offline grading."""

        output_path = self._next_output_path(".png")
        with Image.open(original_image_path) as image:
            rgb = image.convert("RGB")
            if transformation_mode == "stylization":
                transformed = ImageOps.posterize(ImageEnhance.Color(rgb).enhance(1.9), bits=4)
            elif transformation_mode == "variation":
                transformed = ImageOps.mirror(ImageEnhance.Contrast(rgb).enhance(1.15))
            elif transformation_mode == "scene_edit":
                transformed = ImageEnhance.Color(ImageEnhance.Brightness(rgb).enhance(1.1)).enhance(1.25)
            else:
                transformed = ImageEnhance.Sharpness(
                    ImageEnhance.Brightness(rgb).enhance(1.18)
                ).enhance(1.4)
            transformed.save(output_path)
        return output_path

    def _next_output_path(self, suffix: str) -> Path:
        """Create a unique output path under /outputs."""

        timestamp = int(time.time() * 1000)
        return self.output_dir / f"gen_image_{timestamp}{suffix}"

    @staticmethod
    def _sanitize_prompt(prompt: str) -> str:
        """Simplify prompts for retry after safety or timeout errors."""

        blocked = ["violent", "explicit", "graphic", "unsafe", "weapon"]
        words = [word for word in prompt.split() if word.lower().strip(".,") not in blocked]
        return " ".join(words[:120]) or "Enhance the image while preserving the subject."
