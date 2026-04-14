"""Prompt engineering agent for generation-ready transformation prompts."""

from __future__ import annotations

import json
import os

from models import PromptData, VisionContext


class PromptAgent:
    """Fulfills requirement 3.2: rewrites and enriches user instructions with visual context."""

    def __init__(self, model: str | None = None) -> None:
        """Create a prompt agent backed by OpenAI when configured, else rules."""

        self.model = model or os.getenv("OPENAI_TEXT_MODEL", "gpt-4o")

    def execute(self, user_instruction: str, vision_context: VisionContext) -> PromptData:
        """Return a clear refined prompt and preserved transformation mode."""

        if os.getenv("OPENAI_API_KEY"):
            return self._execute_with_openai(user_instruction, vision_context)
        return self._execute_with_rules(user_instruction, vision_context)

    def _execute_with_openai(
        self, user_instruction: str, vision_context: VisionContext
    ) -> PromptData:
        """Call a text LLM to produce the structured prompt object."""

        from openai import OpenAI

        client = OpenAI()
        system = (
            "You are a prompt engineering agent. Return only valid JSON with keys "
            "refined_prompt, transformation_mode, and added_visual_details. "
            "transformation_mode must be one of enhancement, stylization, variation, scene_edit."
        )
        user = {
            "user_instruction": user_instruction,
            "vision_context": vision_context.model_dump(),
        }
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
        )
        return PromptData.model_validate(json.loads(response.choices[0].message.content or "{}"))

    def _execute_with_rules(
        self, user_instruction: str, vision_context: VisionContext
    ) -> PromptData:
        """Produce a strong SDXL-ready prompt without requiring an API key."""

        instruction = user_instruction.strip().rstrip(".")
        mode = self._classify_mode(instruction)
        if not instruction or len(instruction.split()) < 2:
            instruction = "Enhance the image while preserving the original composition"
            mode = "enhancement"

        details = [
            vision_context.caption,
            vision_context.scene_description,
            f"Key visible elements: {', '.join(vision_context.objects[:6])}.",
        ]
        refined = (
            f"{instruction}. Preserve the original composition, subject identity, spatial layout, "
            f"and important objects. Use these visual anchors: {' '.join(details)} "
            "Create a coherent, high-quality image with natural lighting, clean detail, and no "
            "unwanted extra limbs, distorted text, watermarks, or artifacts."
        )
        return PromptData(
            refined_prompt=refined,
            transformation_mode=mode,
            added_visual_details=details,
        )

    @staticmethod
    def _classify_mode(instruction: str) -> str:
        """Classify the intended transformation for denoising and critique choices."""

        text = instruction.lower()
        if any(word in text for word in ["cyberpunk", "painting", "style", "watercolor", "anime"]):
            return "stylization"
        if any(word in text for word in ["enhance", "lighting", "sharpen", "golden hour", "pop"]):
            return "enhancement"
        if any(word in text for word in ["change", "replace", "add", "remove", "edit"]):
            return "scene_edit"
        if any(word in text for word in ["variation", "different", "recreate"]):
            return "variation"
        return "enhancement"
