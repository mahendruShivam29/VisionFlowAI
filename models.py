"""Shared Pydantic schemas for the multimodal multi-agent workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class VisionContext(BaseModel):
    """Fulfills requirement 3.1 by storing caption, objects, scene, and VQA answers."""

    caption: str
    objects: list[str] = Field(default_factory=list)
    scene_description: str
    qa_answers: dict[str, str] = Field(default_factory=dict)


class PromptData(BaseModel):
    """Fulfills requirement 3.2 by storing an enriched prompt and transformation intent."""

    refined_prompt: str
    transformation_mode: Literal["enhancement", "stylization", "variation", "scene_edit"]
    added_visual_details: list[str] = Field(default_factory=list)


class GenerationData(BaseModel):
    """Fulfills requirement 3.3 by storing generated image path and model configuration."""

    generated_image_path: str
    generation_config: dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    """Fulfills requirement 3.4 by storing critique scores and an automatic signal."""

    visual_relevance_score: int = Field(ge=1, le=10)
    prompt_faithfulness_score: int = Field(ge=1, le=10)
    transformation_quality: int = Field(ge=1, le=10)
    clip_similarity_score: float = Field(ge=-1.0, le=1.0)
    critique_summary: str
    decision: Literal["ACCEPT", "REVISE"]


class HumanEvaluationTemplate(BaseModel):
    """Human-grader rubric template required by section 3.4."""

    visual_relevance_1_to_5: int | None = None
    prompt_faithfulness_1_to_5: int | None = None
    transformation_quality_1_to_5: int | None = None
    human_notes: str = ""


class WorkflowState(BaseModel):
    """Shared state object used for explicit agent-to-agent communication."""

    original_image_path: str
    user_instruction: str
    visual_questions: list[str]
    vision_context: VisionContext | None = None
    prompt_data: PromptData | None = None
    generation_data: GenerationData | None = None
    evaluation_metrics: EvaluationMetrics | None = None
    human_evaluation_template: HumanEvaluationTemplate | None = None
    error_log: list[str] = Field(default_factory=list)

    @field_validator("original_image_path")
    @classmethod
    def image_path_must_exist(cls, value: str) -> str:
        """Fail fast on missing images so the orchestrator can halt gracefully."""

        if not Path(value).exists():
            raise ValueError(f"Input image does not exist: {value}")
        return value

    @field_validator("visual_questions")
    @classmethod
    def require_two_questions(cls, value: list[str]) -> list[str]:
        """Enforces the rubric requirement to answer at least two visual questions."""

        cleaned = [question.strip() for question in value if question.strip()]
        if len(cleaned) < 2:
            raise ValueError("At least two visual questions are required.")
        return cleaned

    @property
    def generated_image_path(self) -> str | None:
        """Compatibility accessor matching the SDD state field name."""

        return self.generation_data.generated_image_path if self.generation_data else None

    @property
    def clip_score(self) -> float | None:
        """Compatibility accessor matching the SDD state field name."""

        if not self.evaluation_metrics:
            return None
        return self.evaluation_metrics.clip_similarity_score

    @property
    def critique_decision(self) -> str | None:
        """Compatibility accessor matching the SDD state field name."""

        return self.evaluation_metrics.decision if self.evaluation_metrics else None
