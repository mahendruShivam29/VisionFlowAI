"""Sequential orchestrator for the multimodal multi-agent image workflow."""

from __future__ import annotations

import json
from typing import Any

from agents import CritiqueAgent, GenerationAgent, PromptAgent, VisionAgent
from models import WorkflowState

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
else:
    load_dotenv()


class Orchestrator:
    """Fulfills requirement 3.5: coordinates agent communication and failure handling."""

    def __init__(
        self,
        vision_agent: VisionAgent | None = None,
        prompt_agent: PromptAgent | None = None,
        generation_agent: GenerationAgent | None = None,
        critique_agent: CritiqueAgent | None = None,
    ) -> None:
        """Create an orchestrator with independently replaceable agent modules."""

        self.vision_agent = vision_agent or VisionAgent()
        self.prompt_agent = prompt_agent or PromptAgent()
        self.generation_agent = generation_agent or GenerationAgent()
        self.critique_agent = critique_agent or CritiqueAgent()

    def run(
        self,
        image_path: str,
        instruction: str,
        questions: list[str],
        verbose: bool = True,
    ) -> WorkflowState:
        """Run Vision -> Prompt -> Generation -> Critique and return final state."""

        state = WorkflowState(
            original_image_path=image_path,
            user_instruction=instruction,
            visual_questions=questions,
        )

        try:
            state.vision_context = self.vision_agent.execute(image_path, questions)
            self._log("Vision Agent Output", state.vision_context.model_dump(), verbose)
        except Exception as exc:
            message = str(exc)
            state.error_log.append(message)
            self._log("Error", {"message": message}, verbose)
            return state

        try:
            state.prompt_data = self.prompt_agent.execute(instruction, state.vision_context)
            self._log("Prompt Agent Output", state.prompt_data.model_dump(), verbose)
        except Exception as exc:
            message = f"Prompt agent failed; falling back to enhancement mode: {exc}"
            state.error_log.append(message)
            state.prompt_data = self.prompt_agent._execute_with_rules("", state.vision_context)
            self._log("Prompt Agent Output", state.prompt_data.model_dump(), verbose)

        try:
            state.generation_data = self.generation_agent.execute(
                image_path,
                state.prompt_data.refined_prompt,
                state.prompt_data.transformation_mode,
            )
            self._log("Generation Agent Output", state.generation_data.model_dump(), verbose)
        except Exception as exc:
            message = f"Generation failed; retrying with sanitized prompt: {exc}"
            state.error_log.append(message)
            self._log("Generation Retry", {"message": message}, verbose)
            try:
                state.generation_data = self.generation_agent.execute(
                    image_path,
                    state.prompt_data.refined_prompt,
                    state.prompt_data.transformation_mode,
                    simplified_retry=True,
                )
                self._log("Generation Agent Output", state.generation_data.model_dump(), verbose)
            except Exception as retry_exc:
                retry_message = f"Generation retry failed: {retry_exc}"
                state.error_log.append(retry_message)
                self._log("Error", {"message": retry_message}, verbose)
                return state

        try:
            metrics, human_template = self.critique_agent.execute(
                image_path,
                state.generation_data.generated_image_path,
                state.prompt_data.refined_prompt,
                state.prompt_data.transformation_mode,
            )
            state.evaluation_metrics = metrics
            state.human_evaluation_template = human_template
            self._log(
                "Critique Agent Output",
                {
                    **metrics.model_dump(),
                    "human_evaluation_template": human_template.model_dump(),
                },
                verbose,
            )
        except Exception as exc:
            message = f"Critique agent failed: {exc}"
            state.error_log.append(message)
            self._log("Error", {"message": message}, verbose)

        return state

    @staticmethod
    def _log(label: str, payload: dict[str, Any], verbose: bool) -> None:
        """Print intermediate outputs in JSON to show explicit agent communication."""

        if verbose:
            print(f"[{label}]")
            print(json.dumps(payload, indent=2))
