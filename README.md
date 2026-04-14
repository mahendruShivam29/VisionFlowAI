# Multimodal Multi-Agent Image Workflow

This project implements a modular image understanding and generation workflow for the Lab 2 Part 3 rubric. It uses four independent agents connected by a shared `WorkflowState`:

1. `VisionAgent` extracts image caption, objects, scene details, and answers at least two visual questions.
2. `PromptAgent` rewrites the user instruction into a generation-ready prompt while preserving the requested transformation.
3. `GenerationAgent` runs true image-to-image generation through Replicate SDXL when configured, or a deterministic local mock transformation when `MOCK_MODE=true`.
4. `CritiqueAgent` evaluates visual relevance, prompt faithfulness, transformation quality, and an automatic CLIP-style signal.

## Setup

```powershell
pip install -r requirements.txt
```

Optional real-provider configuration:

```powershell
copy .env.example .env
# Fill OPENAI_API_KEY and REPLICATE_API_TOKEN, then set MOCK_MODE=false.
```

The default `MOCK_MODE=true` path is intentional for reliable local grading without API keys or GPU access. The real OpenAI and Replicate code paths are implemented and activate when the required environment variables are present.

## Run the Workflow

```powershell
python test_suite.py
```

The suite runs the three required use cases:

- captioning and visual questions on a busy street image
- cyberpunk stylization on a cat image
- golden-hour enhancement on a dark landscape image

Outputs are written to:

- `outputs/` for generated images
- `human_evaluation_results.csv`
- `human_evaluation_results.md`

Verbose orchestrator output prints JSON after each agent step, making the multi-agent communication explicit for grading.
