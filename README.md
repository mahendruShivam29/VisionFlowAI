# Multimodal Multi-Agent Image Workflow

This project implements a modular image understanding and generation workflow for the Lab 2 Part 3 rubric. It uses four independent agents connected by a shared `WorkflowState`:

1. `VisionAgent` extracts image caption, objects, scene details, and answers at least two visual questions.
2. `PromptAgent` rewrites the user instruction into a generation-ready prompt while preserving the requested transformation.
3. `GenerationAgent` runs true image-to-image generation through Replicate SDXL when configured, or a deterministic local mock transformation when `MOCK_MODE=true`.
4. `CritiqueAgent` evaluates visual relevance, prompt faithfulness, transformation quality, and an automatic CLIP-style signal.

The project has two entry points:

- `test_suite.py`: a rubric/demo runner that creates three sample images and validates all required cases.
- `run_workflow.py`: a real-user CLI for running the pipeline on any local image.

## Fresh Setup

Use Python 3.10 or newer. The project was written as plain Python modules, so no package build step is required.

### 1. Clone or Open the Project

```powershell
cd path\to\Part-3
```

Confirm the expected files are present:

```powershell
dir
```

You should see files such as `agents/`, `models.py`, `orchestrator.py`, `test_suite.py`, `requirements.txt`, and `README.md`.

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `py` is not available, use:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies include:

- `pydantic` for structured workflow state and agent outputs
- `Pillow` for local sample images and mock image transformations
- `openai` for the real vision/text model path
- `replicate` and `requests` for the real SDXL image-to-image path
- `torch` and `transformers` for optional CLIP evaluation

### 4. Choose Runtime Mode

The default path is offline-friendly mock mode. It requires no API keys, no GPU, and still runs the full multi-agent workflow end to end.

Create a local environment file:

```powershell
copy .env.example .env
```

Leave this setting for local grading without paid services:

```text
MOCK_MODE=true
CLIP_ALLOW_DOWNLOAD=false
```

With `MOCK_MODE=true`, the generation agent creates deterministic transformed images locally under `outputs/`. The real SDXL image-to-image code remains implemented, but it is not called.

## Run End to End

### Option A: Run the Rubric Demo

Run the required three-case test suite:

```powershell
python test_suite.py
```

The suite creates local sample images and runs:

1. Captioning and visual questions on a busy street image.
2. Cyberpunk stylization on a cat image.
3. Golden-hour enhancement on a dark landscape image.

During execution, the orchestrator prints JSON after each agent step:

```text
[Vision Agent Output]
[Prompt Agent Output]
[Generation Agent Output]
[Critique Agent Output]
```

This console output demonstrates the required multi-agent communication sequence:

```text
Vision Understanding -> Prompt Engineering -> Image Generation -> Critique/Evaluation
```

`CritiqueAgent` uses CLIP automatically when the model is already cached locally. Set `CLIP_ALLOW_DOWNLOAD=true` to allow HuggingFace model downloads; otherwise it uses a deterministic fallback signal so offline grading remains reliable.

### Option B: Run On Your Own Image

Use `run_workflow.py` when a real person wants to transform their own image.

Example in mock/offline mode:

```powershell
python run_workflow.py `
  --image "C:\path\to\your\image.jpg" `
  --instruction "Make this look like a cyberpunk digital painting" `
  --question "What is the main subject?" `
  --question "What is the background setting?"
```

The command requires:

- `--image`: path to an existing local image
- `--instruction`: what transformation the user wants
- `--question`: visual questions for the vision agent; pass at least two

By default, the CLI writes the full final state to:

```text
workflow_result.json
```

To choose a different result file:

```powershell
python run_workflow.py `
  --image "C:\path\to\your\image.jpg" `
  --instruction "Enhance the lighting and make colors warmer" `
  --question "Is this indoors or outdoors?" `
  --question "Are there any people?" `
  --output-json "my_result.json"
```

The generated image path is printed at the end of the run and is also stored inside the JSON result.

## Expected Outputs

After a successful run, these files/directories are produced:

- `test_assets/`: generated local input images for the three use cases
- `outputs/`: generated output images
- `human_evaluation_results.csv`: machine-readable evaluation table
- `human_evaluation_results.md`: human-readable evaluation table

These generated artifacts are intentionally ignored by git.

## Optional: Run With Real Providers

To use OpenAI for vision and prompt refinement, and Replicate SDXL for true hosted image-to-image generation:

1. Edit `.env`.
2. Fill in the API keys.
3. Set `MOCK_MODE=false`.

Example:

```text
OPENAI_API_KEY=sk-...
REPLICATE_API_TOKEN=r8_...
MOCK_MODE=false
OPENAI_VISION_MODEL=gpt-4o
OPENAI_TEXT_MODEL=gpt-4o
```

Then run:

```powershell
python test_suite.py
```

Notes:

- OpenAI is used by `VisionAgent` and `PromptAgent` when `OPENAI_API_KEY` is set.
- Replicate is used by `GenerationAgent` when `REPLICATE_API_TOKEN` is set and `MOCK_MODE=false`.
- Real provider usage may incur API costs.
- Network access is required for API calls.

## Optional: Enable CLIP Downloads

By default:

```text
CLIP_ALLOW_DOWNLOAD=false
```

This makes grading reliable offline. If the CLIP model is already cached locally, it can still be used. To allow HuggingFace downloads:

```text
CLIP_ALLOW_DOWNLOAD=true
```

The model used is `openai/clip-vit-base-patch32`.

## Run a Single Custom Example

Developers can also call the orchestrator directly from a short Python script:

```python
from orchestrator import Orchestrator

state = Orchestrator().run(
    image_path="path/to/image.png",
    instruction="Make this look like a watercolor painting.",
    questions=[
        "What is the main subject?",
        "What is the setting?",
    ],
    verbose=True,
)

print(state.model_dump())
```

The input image must exist and the question list must contain at least two questions.

## Troubleshooting

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

If `python` is not found on Windows, install Python from python.org and check "Add python.exe to PATH", or use the Python Launcher command `py`.

If dependency installation fails, upgrade pip and retry:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If CLIP downloads are slow or fail, keep `CLIP_ALLOW_DOWNLOAD=false`. The project will still produce an automatic fallback score and complete the workflow.

If real image generation fails due to missing API keys, set:

```text
MOCK_MODE=true
```

Then rerun:

```powershell
python test_suite.py
```
