"""Three-use-case validation suite for the multimodal multi-agent workflow."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw

from orchestrator import Orchestrator


def create_sample_images(asset_dir: Path) -> dict[str, Path]:
    """Create deterministic local test images for the three required rubric scenarios."""

    asset_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "busy_street": asset_dir / "busy_street.png",
        "cat": asset_dir / "cat.png",
        "dark_landscape": asset_dir / "dark_landscape.png",
    }
    _draw_street(images["busy_street"])
    _draw_cat(images["cat"])
    _draw_landscape(images["dark_landscape"])
    return images


def run_test_suite(verbose: bool = True) -> list[dict[str, str]]:
    """Run captioning/VQA, stylization, and enhancement examples end to end."""

    images = create_sample_images(Path("test_assets"))
    cases = [
        {
            "case_id": "A",
            "name": "Captioning and VQA focus",
            "image": images["busy_street"],
            "instruction": "Just recreate this exactly.",
            "questions": ["How many cars are visible?", "What is the weather like?"],
        },
        {
            "case_id": "B",
            "name": "Style-guided transformation",
            "image": images["cat"],
            "instruction": "Make this look like a cyberpunk digital painting.",
            "questions": ["What animal is this?", "What color is its fur?"],
        },
        {
            "case_id": "C",
            "name": "Prompt-based enhancement",
            "image": images["dark_landscape"],
            "instruction": "Enhance the lighting, make it golden hour, and make the colors pop.",
            "questions": ["Is this indoors or outdoors?", "Are there any people?"],
        },
    ]

    orchestrator = Orchestrator()
    rows: list[dict[str, str]] = []
    for case in cases:
        state = orchestrator.run(
            str(case["image"]),
            case["instruction"],
            case["questions"],
            verbose=verbose,
        )
        metrics = state.evaluation_metrics
        human_template = state.human_evaluation_template
        rows.append(
            {
                "case_id": case["case_id"],
                "name": case["name"],
                "input_image": str(case["image"]),
                "generated_image": state.generated_image_path or "",
                "decision": state.critique_decision or "",
                "visual_relevance_score": str(metrics.visual_relevance_score if metrics else ""),
                "prompt_faithfulness_score": str(metrics.prompt_faithfulness_score if metrics else ""),
                "transformation_quality": str(metrics.transformation_quality if metrics else ""),
                "clip_similarity_score": str(metrics.clip_similarity_score if metrics else ""),
                "human_visual_relevance_1_to_5": str(
                    human_template.visual_relevance_1_to_5 if human_template else ""
                ),
                "human_prompt_faithfulness_1_to_5": str(
                    human_template.prompt_faithfulness_1_to_5 if human_template else ""
                ),
                "human_transformation_quality_1_to_5": str(
                    human_template.transformation_quality_1_to_5 if human_template else ""
                ),
                "human_notes": human_template.human_notes if human_template else "",
                "errors": " | ".join(state.error_log),
            }
        )
    write_reports(rows)
    return rows


def write_reports(rows: list[dict[str, str]]) -> None:
    """Write CSV and Markdown reports for human graders."""

    csv_path = Path("human_evaluation_results.csv")
    md_path = Path("human_evaluation_results.md")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Human Evaluation Results",
        "",
        "| Case | Name | Generated Image | Decision | Visual | Prompt | Quality | CLIP | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['name']} | {row['generated_image']} | "
            f"{row['decision']} | {row['visual_relevance_score']} | "
            f"{row['prompt_faithfulness_score']} | {row['transformation_quality']} | "
            f"{row['clip_similarity_score']} | {row['human_notes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw_street(path: Path) -> None:
    """Draw a busy street image for use case A."""

    image = Image.new("RGB", (640, 360), "#9ec8e8")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 220, 640, 360), fill="#555555")
    draw.rectangle((0, 170, 640, 225), fill="#7fb069")
    for x in range(30, 620, 110):
        draw.rectangle((x, 110, x + 55, 220), fill="#b9bec8")
        draw.rectangle((x + 10, 125, x + 25, 145), fill="#eaf6ff")
    for x, color in [(70, "#d7263d"), (210, "#f4d35e"), (360, "#1b998b"), (500, "#2e86ab")]:
        draw.rectangle((x, 250, x + 80, 292), fill=color)
        draw.ellipse((x + 10, 285, x + 28, 303), fill="#111111")
        draw.ellipse((x + 55, 285, x + 73, 303), fill="#111111")
    image.save(path)


def _draw_cat(path: Path) -> None:
    """Draw a cat portrait image for use case B."""

    image = Image.new("RGB", (512, 512), "#6f7d8c")
    draw = ImageDraw.Draw(image)
    draw.ellipse((125, 135, 385, 405), fill="#d2a679")
    draw.polygon([(155, 170), (195, 70), (235, 175)], fill="#d2a679")
    draw.polygon([(275, 175), (320, 70), (360, 170)], fill="#d2a679")
    draw.ellipse((205, 240, 235, 270), fill="#1d3557")
    draw.ellipse((285, 240, 315, 270), fill="#1d3557")
    draw.polygon([(255, 280), (240, 305), (270, 305)], fill="#ff8fab")
    for y in (285, 305, 325):
        draw.line((125, y, 220, y - 10), fill="#3d2b1f", width=2)
        draw.line((292, y - 10, 385, y), fill="#3d2b1f", width=2)
    image.save(path)


def _draw_landscape(path: Path) -> None:
    """Draw a dark landscape image for use case C."""

    image = Image.new("RGB", (640, 360), "#101728")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 180), fill="#17203a")
    draw.polygon([(0, 250), (160, 110), (320, 250)], fill="#1f2f24")
    draw.polygon([(230, 250), (445, 95), (640, 250)], fill="#263a2a")
    draw.rectangle((0, 250, 640, 360), fill="#152718")
    draw.ellipse((500, 45, 540, 85), fill="#f0c36a")
    image.save(path)


if __name__ == "__main__":
    run_test_suite(verbose=True)
