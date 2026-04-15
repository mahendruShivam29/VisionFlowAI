"""Command-line entry point for running the workflow on a user's own image."""

from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
USER_SITE = site.getusersitepackages()
if USER_SITE and USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

from orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a custom end-to-end workflow run."""

    parser = argparse.ArgumentParser(
        description="Run the multimodal multi-agent workflow on a custom image."
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--instruction",
        required=True,
        help="User transformation instruction, such as 'make this cyberpunk'.",
    )
    parser.add_argument(
        "--question",
        action="append",
        required=True,
        help="Visual question. Pass this flag at least twice.",
    )
    parser.add_argument(
        "--output-json",
        default="agent_logs/workflow_result.json",
        help="Where to write the final WorkflowState JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose intermediate agent logs.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the orchestrator and persist the final state for a real user."""

    args = parse_args()
    if len(args.question) < 2:
        print("Error: pass --question at least twice.", file=sys.stderr)
        return 2

    state = Orchestrator().run(
        image_path=args.image,
        instruction=args.instruction,
        questions=args.question,
        verbose=not args.quiet,
    )

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(state.model_dump(), indent=2), encoding="utf-8")

    if state.generated_image_path:
        print(f"Generated image: {state.generated_image_path}")
    if state.critique_decision:
        print(f"Decision: {state.critique_decision}")
    print(f"Full result JSON: {output_path}")
    return 0 if not state.error_log else 1


if __name__ == "__main__":
    raise SystemExit(main())
