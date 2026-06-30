#!/usr/bin/env python3
"""Create a concrete LLaMA-Factory YAML config with an overridden output_dir."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


OUTPUT_DIR_PATTERN = re.compile(r"^(?P<indent>\s*)output_dir:\s*(?P<value>.+?)\s*$")
RESUME_PATTERN = re.compile(r"^(?P<indent>\s*)resume_from_checkpoint:\s*(?P<value>.*?)\s*$")


def render_train_config(template_text: str, output_dir: str, resume_from_checkpoint: str | None = None) -> str:
    lines = template_text.splitlines()
    output_replaced = False
    resume_replaced = False
    rendered: list[str] = []

    for line in lines:
        output_match = OUTPUT_DIR_PATTERN.match(line)
        if output_match and not output_replaced:
            indent = output_match.group("indent")
            rendered.append(f'{indent}output_dir: "{output_dir}"')
            output_replaced = True
            continue
        resume_match = RESUME_PATTERN.match(line)
        if resume_match and not resume_replaced:
            indent = resume_match.group("indent")
            value = f'"{resume_from_checkpoint}"' if resume_from_checkpoint else "null"
            rendered.append(f"{indent}resume_from_checkpoint: {value}")
            resume_replaced = True
            continue
        rendered.append(line)

    if not output_replaced:
        raise ValueError("Template YAML is missing a top-level output_dir entry.")
    if resume_from_checkpoint and not resume_replaced:
        rendered.append(f'resume_from_checkpoint: "{resume_from_checkpoint}"')

    return "\n".join(rendered) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a train YAML with a concrete output_dir.")
    parser.add_argument("--template", required=True, help="Path to the source YAML template.")
    parser.add_argument("--output_dir", required=True, help="Resolved training output directory.")
    parser.add_argument("--destination", required=True, help="Path to write the materialized YAML.")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Optional checkpoint path to resume from.")
    args = parser.parse_args()

    template_path = Path(args.template).expanduser().resolve()
    destination_path = Path(args.destination).expanduser().resolve()

    rendered = render_train_config(
        template_path.read_text(encoding="utf-8"),
        args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(rendered, encoding="utf-8")
    print(f"[train] wrote config: {destination_path}")


if __name__ == "__main__":
    main()
