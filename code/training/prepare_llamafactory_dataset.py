#!/usr/bin/env python3
"""Prepare generated UAV ShareGPT data for LLaMA-Factory.

This script copies train.jsonl/eval.jsonl into LLaMA-Factory/data/uav_nav,
rewrites image references to absolute paths, and registers uav_full/uav_eval
in LLaMA-Factory/data/dataset_info.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


DATASET_KEYS = {
    "uav_full": "train.jsonl",
    "uav_eval": "eval.jsonl",
}
ASSISTANT_FORMATS = ("auto", "thought_action", "action_only", "decision_action")
ACTION_LINE_RE = re.compile(r"Action:\s*(STOP|\((-?\d+)\s*,\s*(-?\d+)\))", re.IGNORECASE)
DECISION_LINE_RE = re.compile(r"Decision:\s*(STOP|MOVE)", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_assistant_format(generated_dir: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    config_path = generated_dir / "generation_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        fmt = str(config.get("assistant_format", "thought_action"))
        if fmt in ASSISTANT_FORMATS[1:]:
            return fmt
    return "thought_action"


def rewrite_assistant_value(value: str, assistant_format: str, src_jsonl: Path, row_idx: int) -> str:
    text = str(value).strip()
    if assistant_format == "thought_action":
        return text
    action_match = None
    for line in reversed(text.splitlines()):
        action_match = ACTION_LINE_RE.search(line)
        if action_match:
            break
    if action_match is None:
        raise ValueError(f"{src_jsonl}:{row_idx} missing parsable Action line in assistant response")
    action_value = action_match.group(1)
    if assistant_format == "action_only":
        return f"Action: {action_value}"

    decision_match = None
    for line in reversed(text.splitlines()):
        decision_match = DECISION_LINE_RE.search(line)
        if decision_match:
            break
    inferred_decision = "STOP" if action_value.upper() == "STOP" else "MOVE"
    decision = decision_match.group(1).upper() if decision_match else inferred_decision
    if decision != inferred_decision:
        raise ValueError(
            f"{src_jsonl}:{row_idx} Decision={decision} conflicts with Action={action_value}"
        )
    return f"Decision: {decision}\nAction: {action_value}"


def resolve_image_ref(image_ref: str, jsonl_dir: Path, generated_dir: Path) -> str:
    p = Path(image_ref)
    if p.is_absolute():
        resolved = p
    else:
        candidates = [
            (jsonl_dir / p).resolve(),
            (generated_dir.parent / p).resolve(),
            p.resolve(),
        ]
        resolved = None
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                "Image path does not exist: "
                f"{image_ref} -> tried {[str(candidate) for candidate in candidates]}"
            )
    if not resolved.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_ref} -> {resolved}")
    return resolved.as_posix()


def rewrite_rows_with_absolute_images(
    src_jsonl: Path,
    generated_dir: Path,
    assistant_format: str,
) -> list[dict]:
    rows = load_jsonl(src_jsonl)
    jsonl_dir = src_jsonl.parent.resolve()
    for row_idx, row in enumerate(rows, start=1):
        images = row.get("images")
        if not isinstance(images, list):
            raise ValueError(f"{src_jsonl}:{row_idx} missing list field: images")

        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"{src_jsonl}:{row_idx} missing list field: conversations")

        human_text = ""
        for msg in conversations:
            if isinstance(msg, dict) and msg.get("from") == "human":
                human_text = str(msg.get("value", ""))
                break
        if human_text.count("<image>") != len(images):
            raise ValueError(
                f"{src_jsonl}:{row_idx} has {human_text.count('<image>')} <image> tokens "
                f"but {len(images)} image paths."
            )

        row["images"] = [
            resolve_image_ref(str(img), jsonl_dir, generated_dir) for img in images
        ]
        for msg in conversations:
            if isinstance(msg, dict) and msg.get("from") == "gpt":
                msg["value"] = rewrite_assistant_value(
                    str(msg.get("value", "")),
                    assistant_format=assistant_format,
                    src_jsonl=src_jsonl,
                    row_idx=row_idx,
                )
    return rows


def update_dataset_info(dataset_info_path: Path, data_subdir: str) -> None:
    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    columns = {"messages": "conversations", "images": "images"}
    for key, file_name in DATASET_KEYS.items():
        dataset_info[key] = {
            "file_name": f"{data_subdir}/{file_name}",
            "formatting": "sharegpt",
            "columns": columns,
        }

    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")


def copy_sidecar_if_exists(generated_dir: Path, target_dir: Path, name: str) -> None:
    src = generated_dir / name
    if src.exists():
        shutil.copy2(src, target_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register UAV JSONL data for LLaMA-Factory.")
    parser.add_argument(
        "--generated_dir",
        required=True,
        help="Directory containing generated train.jsonl and eval.jsonl.",
    )
    parser.add_argument(
        "--llamafactory_dir",
        required=True,
        help="LLaMA-Factory repository root.",
    )
    parser.add_argument(
        "--data_subdir",
        default="uav_nav",
        help="Subdirectory under LLaMA-Factory/data used to store rewritten JSONL files.",
    )
    parser.add_argument(
        "--assistant_format",
        default="auto",
        choices=ASSISTANT_FORMATS,
        help="Assistant label format. 'auto' follows generation_config.json when available.",
    )
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir).expanduser().resolve()
    llamafactory_dir = Path(args.llamafactory_dir).expanduser().resolve()
    lf_data_dir = llamafactory_dir / "data"
    target_dir = lf_data_dir / args.data_subdir

    if not generated_dir.exists():
        raise FileNotFoundError(f"generated_dir does not exist: {generated_dir}")
    if not llamafactory_dir.exists():
        raise FileNotFoundError(f"llamafactory_dir does not exist: {llamafactory_dir}")
    assistant_format = resolve_assistant_format(generated_dir, args.assistant_format)

    for file_name in DATASET_KEYS.values():
        if not (generated_dir / file_name).exists():
            raise FileNotFoundError(f"missing generated file: {generated_dir / file_name}")

    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in DATASET_KEYS.values():
        src = generated_dir / file_name
        rows = rewrite_rows_with_absolute_images(
            src,
            generated_dir,
            assistant_format=assistant_format,
        )
        dst = target_dir / file_name
        write_jsonl(rows, dst)
        print(f"[prepare] wrote {len(rows)} rows: {dst}")

    for sidecar in (
        "image_ids_train.txt",
        "image_ids_val.txt",
        "train_manifest.jsonl",
        "eval_manifest.jsonl",
    ):
        copy_sidecar_if_exists(generated_dir, target_dir, sidecar)

    dataset_info_path = lf_data_dir / "dataset_info.json"
    update_dataset_info(dataset_info_path, args.data_subdir)
    print(f"[prepare] assistant_format={assistant_format}")
    print(f"[prepare] registered uav_full/uav_eval in {dataset_info_path}")
    print("[prepare] done")


if __name__ == "__main__":
    main()
