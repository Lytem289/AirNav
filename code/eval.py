import argparse
import base64
import csv
import json
import math
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2

try:
    from openai import OpenAI
except ImportError:  # Model baselines are optional; random/greedy do not need OpenAI.
    OpenAI = None

from train_data_generate import (
    ACTION_GRID_SIZE_M,
    HEIGHT_BINS,
    LOCAL_SEARCH_ACTION_GRID_SIZE_M,
    LOCAL_SEARCH_MAX_STEP,
    MAX_STEP,
    SCENARIO_KEYS,
    TERMINAL_ACTION_GRID_SIZE_M,
    TERMINAL_MAX_STEP,
    TERMINAL_RADIUS_FACTOR,
    build_instruction,
    build_pid_gsd_index,
    build_pid_to_path_index,
    can_crop_without_padding,
    compatible_height_bins,
    compute_instruction_context_at_start,
    execute_grid_action,
    filter_pid_index,
    limit_pid_index,
    parse_dota_annotation,
    pixel_per_meter,
    relation_guided_search_action,
    sample_episode,
    sample_height_bin,
    search_best_grid_action,
    local_scan_action,
    landmark_relation_context,
    stop_eps_m,
    object_in_fov,
    world_to_local,
)

IMG_SIZE = 336
DEFAULT_MAX_STEPS = 20
DEFAULT_STOP_GRACE_STEPS = 3
DEFAULT_LANDMARK_MAX_DISTANCE_M = 25.0
DEFAULT_LANDMARK_NEARBY_DISTANCE_M = 20.0
DEFAULT_MIN_START_DISTANCE_M = 50.0
DEFAULT_EVAL_SPLIT_FILE = "image_ids_val.txt"


def img_to_b64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("ascii")


def parse_action_detail(text):
    info = {
        "status": "ok",
        "reason": "ok",
        "raw_text": text,
        "contains_stop_token": False,
        "matched_text": None,
        "scaled": False,
        "original_action": None,
        "parsed_action": None,
    }
    if not text:
        info["status"] = "invalid"
        info["reason"] = "empty_response"
        return None, info

    action_matches = list(
        re.finditer(
            r"Action:\s*(STOP|\((-?\d+)\s*,\s*(-?\d+)\))",
            text,
            flags=re.IGNORECASE,
        )
    )
    if action_matches:
        match = action_matches[-1]
        info["matched_text"] = match.group(0)
        if match.group(1).upper() == "STOP":
            info["status"] = "stop"
            info["reason"] = "explicit_stop"
            info["contains_stop_token"] = True
            return None, info
        gx, gy = int(match.group(2)), int(match.group(3))
    else:
        info["contains_stop_token"] = "STOP" in text.upper()
        match = re.search(r"\((-?\d+)\s*,\s*(-?\d+)\)", text)
        if not match:
            info["status"] = "invalid"
            info["reason"] = "no_action_match"
            return None, info
        info["matched_text"] = match.group(0)
        gx, gy = int(match.group(1)), int(match.group(2))

    info["original_action"] = [gx, gy]
    max_abs = max(abs(gx), abs(gy))
    if max_abs > MAX_STEP:
        scale = MAX_STEP / max_abs
        gx = int(round(gx * scale))
        gy = int(round(gy * scale))
        info["scaled"] = True
    if gx == 0 and gy == 0:
        info["status"] = "stop"
        info["reason"] = "zero_action"
        info["parsed_action"] = [0, 0]
        return None, info
    info["parsed_action"] = [gx, gy]
    return (gx, gy), info


def parse_action(text):
    action, _ = parse_action_detail(text)
    return action


class ModelPolicy:
    def __init__(self, name, model, base_url, api_key="EMPTY", temperature=0.0):
        if OpenAI is None:
            raise RuntimeError("openai package is not installed; model policies cannot run.")
        self.name = name
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(self, state):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_to_b64(state["hist_img"])}},
                {"type": "image_url", "image_url": {"url": img_to_b64(state["cur_img"])}},
                {"type": "text", "text": state["prompt"]},
            ],
        }]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=256,
        )
        text = response.choices[0].message.content
        action, parse_info = parse_action_detail(text)
        return action, text, parse_info


def random_policy(state):
    if random.random() < 0.1:
        return None, "Action: STOP", {
            "status": "stop",
            "reason": "policy_stop",
            "raw_text": "Action: STOP",
            "contains_stop_token": True,
            "matched_text": None,
            "scaled": False,
            "original_action": None,
            "parsed_action": None,
        }
    while True:
        gx = random.randint(-MAX_STEP, MAX_STEP)
        gy = random.randint(-MAX_STEP, MAX_STEP)
        if gx != 0 or gy != 0:
            raw = f"Action: ({gx}, {gy})"
            return (gx, gy), raw, {
                "status": "ok",
                "reason": "policy_action",
                "raw_text": raw,
                "contains_stop_token": False,
                "matched_text": raw,
                "scaled": False,
                "original_action": [gx, gy],
                "parsed_action": [gx, gy],
            }


def greedy_policy(state):
    dist_m = math.hypot(
        state["target"]["cx"] - state["x"],
        state["target"]["cy"] - state["y"],
    ) / state["ppm"]
    stop_radius_m = stop_eps_m(state["grid_size_m"])
    target_visible = state.get("target_visible", True)
    landmark_visible = state.get("landmark_visible", False)
    if dist_m <= stop_radius_m and target_visible:
        return None, "Action: STOP", {
            "status": "stop",
            "reason": "policy_stop",
            "raw_text": "Action: STOP",
            "contains_stop_token": True,
            "matched_text": None,
            "scaled": False,
            "original_action": None,
            "parsed_action": None,
        }

    if target_visible:
        terminal_radius_m = max(stop_radius_m, TERMINAL_RADIUS_FACTOR * ACTION_GRID_SIZE_M)
        action_grid_size_m = (
            TERMINAL_ACTION_GRID_SIZE_M if dist_m <= terminal_radius_m else ACTION_GRID_SIZE_M
        )
        candidate_max_step = TERMINAL_MAX_STEP if dist_m <= terminal_radius_m else MAX_STEP
        best = search_best_grid_action(
            state["x"],
            state["y"],
            state["yaw"],
            state["target"],
            ppm=state["ppm"],
            action_grid_size_m=action_grid_size_m,
            stop_radius_m=stop_radius_m,
            candidate_max_step=candidate_max_step,
        )
    elif landmark_visible:
        best = relation_guided_search_action(
            state.get("landmark_local_x_px"),
            state.get("landmark_local_y_px"),
            state.get("landmark_relation_word"),
            state.get("landmark_relation_dist_m"),
            state["yaw"],
            state["ppm"],
            LOCAL_SEARCH_ACTION_GRID_SIZE_M,
            state.get("step", 0),
            candidate_max_step=LOCAL_SEARCH_MAX_STEP,
        )
    else:
        best = local_scan_action(
            state.get("step", 0) + state.get("lost_landmark_steps", 0),
            candidate_max_step=LOCAL_SEARCH_MAX_STEP,
        )
    if best is None:
        return None, "Action: STOP", {
            "status": "stop",
            "reason": "policy_stop",
            "raw_text": "Action: STOP",
            "contains_stop_token": True,
            "matched_text": None,
            "scaled": False,
            "original_action": None,
            "parsed_action": None,
        }
    gx, gy = best["grid_x"], best["grid_y"]
    raw = f"Action: ({gx}, {gy})"
    return (gx, gy), raw, {
        "status": "ok",
        "reason": "policy_action",
        "raw_text": raw,
        "contains_stop_token": False,
        "matched_text": raw,
        "scaled": False,
        "original_action": [gx, gy],
        "parsed_action": [gx, gy],
    }


def crop_uav_view(large_img, current_x, current_y, current_yaw, fov_m, ppm, img_size, out_path):
    crop_size = int(fov_m * ppm)
    extend_size = int(crop_size * 1.414) + 2
    if not can_crop_without_padding(large_img.shape, current_x, current_y, extend_size):
        raise ValueError(
            f"crop would require padding at x={current_x:.1f}, y={current_y:.1f}, extend={extend_size}"
        )

    x1 = int(current_x - extend_size // 2)
    y1 = int(current_y - extend_size // 2)
    x2 = x1 + extend_size
    y2 = y1 + extend_size
    crop = large_img[y1:y2, x1:x2]

    center = (extend_size // 2, extend_size // 2)
    mat = cv2.getRotationMatrix2D(center, -current_yaw, 1.0)
    rotated = cv2.warpAffine(crop, mat, (extend_size, extend_size))
    s = (extend_size - crop_size) // 2
    final = rotated[s:s + crop_size, s:s + crop_size]
    final = cv2.resize(final, (img_size, img_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, final)
    return out_path


def global_grid_displacement(yaw_history, action_history):
    north, east = 0.0, 0.0
    for yaw, (gx, gy) in zip(yaw_history, action_history):
        dist_grid = math.hypot(gx, gy)
        yaw_rad = math.radians(yaw)
        east += dist_grid * math.sin(yaw_rad)
        north += dist_grid * math.cos(yaw_rad)
    return north, east


def eval_ppm_for_entry(entry, label_path):
    if entry.get("real_gsd") is not None:
        return pixel_per_meter(real_gsd=float(entry["real_gsd"]), jitter=False)
    targets, parsed_gsd, parsed_ok = parse_dota_annotation(label_path, entry.get("meta"))
    ext = str(label_path).lower().rsplit(".", 1)[-1]
    if parsed_ok and ext in ("tif", "tiff", "png", "jpg", "jpeg"):
        return pixel_per_meter(real_gsd=parsed_gsd, jitter=False)
    return pixel_per_meter(jitter=False)


def load_pid_list(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def load_generation_config(generated_dir):
    if not generated_dir:
        return {}
    path = Path(generated_dir) / "generation_config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_scenarios_arg(text):
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    for item in items:
        if item not in SCENARIO_KEYS:
            raise ValueError(f"Unknown scenario: {item}")
    return items


def resolve_eval_pid_filter(split, generated_dir=None, eval_pid_list_path=None):
    pid_list_path = None
    if eval_pid_list_path:
        pid_list_path = Path(eval_pid_list_path).expanduser().resolve()
    elif generated_dir and split == "train":
        candidate = Path(generated_dir).expanduser().resolve() / DEFAULT_EVAL_SPLIT_FILE
        if candidate.exists():
            pid_list_path = candidate
    if pid_list_path is None:
        return None, None
    if not pid_list_path.exists():
        raise FileNotFoundError(f"eval pid list does not exist: {pid_list_path}")
    return load_pid_list(pid_list_path), str(pid_list_path)


def resolve_eval_settings(args):
    generated_config = load_generation_config(args.generated_dir)

    landmark_max_distance_m = (
        float(args.landmark_max_distance_m)
        if args.landmark_max_distance_m is not None
        else float(generated_config.get("landmark_max_distance_m", DEFAULT_LANDMARK_MAX_DISTANCE_M))
    )
    landmark_nearby_distance_m = (
        float(args.landmark_nearby_distance_m)
        if args.landmark_nearby_distance_m is not None
        else float(generated_config.get("landmark_nearby_distance_m", DEFAULT_LANDMARK_NEARBY_DISTANCE_M))
    )
    min_start_distance_m = (
        float(args.min_start_distance_m)
        if args.min_start_distance_m is not None
        else float(generated_config.get("min_start_distance_m", DEFAULT_MIN_START_DISTANCE_M))
    )

    if args.scenarios:
        scenario_choices = parse_scenarios_arg(args.scenarios)
    else:
        include_stop = bool(generated_config.get("include_stop_scenario", True))
        scenario_choices = ["A", "B", "C", "D", "E"] if include_stop else ["A", "B", "C", "D"]

    pid_filter, pid_filter_path = resolve_eval_pid_filter(
        split=args.split,
        generated_dir=args.generated_dir,
        eval_pid_list_path=args.eval_pid_list,
    )

    return {
        "generated_config": generated_config,
        "landmark_max_distance_m": landmark_max_distance_m,
        "landmark_nearby_distance_m": landmark_nearby_distance_m,
        "min_start_distance_m": min_start_distance_m,
        "scenario_choices": scenario_choices,
        "pid_filter": pid_filter,
        "pid_filter_path": pid_filter_path,
    }


def simulate_episode(episode, policy_name, policy_fn, output_dir, max_steps=DEFAULT_MAX_STEPS,
                     img_size=IMG_SIZE, save_video=False,
                     stop_grace_steps=DEFAULT_STOP_GRACE_STEPS):
    entry = episode["entry"]
    large_img = cv2.imread(entry["img"])
    if large_img is None:
        raise RuntimeError(f"Failed to read image: {entry['img']}")

    ppm = episode["ppm"]
    x, y, yaw = episode["start_x"], episode["start_y"], episode["start_yaw"]
    target = episode["target"]
    landmark = episode.get("landmark")
    grid_size_m = episode["grid_size_m"]
    fov_m = episode["fov_m"]
    eps_m = stop_eps_m(grid_size_m)
    shortest_path_m = math.hypot(target["cx"] - x, target["cy"] - y) / ppm

    count_in_view, spatial_word = compute_instruction_context_at_start(
        episode["all_targets"], target, x, y, yaw, ppm=ppm, fov_m=fov_m, grid_size_m=grid_size_m
    )
    instruction_text = build_instruction(
        target,
        landmark=landmark,
        spatial_word=spatial_word,
        target_count_in_view=count_in_view,
        ppm=ppm,
    )

    tmp_dir = tempfile.mkdtemp(prefix=f"uav_eval_{policy_name}_", dir=output_dir)
    history_actions = []
    yaw_history = []
    action_history = []
    seen_landmark_once = False
    lost_landmark_steps = 0
    path_length_m = 0.0
    enter_radius_success = False
    enter_radius_step = None
    stop_success = False
    stopped = False
    terminated_reason = None
    rows = []

    try:
        hist_img = crop_uav_view(
            large_img, x, y, yaw, fov_m, ppm, img_size, os.path.join(tmp_dir, "hist_000.jpg")
        )
    except ValueError as exc:
        terminated_reason = "initial_crop_oob"
        rows.append({
            "step": 0,
            "x": x,
            "y": y,
            "yaw": yaw,
            "dist_m": math.hypot(target["cx"] - x, target["cy"] - y) / ppm,
            "action": "STOP",
            "raw": None,
            "parse_status": "invalid",
            "parse_reason": terminated_reason,
            "crop_ok": False,
            "error": str(exc),
        })
        return {
            "policy": policy_name,
            "episode_id": episode["episode_id"],
            "pid": episode["pid"],
            "success": False,
            "stop_success": False,
            "enter_radius_success": False,
            "stopped": False,
            "path_length": 0.0,
            "shortest_path": shortest_path_m,
            "steps": 0,
            "height_m": episode["height_m"],
            "target_class": target["class"],
            "final_dist_m": rows[-1]["dist_m"],
            "terminated_reason": terminated_reason,
            "trace": rows,
        }

    for step in range(max_steps):
        try:
            cur_img = crop_uav_view(
                large_img, x, y, yaw, fov_m, ppm, img_size, os.path.join(tmp_dir, f"cur_{step:03d}.jpg")
            )
        except ValueError as exc:
            terminated_reason = "crop_oob"
            rows.append({
                "step": step,
                "x": x,
                "y": y,
                "yaw": yaw,
                "dist_m": math.hypot(target["cx"] - x, target["cy"] - y) / ppm,
                "action": "STOP",
                "raw": None,
                "parse_status": "invalid",
                "parse_reason": terminated_reason,
                "crop_ok": False,
                "error": str(exc),
            })
            break
        north, east = global_grid_displacement(yaw_history, action_history)
        target_visible = object_in_fov(target, x, y, yaw, ppm, fov_m)
        landmark_visible = False
        landmark_local_x_px = landmark_local_y_px = None
        if landmark is not None:
            landmark_vec_x_px = landmark["cx"] - x
            landmark_vec_y_px = landmark["cy"] - y
            landmark_local_x_px, landmark_local_y_px = world_to_local(
                landmark_vec_x_px,
                landmark_vec_y_px,
                yaw,
            )
            landmark_visible = object_in_fov(landmark, x, y, yaw, ppm, fov_m)
        if landmark_visible:
            seen_landmark_once = True
            lost_landmark_steps = 0
        elif seen_landmark_once:
            lost_landmark_steps += 1
        prompt = (
            f"{instruction_text}\n"
            f"历史动作: {history_actions[-5:]}\n"
            f"累计位移: 向北{north:.1f}格, 向东{east:.1f}格, 已飞{step}步\n"
            f"请输出下一步动作。"
        )
        state = {
            "x": x,
            "y": y,
            "yaw": yaw,
            "target": target,
            "ppm": ppm,
            "grid_size_m": grid_size_m,
            "fov_m": fov_m,
            "hist_img": hist_img,
            "cur_img": cur_img,
            "prompt": prompt,
            "step": step,
            "target_visible": target_visible,
            "landmark_visible": landmark_visible,
            "landmark_local_x_px": landmark_local_x_px,
            "landmark_local_y_px": landmark_local_y_px,
            "landmark_relation_word": episode.get("landmark_relation_word"),
            "landmark_relation_dist_m": episode.get("landmark_relation_dist_m"),
            "lost_landmark_steps": lost_landmark_steps,
        }
        dist_m = math.hypot(target["cx"] - x, target["cy"] - y) / ppm
        terminal_radius_m = max(eps_m, TERMINAL_RADIUS_FACTOR * ACTION_GRID_SIZE_M)
        terminal_zone = bool(target_visible and dist_m <= terminal_radius_m)
        should_stop = bool(target_visible and dist_m <= eps_m)
        if dist_m <= eps_m and target_visible:
            enter_radius_success = True
            if enter_radius_step is None:
                enter_radius_step = step
            terminated_reason = terminated_reason or "success_radius_reached"

        action, raw, parse_info = policy_fn(state)
        rows.append({
            "step": step,
            "x": x,
            "y": y,
            "yaw": yaw,
            "dist_m": dist_m,
            "action": "STOP" if action is None else str(action),
            "raw": raw,
            "parse_status": parse_info["status"],
            "parse_reason": parse_info["reason"],
            "matched_text": parse_info["matched_text"],
            "parsed_action": parse_info["parsed_action"],
            "original_action": parse_info["original_action"],
            "scaled_action": parse_info["scaled"],
            "target_visible": target_visible,
            "landmark_visible": landmark_visible,
            "terminal_zone": terminal_zone,
            "should_stop": should_stop,
            "crop_ok": True,
        })

        if action is None:
            stopped = True
            stop_success = dist_m <= eps_m and target_visible
            if stop_success:
                enter_radius_success = True
            terminated_reason = "stop_success" if stop_success else parse_info["reason"]
            break

        gx, gy = action
        action_grid_size_m = (
            TERMINAL_ACTION_GRID_SIZE_M if enter_radius_success
            else TERMINAL_ACTION_GRID_SIZE_M if target_visible and dist_m <= terminal_radius_m
            else ACTION_GRID_SIZE_M if target_visible
            else LOCAL_SEARCH_ACTION_GRID_SIZE_M
        )
        x, y, yaw, move_dist_m = execute_grid_action(
            x, y, yaw, gx, gy, ppm, action_grid_size_m,
            noise_yaw_std=0.0, noise_dist_std=0.0, noise_lateral_std=0.0,
        )
        path_length_m += math.hypot(gx, gy) * action_grid_size_m
        hist_img = cur_img
        history_actions.append((gx, gy))
        history_actions = history_actions[-5:]
        yaw_history.append(yaw)
        action_history.append((gx, gy))

        if math.hypot(target["cx"] - x, target["cy"] - y) / ppm <= eps_m:
            enter_radius_success = True
            if enter_radius_step is None:
                enter_radius_step = step + 1
            terminated_reason = terminated_reason or "success_radius_reached"
        if (
            enter_radius_step is not None
            and not stop_success
            and (step - enter_radius_step + 1) >= max(1, int(stop_grace_steps))
        ):
            terminated_reason = "missed_stop"
            break
        if not can_crop_without_padding(large_img.shape, x, y, int(fov_m * ppm * 1.414) + 2):
            terminated_reason = "next_crop_oob"
            break

    return {
        "policy": policy_name,
        "episode_id": episode["episode_id"],
        "pid": episode["pid"],
        "success": bool(enter_radius_success),
        "enter_radius_success": bool(enter_radius_success),
        "stop_success": bool(stop_success),
        "stopped": bool(stopped),
        "path_length": path_length_m,
        "shortest_path": shortest_path_m,
        "steps": len(rows),
        "height_m": episode["height_m"],
        "target_class": target["class"],
        "final_dist_m": rows[-1]["dist_m"] if rows else shortest_path_m,
        "terminated_reason": terminated_reason or (
            "max_steps" if len(rows) >= max_steps and not stop_success else None
        ),
        "trace": rows,
    }


def calculate_metrics(results):
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "sr": 0.0,
            "enter_radius_sr": 0.0,
            "stop_success_sr": 0.0,
            "spl": 0.0,
            "avg_steps": 0.0,
        }, {}
    succ = sum(1 for r in results if r["success"])
    stop_succ = sum(1 for r in results if r.get("stop_success"))
    spl = 0.0
    step_sum = 0
    buckets = defaultdict(list)
    for r in results:
        buckets[r["height_m"]].append(r)
        if r["success"]:
            spl += r["shortest_path"] / max(r["path_length"], r["shortest_path"], 1e-6)
            step_sum += r["steps"]
    overall = {
        "n": n,
        "sr": succ / n,
        "enter_radius_sr": succ / n,
        "stop_success_sr": stop_succ / n,
        "spl": spl / n,
        "avg_steps": step_sum / succ if succ else 0.0,
    }
    by_height = {}
    for h, items in sorted(buckets.items()):
        h_succ = sum(1 for r in items if r["success"])
        h_stop_succ = sum(1 for r in items if r.get("stop_success"))
        h_spl = 0.0
        h_steps = 0
        for r in items:
            if r["success"]:
                h_spl += r["shortest_path"] / max(r["path_length"], r["shortest_path"], 1e-6)
                h_steps += r["steps"]
        by_height[h] = {
            "n": len(items),
            "sr": h_succ / len(items),
            "enter_radius_sr": h_succ / len(items),
            "stop_success_sr": h_stop_succ / len(items),
            "spl": h_spl / len(items),
            "avg_steps": h_steps / h_succ if h_succ else 0.0,
        }
    return overall, by_height


def summarize_policy_debug(results):
    stats = {
        "episodes": len(results),
        "successful_episodes": sum(1 for r in results if r["success"]),
        "stop_successful_episodes": sum(1 for r in results if r.get("stop_success")),
        "enter_radius_episodes": sum(1 for r in results if r.get("enter_radius_success", r["success"])),
        "stopped_episodes": sum(1 for r in results if r["stopped"]),
        "terminated_reasons": dict(Counter(
            r.get("terminated_reason") or "unknown" for r in results
        )),
        "trace_parse_status": {},
        "trace_parse_reason": {},
        "invalid_action_steps": 0,
        "stop_action_steps": 0,
        "scaled_action_steps": 0,
        "wrong_stop_outside_radius": 0,
        "missed_stop_after_enter_radius": sum(
            1 for r in results if r.get("terminated_reason") == "missed_stop"
        ),
        "terminal_move_steps": 0,
        "should_stop_move_steps": 0,
        "raw_output_examples": [],
    }
    status_counter = Counter()
    reason_counter = Counter()
    examples = []
    for result in results:
        for row in result.get("trace", []):
            status = row.get("parse_status")
            reason = row.get("parse_reason")
            if status:
                status_counter[status] += 1
            if reason:
                reason_counter[reason] += 1
            if status == "invalid":
                stats["invalid_action_steps"] += 1
            if status == "stop":
                stats["stop_action_steps"] += 1
                if not row.get("should_stop"):
                    stats["wrong_stop_outside_radius"] += 1
            if row.get("scaled_action"):
                stats["scaled_action_steps"] += 1
            if row.get("terminal_zone") and status != "stop":
                stats["terminal_move_steps"] += 1
            if row.get("should_stop") and status != "stop":
                stats["should_stop_move_steps"] += 1
            raw = row.get("raw")
            if raw and len(examples) < 10:
                examples.append({
                    "episode_id": result["episode_id"],
                    "step": row["step"],
                    "raw": raw,
                    "parse_status": status,
                    "parse_reason": reason,
                })
    stats["trace_parse_status"] = dict(status_counter)
    stats["trace_parse_reason"] = dict(reason_counter)
    stats["raw_output_examples"] = examples
    return stats


def build_eval_episodes(data_root, dataset, split, n_episodes, seed, max_images,
                        landmark_max_distance_m, landmark_nearby_distance_m,
                        min_start_distance_m, scenario_choices, pid_filter=None):
    rng_state = random.getstate()
    random.seed(seed)
    pid_index, stats = build_pid_to_path_index(data_root, split=split, dataset=dataset)
    if pid_filter is not None:
        pid_index = filter_pid_index(pid_index, pid_filter)
    pid_index = limit_pid_index(pid_index, max_images=max_images, seed=seed)
    if not pid_index:
        raise RuntimeError(f"No valid images found under {data_root} split={split} dataset={dataset}")

    gsd_cache = build_pid_gsd_index(pid_index)
    pids = sorted(pid_index.keys())
    episodes = []
    attempts = 0
    max_attempts = max(100, n_episodes * 50)

    while len(episodes) < n_episodes and attempts < max_attempts:
        attempts += 1
        pid = random.choice(pids)
        entry = pid_index[pid]
        real_gsd = entry.get("real_gsd")
        if real_gsd is None:
            real_gsd = gsd_cache.get(pid)
        allowed = compatible_height_bins(real_gsd, dataset_name=entry.get("dataset"))
        height_m, grid_size_m, fov_m = sample_height_bin(allowed_heights=allowed)
        ppm = eval_ppm_for_entry(entry, entry["label"])
        scenario = random.choice(scenario_choices)
        ep, _ = sample_episode(
            entry,
            scenario,
            height_m,
            grid_size_m,
            fov_m,
            ppm,
            max_attempts=12,
            landmark_max_distance_m=landmark_max_distance_m,
            landmark_nearby_distance_m=landmark_nearby_distance_m,
            min_start_distance_m=min_start_distance_m,
        )
        if ep is None:
            continue
        ep.update({
            "episode_id": len(episodes),
            "pid": pid,
            "entry": entry,
            "height_m": height_m,
            "grid_size_m": grid_size_m,
            "fov_m": fov_m,
            "ppm": ppm,
            "scenario": scenario,
        })
        if ep.get("landmark") is not None:
            relation_word, relation_dist_m = landmark_relation_context(
                ep["target"],
                ep["landmark"],
                ppm=ppm,
            )
            ep["landmark_relation_word"] = relation_word
            ep["landmark_relation_dist_m"] = relation_dist_m
        episodes.append(ep)

    random.setstate(rng_state)
    if len(episodes) < n_episodes:
        print(f"[warn] sampled only {len(episodes)}/{n_episodes} eval episodes after {attempts} attempts")
    print(f"[eval] dataset stats: {json.dumps(stats, ensure_ascii=False)}")
    print(f"[eval] episodes: {len(episodes)}, images used: {len(pid_index)}")
    scenario_counter = defaultdict(int)
    for ep in episodes:
        scenario_counter[ep["scenario"]] += 1
    if episodes:
        print(f"[eval] scenario counts: {dict(sorted(scenario_counter.items()))}")
    return episodes


def make_policies(args):
    requested = [p.strip() for p in args.policies.split(",") if p.strip()]
    policies = {}
    for name in requested:
        if name == "random":
            policies[name] = random_policy
        elif name == "greedy":
            policies[name] = greedy_policy
        elif name == "zero_shot":
            policies[name] = ModelPolicy(
                name, args.zero_shot_model, args.zero_shot_base_url, args.api_key, args.temperature
            )
        elif name == "finetuned":
            policies[name] = ModelPolicy(
                name, args.finetuned_model, args.finetuned_base_url, args.api_key, args.temperature
            )
        else:
            raise ValueError(f"Unknown policy: {name}")
    return policies


def write_outputs(output_dir, all_results, summary, debug_summary):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "eval_results.jsonl"), "w", encoding="utf-8") as f:
        for r in all_results:
            row = dict(r)
            row.pop("trace", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "eval_traces.jsonl"), "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(os.path.join(output_dir, "eval_debug_summary.json"), "w", encoding="utf-8") as f:
        json.dump(debug_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(os.path.join(output_dir, "eval_summary.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "height_m",
                "n",
                "sr",
                "enter_radius_sr",
                "stop_success_sr",
                "spl",
                "avg_steps",
            ],
        )
        writer.writeheader()
        for policy, metrics in summary.items():
            overall = dict(metrics["overall"])
            overall.update({"policy": policy, "height_m": "overall"})
            writer.writerow(overall)
            for h, vals in metrics["by_height"].items():
                row = dict(vals)
                row.update({"policy": policy, "height_m": h})
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../data/Potsdam")
    parser.add_argument("--dataset", default="potsdam", choices=["potsdam", "dota", "auto"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--generated_dir", default=None,
                        help="Generated dataset directory. If set, eval defaults align to generation_config.json and image_ids_val.txt.")
    parser.add_argument("--eval_pid_list", default=None,
                        help="Optional PID list file for held-out evaluation. Overrides generated_dir/image_ids_val.txt.")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--stop_grace_steps", type=int, default=DEFAULT_STOP_GRACE_STEPS,
                        help="Steps allowed after entering success radius before missed_stop.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="../data/eval_runs")
    parser.add_argument(
        "--policies",
        default="random,greedy",
        help="Comma-separated: random,greedy,zero_shot,finetuned",
    )
    parser.add_argument("--zero_shot_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--zero_shot_model", default="Qwen2.5-VL-7B")
    parser.add_argument("--finetuned_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--finetuned_model", default="Qwen2.5-VL-7B-UAV")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--landmark_max_distance_m", type=float, default=None)
    parser.add_argument("--landmark_nearby_distance_m", type=float, default=None)
    parser.add_argument("--min_start_distance_m", type=float, default=None)
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated scenario set, e.g. A,B,C,D,E. Defaults to generation config.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    settings = resolve_eval_settings(args)
    if settings["pid_filter_path"]:
        print(f"[eval] using held-out pid list: {settings['pid_filter_path']}")
    print(
        "[eval] episode params: "
        f"landmark_max_distance_m={settings['landmark_max_distance_m']}, "
        f"landmark_nearby_distance_m={settings['landmark_nearby_distance_m']}, "
        f"min_start_distance_m={settings['min_start_distance_m']}, "
        f"scenarios={settings['scenario_choices']}"
    )
    episodes = build_eval_episodes(
        args.data_root,
        args.dataset,
        args.split,
        args.num_episodes,
        args.seed,
        args.max_images,
        settings["landmark_max_distance_m"],
        settings["landmark_nearby_distance_m"],
        settings["min_start_distance_m"],
        settings["scenario_choices"],
        pid_filter=settings["pid_filter"],
    )
    policies = make_policies(args)

    all_results = []
    summary = {}
    debug_summary = {}
    for policy_name, policy_fn in policies.items():
        print(f"[eval] running policy={policy_name}")
        policy_results = []
        for ep in episodes:
            result = simulate_episode(
                ep,
                policy_name,
                policy_fn,
                output_dir=args.output_dir,
                max_steps=args.max_steps,
                img_size=IMG_SIZE,
                stop_grace_steps=args.stop_grace_steps,
            )
            policy_results.append(result)
            all_results.append(result)
        overall, by_height = calculate_metrics(policy_results)
        summary[policy_name] = {"overall": overall, "by_height": by_height}
        debug_summary[policy_name] = summarize_policy_debug(policy_results)
        print(
            f"[{policy_name}] N={overall['n']} "
            f"SR={overall['sr']:.3f} "
            f"EnterRadiusSR={overall['enter_radius_sr']:.3f} "
            f"StopSR={overall['stop_success_sr']:.3f} "
            f"SPL={overall['spl']:.3f} "
            f"AvgSteps={overall['avg_steps']:.2f}"
        )
        print(
            f"[{policy_name}] parse_status={debug_summary[policy_name]['trace_parse_status']} "
            f"parse_reason={debug_summary[policy_name]['trace_parse_reason']}"
        )

    write_outputs(args.output_dir, all_results, summary, debug_summary)
    print(f"[eval] wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
