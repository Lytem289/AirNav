import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from train_data_generate import HEIGHT_BIN_RATIOS, NOMINAL_SCENARIO_RATIOS


IMG_NAME_RE = re.compile(r'^(.+)_traj(\d+)_step(\d+)\.jpg$', re.IGNORECASE)


def load_jsonl(path):
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{lineno} JSON decode failed: {exc}') from exc
    return rows


def load_pid_list(path):
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def resolve_image_path(image_ref, jsonl_dir):
    p = Path(image_ref)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    candidate = (jsonl_dir / p).resolve()
    if candidate.exists():
        return candidate
    return p.resolve()


def valid_pid(pid):
    """Accept both DOTA ids (P0001) and Potsdam tile ids (for example 4_11)."""
    return isinstance(pid, str) and pid.strip() and '_traj' not in pid and '/' not in pid and '\\' not in pid


def parse_image_name(image_ref):
    name = Path(image_ref).name
    m = IMG_NAME_RE.match(name)
    if not m:
        return None
    pid, traj_id, step = m.groups()
    if not valid_pid(pid):
        return None
    return {
        'pid': pid,
        'traj_id': int(traj_id),
        'step': int(step),
    }


def validate_sharegpt_rows(jsonl_path, split_name, errors, warnings):
    rows = load_jsonl(jsonl_path)
    jsonl_dir = jsonl_path.parent
    image_pids = set()
    parsed_rows = []

    for idx, row in enumerate(rows, 1):
        prefix = f'{jsonl_path.name}:line{idx}'
        conversations = row.get('conversations')
        images = row.get('images')

        if not isinstance(conversations, list) or len(conversations) != 2:
            errors.append(f'{prefix} conversations must be a 2-item list.')
            continue
        if not isinstance(images, list) or len(images) != 2:
            errors.append(f'{prefix} images must be a 2-item list.')
            continue

        human = conversations[0]
        gpt = conversations[1]
        if human.get('from') != 'human':
            errors.append(f'{prefix} first conversation role must be human.')
        if gpt.get('from') != 'gpt':
            errors.append(f'{prefix} second conversation role must be gpt.')

        human_text = human.get('value', '')
        gpt_text = gpt.get('value', '')
        if human_text.count('<image>') != len(images):
            errors.append(
                f'{prefix} <image> count {human_text.count("<image>")} != images count {len(images)}.'
            )
        if 'Action:' not in gpt_text:
            errors.append(f'{prefix} gpt response missing "Action:".')
        # if 'Thought:' not in gpt_text:
        #     warnings.append(f'{prefix} gpt response missing "Thought:".')

        img_meta = []
        for img_ref in images:
            resolved = resolve_image_path(img_ref, jsonl_dir)
            if not resolved.exists():
                errors.append(f'{prefix} missing image: {img_ref}')
            parsed = parse_image_name(img_ref)
            if parsed is None:
                errors.append(f'{prefix} image filename does not match expected pattern: {img_ref}')
            else:
                image_pids.add(parsed['pid'])
            img_meta.append({'ref': img_ref, 'resolved': str(resolved), 'parsed': parsed})

        parsed_rows.append({
            'split': split_name,
            'index': idx - 1,
            'images': img_meta,
            'row': row,
        })

    return rows, parsed_rows, image_pids


def validate_manifest(manifest_path, split_name, parsed_rows, errors, warnings):
    if not manifest_path.exists():
        warnings.append(f'{manifest_path.name} not found; scenario/height distribution checks skipped.')
        return [], set(), {}

    rows = load_jsonl(manifest_path)
    manifest_pids = set()
    traj_meta = {}

    if len(rows) != len(parsed_rows):
        errors.append(
            f'{manifest_path.name} rows={len(rows)} does not match {split_name}.jsonl rows={len(parsed_rows)}.'
        )

    for idx, row in enumerate(rows, 1):
        prefix = f'{manifest_path.name}:line{idx}'
        if row.get('split') != split_name:
            errors.append(f'{prefix} split={row.get("split")} != {split_name}.')

        pid = row.get('pid')
        traj_id = row.get('traj_id')
        step = row.get('step')
        scenario = row.get('scenario')
        height_m = row.get('height_m')
        phase = row.get('phase')

        if not valid_pid(pid):
            errors.append(f'{prefix} invalid pid: {pid}')
        else:
            manifest_pids.add(pid)
        if not isinstance(traj_id, int):
            errors.append(f'{prefix} invalid traj_id: {traj_id}')
        if not isinstance(step, int):
            errors.append(f'{prefix} invalid step: {step}')
        if scenario not in NOMINAL_SCENARIO_RATIOS:
            errors.append(f'{prefix} invalid scenario: {scenario}')
        if height_m not in HEIGHT_BIN_RATIOS:
            errors.append(f'{prefix} invalid height_m: {height_m}')

        key = (pid, traj_id)
        if key not in traj_meta:
            traj_meta[key] = {
                'scenario': scenario,
                'height_m': height_m,
                'steps': [],
                'phases': [],
                'target_visible': [],
                'landmark_visible': [],
                'recovery_start': [],
                'terminal_zone': [],
                'should_stop': [],
            }
        else:
            if traj_meta[key]['scenario'] != scenario:
                errors.append(f'{prefix} scenario changed inside same trajectory {key}.')
            if traj_meta[key]['height_m'] != height_m:
                errors.append(f'{prefix} height changed inside same trajectory {key}.')
        traj_meta[key]['steps'].append(step)
        if phase:
            traj_meta[key]['phases'].append(phase)
        traj_meta[key]['target_visible'].append(bool(row.get('target_visible')))
        traj_meta[key]['landmark_visible'].append(bool(row.get('landmark_visible')))
        traj_meta[key]['recovery_start'].append(bool(row.get('recovery_start')))
        traj_meta[key]['terminal_zone'].append(bool(row.get('terminal_zone')))
        traj_meta[key]['should_stop'].append(bool(row.get('should_stop')))

        if idx - 1 < len(parsed_rows):
            parsed = parsed_rows[idx - 1]
            img_hist = row.get('image_hist')
            img_cur = row.get('image_cur')
            expected_hist = parsed['row'].get('images', [None, None])[0]
            expected_cur = parsed['row'].get('images', [None, None])[1]
            if img_hist != expected_hist or img_cur != expected_cur:
                errors.append(f'{prefix} manifest images do not match {split_name}.jsonl row {idx}.')

            parsed_cur = parsed['images'][1]['parsed']
            if parsed_cur is not None:
                if parsed_cur['pid'] != pid:
                    errors.append(f'{prefix} manifest pid != image pid.')
                if parsed_cur['traj_id'] != traj_id:
                    errors.append(f'{prefix} manifest traj_id != image traj_id.')
                if parsed_cur['step'] != step:
                    errors.append(f'{prefix} manifest step != image step.')

        if phase == 'LANDMARK_GUIDED_SEARCH' and row.get('target_visible'):
            warnings.append(f'{prefix} LANDMARK_GUIDED_SEARCH has target_visible=true.')
        if phase == 'APPROACH_TARGET' and not row.get('target_visible'):
            errors.append(f'{prefix} APPROACH_TARGET requires target_visible=true.')
        if 'should_stop' in row:
            action = str(row.get('action'))
            is_stop = action == 'STOP' or action == '(0, 0)'
            if row.get('should_stop') and not is_stop:
                errors.append(f'{prefix} should_stop=true but action={action}.')
            if is_stop and not row.get('should_stop'):
                errors.append(f'{prefix} action={action} but should_stop=false.')

    for key, meta in traj_meta.items():
        steps = sorted(meta['steps'])
        if steps != list(range(len(steps))):
            warnings.append(f'{manifest_path.name} trajectory {key} has non-consecutive steps: {steps}')

    return rows, manifest_pids, traj_meta


def summarize_distribution(traj_meta, key_name):
    total = len(traj_meta)
    counts = Counter(meta[key_name] for meta in traj_meta.values())
    ratios = {}
    for key, count in sorted(counts.items(), key=lambda kv: kv[0]):
        ratios[key] = count / max(1, total)
    return counts, ratios, total


def compare_ratios(actual_ratios, target_ratios, label, tolerance, warnings):
    for key, target in target_ratios.items():
        actual = actual_ratios.get(key, 0.0)
        delta = actual - target
        if abs(delta) > tolerance:
            warnings.append(
                f'{label} ratio drift for {key}: actual={actual:.1%}, target={target:.1%}, delta={delta:+.1%}.'
            )


def summarize_stop_ratio(rows):
    stop_count = 0
    total = 0
    for row in rows:
        conversations = row.get('conversations') or []
        if len(conversations) < 2:
            continue
        gpt_text = conversations[1].get('value', '')
        total += 1
        if 'Action: STOP' in gpt_text or 'Action: (0, 0)' in gpt_text:
            stop_count += 1
    return {
        'stop_transitions': stop_count,
        'total_transitions': total,
        'stop_ratio': (stop_count / total) if total else 0.0,
    }


def summarize_trajectory_lengths(traj_meta):
    lengths = [len(meta.get('steps', [])) for meta in traj_meta.values()]
    if not lengths:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'avg': 0.0,
            'histogram': {},
        }
    histogram = Counter(lengths)
    return {
        'count': len(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'avg': sum(lengths) / len(lengths),
        'histogram': dict(sorted(histogram.items())),
    }


def summarize_phase_distribution(traj_meta):
    counts = Counter()
    for meta in traj_meta.values():
        counts.update(meta.get('phases', []))
    total = sum(counts.values())
    ratios = {
        key: count / total
        for key, count in sorted(counts.items())
    } if total else {}
    return counts, ratios, total


def summarize_recovery_starts(traj_meta):
    count = sum(1 for meta in traj_meta.values() if any(meta.get('recovery_start', [])))
    total = len(traj_meta)
    return {
        'count': count,
        'total': total,
        'ratio': (count / total) if total else 0.0,
    }


def summarize_terminal_contrast(manifest_rows):
    stats = Counter()
    for row in manifest_rows:
        action = str(row.get('action'))
        is_stop = action == 'STOP' or action == '(0, 0)'
        terminal_zone = bool(row.get('terminal_zone'))
        should_stop = bool(row.get('should_stop'))
        if terminal_zone:
            stats['terminal_zone'] += 1
        if should_stop:
            stats['should_stop'] += 1
        if terminal_zone and not should_stop:
            stats['terminal_nonstop'] += 1
        if is_stop:
            stats['stop_action'] += 1
        if is_stop and not should_stop:
            stats['wrong_stop_label'] += 1
        if should_stop and not is_stop:
            stats['missed_stop_label'] += 1
    total = len(manifest_rows)
    return {
        'total': total,
        'terminal_zone': stats['terminal_zone'],
        'should_stop': stats['should_stop'],
        'terminal_nonstop': stats['terminal_nonstop'],
        'stop_action': stats['stop_action'],
        'wrong_stop_label': stats['wrong_stop_label'],
        'missed_stop_label': stats['missed_stop_label'],
        'should_stop_ratio': (stats['should_stop'] / total) if total else 0.0,
        'terminal_nonstop_ratio': (stats['terminal_nonstop'] / total) if total else 0.0,
    }


def load_generation_config(config_path, warnings):
    if not config_path.exists():
        warnings.append(
            f'{config_path.name} not found; ratio checks fall back to nominal defaults and may over-warn.'
        )
        return {}
    try:
        return json.loads(config_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        warnings.append(f'{config_path.name} JSON decode failed; ratio checks fall back to defaults: {exc}')
        return {}


def resolve_ratio_targets(generation_config, split_name, warnings):
    split_cfg = generation_config.get('splits', {}).get(split_name, {})
    scenario_targets = split_cfg.get('scenario_target_ratios')
    height_targets = split_cfg.get('height_target_ratios')

    if not isinstance(scenario_targets, dict):
        scenario_targets = dict(NOMINAL_SCENARIO_RATIOS)
        warnings.append(
            f'{split_name} scenario targets missing in generation_config; using nominal defaults.'
        )
    else:
        scenario_targets = {str(k): float(v) for k, v in scenario_targets.items()}
    if not isinstance(height_targets, dict):
        height_targets = dict(HEIGHT_BIN_RATIOS)
        warnings.append(
            f'{split_name} height targets missing in generation_config; using nominal defaults.'
        )
    else:
        coerced_height_targets = {}
        for key, value in height_targets.items():
            try:
                coerced_height_targets[int(key)] = float(value)
            except (TypeError, ValueError):
                warnings.append(
                    f'{split_name} height target key {key!r} is invalid; falling back to nominal defaults.'
                )
                coerced_height_targets = dict(HEIGHT_BIN_RATIOS)
                break
        height_targets = coerced_height_targets
    return scenario_targets, height_targets


def validate_dataset_info(dataset_info_path, errors, warnings):
    if not dataset_info_path.exists():
        warnings.append(f'{dataset_info_path.name} not found; dataset registration check skipped.')
        return
    try:
        data = json.loads(dataset_info_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        errors.append(f'{dataset_info_path} JSON decode failed: {exc}')
        return

    for key, file_name in (('uav_full', 'train.jsonl'), ('uav_eval', 'eval.jsonl')):
        entry = data.get(key)
        if entry is None:
            errors.append(f'{dataset_info_path.name} missing dataset key: {key}')
            continue
        if entry.get('formatting') != 'sharegpt':
            errors.append(f'{dataset_info_path.name}:{key} formatting must be sharegpt.')
        columns = entry.get('columns', {})
        if columns.get('messages') != 'conversations' or columns.get('images') != 'images':
            errors.append(f'{dataset_info_path.name}:{key} columns mapping is invalid.')
        if Path(entry.get('file_name', '')).name != file_name:
            warnings.append(
                f'{dataset_info_path.name}:{key} file_name={entry.get("file_name")} expected basename {file_name}.'
            )


def main():
    parser = argparse.ArgumentParser(description='Validate generated UAV ShareGPT data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing train/eval jsonl outputs.')
    parser.add_argument('--ratio_tolerance', type=float, default=0.03, help='Allowed ratio drift before warning.')
    parser.add_argument(
        '--min_ratio_sample',
        type=int,
        default=50,
        help='Minimum trajectory count required before ratio drift checks are enforced.',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    train_jsonl = output_dir / 'train.jsonl'
    eval_jsonl = output_dir / 'eval.jsonl'
    train_manifest = output_dir / 'train_manifest.jsonl'
    eval_manifest = output_dir / 'eval_manifest.jsonl'
    train_ids = output_dir / 'image_ids_train.txt'
    eval_ids = output_dir / 'image_ids_val.txt'
    dataset_info = output_dir / 'dataset_info.json'
    generation_config_path = output_dir / 'generation_config.json'

    errors = []
    warnings = []

    if not train_jsonl.exists():
        errors.append(f'missing {train_jsonl}')
    if not eval_jsonl.exists():
        errors.append(f'missing {eval_jsonl}')
    if errors:
        for msg in errors:
            print(f'ERROR: {msg}')
        return 1

    train_rows, train_parsed, train_image_pids = validate_sharegpt_rows(train_jsonl, 'train', errors, warnings)
    eval_rows, eval_parsed, eval_image_pids = validate_sharegpt_rows(eval_jsonl, 'eval', errors, warnings)

    train_manifest_rows, train_manifest_pids, train_traj_meta = validate_manifest(
        train_manifest, 'train', train_parsed, errors, warnings
    )
    eval_manifest_rows, eval_manifest_pids, eval_traj_meta = validate_manifest(
        eval_manifest, 'eval', eval_parsed, errors, warnings
    )

    train_pid_list = set(load_pid_list(train_ids))
    eval_pid_list = set(load_pid_list(eval_ids))

    leakage = train_image_pids & eval_image_pids
    if leakage:
        errors.append(f'train/eval image PID leakage detected: {sorted(leakage)[:10]}')

    if train_pid_list and eval_pid_list:
        split_leakage = train_pid_list & eval_pid_list
        if split_leakage:
            errors.append(f'image_ids split leakage detected: {sorted(split_leakage)[:10]}')
        missing_train = train_image_pids - train_pid_list
        missing_eval = eval_image_pids - eval_pid_list
        if missing_train:
            warnings.append(f'train.jsonl contains PIDs not listed in image_ids_train.txt: {sorted(missing_train)[:10]}')
        if missing_eval:
            warnings.append(f'eval.jsonl contains PIDs not listed in image_ids_val.txt: {sorted(missing_eval)[:10]}')

    if train_manifest_pids and train_manifest_pids != train_image_pids:
        warnings.append('train manifest PID set differs from train jsonl PID set.')
    if eval_manifest_pids and eval_manifest_pids != eval_image_pids:
        warnings.append('eval manifest PID set differs from eval jsonl PID set.')

    validate_dataset_info(dataset_info, errors, warnings)
    generation_config = load_generation_config(generation_config_path, warnings)
    train_scenario_targets, train_height_targets = resolve_ratio_targets(
        generation_config, 'train', warnings
    )
    eval_scenario_targets, eval_height_targets = resolve_ratio_targets(
        generation_config, 'eval', warnings
    )

    if train_traj_meta:
        counts, ratios, total = summarize_distribution(train_traj_meta, 'scenario')
        print(f'Train trajectories: {total}')
        print(f'Train scenario counts: {dict(counts)}')
        print(f'Train scenario ratios: { {k: round(v, 4) for k, v in ratios.items()} }')
        if total >= args.min_ratio_sample:
            compare_ratios(ratios, train_scenario_targets, 'train scenario', args.ratio_tolerance, warnings)
        else:
            warnings.append(
                f'train scenario ratio check skipped: only {total} trajectories < min_ratio_sample={args.min_ratio_sample}.'
            )

        h_counts, h_ratios, _ = summarize_distribution(train_traj_meta, 'height_m')
        print(f'Train height counts: {dict(h_counts)}')
        print(f'Train height ratios: { {k: round(v, 4) for k, v in h_ratios.items()} }')
        if total >= args.min_ratio_sample:
            compare_ratios(h_ratios, train_height_targets, 'train height', args.ratio_tolerance, warnings)
        else:
            warnings.append(
                f'train height ratio check skipped: only {total} trajectories < min_ratio_sample={args.min_ratio_sample}.'
            )
        stop_stats = summarize_stop_ratio(train_rows)
        print(
            f"Train stop transitions: {stop_stats['stop_transitions']}/{stop_stats['total_transitions']} "
            f"({stop_stats['stop_ratio']:.4f})"
        )
        length_stats = summarize_trajectory_lengths(train_traj_meta)
        print(
            f"Train trajectory lengths: min={length_stats['min']} avg={length_stats['avg']:.2f} "
            f"max={length_stats['max']}"
        )
        print(f"Train trajectory length histogram: {length_stats['histogram']}")
        phase_counts, phase_ratios, _ = summarize_phase_distribution(train_traj_meta)
        print(f'Train phase counts: {dict(phase_counts)}')
        print(f'Train phase ratios: { {k: round(v, 4) for k, v in phase_ratios.items()} }')
        recovery_stats = summarize_recovery_starts(train_traj_meta)
        print(
            f"Train recovery starts: {recovery_stats['count']}/{recovery_stats['total']} "
            f"({recovery_stats['ratio']:.4f})"
        )
        terminal_stats = summarize_terminal_contrast(train_manifest_rows)
        print(
            "Train terminal contrast: "
            f"terminal={terminal_stats['terminal_zone']} "
            f"should_stop={terminal_stats['should_stop']} "
            f"terminal_nonstop={terminal_stats['terminal_nonstop']} "
            f"wrong_stop={terminal_stats['wrong_stop_label']} "
            f"missed_stop={terminal_stats['missed_stop_label']}"
        )

    if eval_traj_meta:
        counts, ratios, total = summarize_distribution(eval_traj_meta, 'scenario')
        print(f'Eval trajectories: {total}')
        print(f'Eval scenario counts: {dict(counts)}')
        print(f'Eval scenario ratios: { {k: round(v, 4) for k, v in ratios.items()} }')
        if total >= args.min_ratio_sample:
            compare_ratios(ratios, eval_scenario_targets, 'eval scenario', args.ratio_tolerance, warnings)

        h_counts, h_ratios, _ = summarize_distribution(eval_traj_meta, 'height_m')
        print(f'Eval height counts: {dict(h_counts)}')
        print(f'Eval height ratios: { {k: round(v, 4) for k, v in h_ratios.items()} }')
        if total >= args.min_ratio_sample:
            compare_ratios(h_ratios, eval_height_targets, 'eval height', args.ratio_tolerance, warnings)
        stop_stats = summarize_stop_ratio(eval_rows)
        print(
            f"Eval stop transitions: {stop_stats['stop_transitions']}/{stop_stats['total_transitions']} "
            f"({stop_stats['stop_ratio']:.4f})"
        )
        length_stats = summarize_trajectory_lengths(eval_traj_meta)
        print(
            f"Eval trajectory lengths: min={length_stats['min']} avg={length_stats['avg']:.2f} "
            f"max={length_stats['max']}"
        )
        print(f"Eval trajectory length histogram: {length_stats['histogram']}")
        phase_counts, phase_ratios, _ = summarize_phase_distribution(eval_traj_meta)
        print(f'Eval phase counts: {dict(phase_counts)}')
        print(f'Eval phase ratios: { {k: round(v, 4) for k, v in phase_ratios.items()} }')
        recovery_stats = summarize_recovery_starts(eval_traj_meta)
        print(
            f"Eval recovery starts: {recovery_stats['count']}/{recovery_stats['total']} "
            f"({recovery_stats['ratio']:.4f})"
        )
        terminal_stats = summarize_terminal_contrast(eval_manifest_rows)
        print(
            "Eval terminal contrast: "
            f"terminal={terminal_stats['terminal_zone']} "
            f"should_stop={terminal_stats['should_stop']} "
            f"terminal_nonstop={terminal_stats['terminal_nonstop']} "
            f"wrong_stop={terminal_stats['wrong_stop_label']} "
            f"missed_stop={terminal_stats['missed_stop_label']}"
        )

    print(f'Train transitions: {len(train_rows)}')
    print(f'Eval transitions: {len(eval_rows)}')
    print(f'Train image PIDs: {len(train_image_pids)}')
    print(f'Eval image PIDs: {len(eval_image_pids)}')

    for msg in warnings:
        print(f'WARN: {msg}')
    for msg in errors:
        print(f'ERROR: {msg}')

    if errors:
        print(f'Validation failed with {len(errors)} error(s) and {len(warnings)} warning(s).')
        return 1
    print(f'Validation passed with {len(warnings)} warning(s).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
