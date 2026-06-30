# import json
# from pathlib import Path
# from collections import defaultdict

# manifest = Path(r"../data/train_manifest.jsonl")
# if not manifest.exists():
#     manifest = Path(r"../data/eval_manifest.jsonl")

# traj_steps = defaultdict(int)
# with manifest.open("r", encoding="utf-8") as f:
#     for line in f:
#         row = json.loads(line)
#         key = (row["pid"], row["traj_id"])
#         traj_steps[key] += 1

# total = len(traj_steps)
# bad = {k:v for k,v in traj_steps.items() if v < 3}  # 2次移动 + 1次STOP/末帧，通常至少3条transition
# print("total trajectories:", total)
# print("too short trajectories:", len(bad))
# for i, (k, v) in enumerate(bad.items()):
#     if i >= 10:
#         break
#     print(k, v)

import json, statistics
from collections import Counter

path = "eval_finetuned/eval_results.jsonl"
rows = [json.loads(x) for x in open(path, "r", encoding="utf-8")]

fails = [r for r in rows if not r["success"]]
print("N =", len(rows))
print("SR =", sum(r["success"] for r in rows) / len(rows))
print("terminated =", Counter(r["terminated_reason"] for r in rows))
print("failed final_dist mean =", sum(r["final_dist_m"] for r in fails) / max(1, len(fails)))
print("failed final_dist median =", statistics.median([r["final_dist_m"] for r in fails]) if fails else None)

for thr in [5, 7.5, 10, 15]:
    sr = sum(r["final_dist_m"] <= thr for r in rows) / len(rows)
    print(f"SR@{thr}m =", round(sr, 3))

dbg = json.load(open("eval_finetuned/eval_debug_summary.json", "r", encoding="utf-8"))
print(json.dumps(dbg["finetuned"]["terminated_reasons"], ensure_ascii=False, indent=2))
print(json.dumps(dbg["finetuned"]["raw_output_examples"][:10], ensure_ascii=False, indent=2))