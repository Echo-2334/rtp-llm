"""Replay Indexer pool reuse on captured GLM-5.3 decode logits.

Input: a directory of captures, one per request, as written by the Indexer
logits probe: `logits-<pid>-request<i>.jsonl` lists one record per decode
step and layer ({"layer", "kv_length", "offset_bytes", "count"}), and
`logits-<pid>-request<i>-layer<L>.f16` holds the fp16 logits rows.

For each request and layer, an anchor step every N + 1 steps builds a top-P
pool; the next N steps select top-2048 from the pool plus the tokens appended
since the anchor. Recall counts selections whose logit reaches the exact
2048-th logit of that step, so ties do not count as misses. The stale baseline
reuses the anchor step's top-2048 unchanged.

Usage: replay_indexer_pool_recall.py CAPTURE_DIR [--pools 8192,16384] [--steps 4,15]
"""

import argparse
import collections
import glob
import json

import numpy as np
import torch

TOPK = 2048


def summary(values):
    v = np.array(values)
    return f"mean {v.mean():.4f}  p10 {np.percentile(v, 10):.4f}  p1 {np.percentile(v, 1):.4f}  (n={len(v)})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capture_dir")
    p.add_argument("--pools", default="8192,16384,32768")
    p.add_argument("--steps", default="4,15")
    args = p.parse_args()
    configs = [
        (int(a), int(b)) for a in args.pools.split(",") for b in args.steps.split(",")
    ]
    recall = collections.defaultdict(list)
    per_layer = collections.defaultdict(list)
    stale = collections.defaultdict(list)
    for manifest in sorted(glob.glob(f"{args.capture_dir}/logits-*-request*.jsonl")):
        prefix = manifest[: -len(".jsonl")]
        by_layer = collections.defaultdict(list)
        for line in open(manifest):
            record = json.loads(line)
            by_layer[record["layer"]].append(record)
        for layer, records in sorted(by_layer.items()):
            data = np.memmap(f"{prefix}-layer{layer}.f16", dtype=np.float16, mode="r")
            steps = [
                torch.from_numpy(
                    np.array(
                        data[
                            r["offset_bytes"] // 2 : r["offset_bytes"] // 2 + r["count"]
                        ]
                    )
                )
                .cuda()
                .float()
                for r in records
            ]
            exact = [s.topk(TOPK) for s in steps]
            for pool_size, n in configs:
                for t0 in range(0, len(steps) - n, n + 1):
                    pool = steps[t0].topk(pool_size).indices
                    anchor = steps[t0].numel()
                    for t in range(t0 + 1, t0 + n + 1):
                        tail = torch.arange(anchor, steps[t].numel(), device="cuda")
                        chosen = steps[t][torch.cat([pool, tail])].topk(TOPK).values
                        hit = (chosen >= exact[t].values[-1]).sum().item() / TOPK
                        recall[(pool_size, n)].append(hit)
                        per_layer[(pool_size, n, layer)].append(hit)
                        if pool_size == configs[0][0]:
                            kept = steps[t][exact[t0].indices] >= exact[t].values[-1]
                            stale[n].append(kept.sum().item() / TOPK)
            del steps, exact
            torch.cuda.empty_cache()

    print("| pool | N | recall of exact top-2048 |")
    print("|---|---|---|")
    for pool_size, n in configs:
        print(f"| {pool_size} | {n} | {summary(recall[(pool_size, n)])} |")
    for n, values in sorted(stale.items()):
        print(f"stale top-2048, N={n}: {summary(values)}")
    pool_size, n = configs[0]
    print(f"per layer, pool {pool_size}, N={n}:")
    for (p_, n_, layer), values in sorted(per_layer.items()):
        if (p_, n_) == (pool_size, n):
            print(f"  layer {layer:2d}: {summary(values)}")


if __name__ == "__main__":
    main()
