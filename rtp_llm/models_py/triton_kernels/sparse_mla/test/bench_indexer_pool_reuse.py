"""Per-layer decode Indexer latency under CUDA graphs: original vs pool reuse.

Prints, per batch size and context length, the original full-context Indexer,
the pool path's refresh and reuse steps, and the average over one refresh plus
`steps` reuse steps. Usage: bench_indexer_pool_reuse.py [--steps 15] [--pool 8192]
"""

import argparse

import deep_gemm
import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.indexer_pool_reuse import (
    PAGE,
    TOPK,
    IndexerPool,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops

HEADS = 32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--batches", default="1,8,32,64")
    p.add_argument("--contexts", default="32768,65536,131072")
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--pool", type=int, default=8192)
    args = p.parse_args()
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    num_sms = deep_gemm.get_num_sms()
    print(f"| BS | context | original us | refresh us | reuse us | 1+{args.steps} avg us | speedup |")
    print("|---|---|---|---|---|---|---|")
    for rows in map(int, args.batches.split(",")):
        for context in map(int, args.contexts.split(",")):
            cols = context // PAGE + 1
            pages = rows * cols
            cache = torch.randint(0, 120, (pages, PAGE * 132), dtype=torch.uint8, device="cuda")
            cache.view(pages, -1)[:, PAGE * 128 :].view(torch.float32).uniform_(0.5, 1.5)
            table = torch.randperm(pages, device="cuda").int().view(rows, cols)
            q = torch.randn((rows, HEADS, 128), device="cuda").to(torch.float8_e4m3fn)
            w = torch.randn((rows, HEADS), device="cuda") / HEADS
            lengths = torch.full((rows,), context - args.steps - 2, dtype=torch.int32, device="cuda")
            max_len = cols * PAGE

            def logits(n):
                schedule = deep_gemm.get_paged_mqa_logits_metadata(n.view(-1, 1), PAGE, num_sms)
                return deep_gemm.fp8_paged_mqa_logits(
                    q.view(rows, 1, HEADS, 128), cache.view(-1, PAGE, 1, 132), w,
                    n.view(-1, 1), table, schedule, max_len, clean_logits=False,
                )

            def topk(x, n, out, m):
                rtp_llm_ops.dsv4_persistent_topk(x, n, out, workspace, TOPK, m)

            schedule = deep_gemm.get_paged_mqa_logits_metadata(lengths.view(-1, 1), PAGE, num_sms)

            def original():
                out = torch.empty((rows, TOPK), dtype=torch.int32, device="cuda")
                x = deep_gemm.fp8_paged_mqa_logits(
                    q.view(rows, 1, HEADS, 128), cache.view(-1, PAGE, 1, 132), w,
                    lengths.view(-1, 1), table, schedule, max_len, clean_logits=False,
                )
                topk(x, lengths, out, max_len)
                return out

            pool = IndexerPool(
                slots=max(rows, 16), reuse_steps=args.steps, min_length=16384, device="cuda", pool_size=args.pool
            )
            graphs = {}
            def pooled():
                return pool.topk(q, w, cache, table, lengths, logits, topk)

            for name, fn in (("original", original), ("pool", pooled)):
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    fn()
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                graphs[name] = graph
            pool.reset()

            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            def timed(graph):
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) * 1000

            base = lengths.clone()
            for _ in range(5):
                graphs["original"].replay()
            orig = sorted(timed(graphs["original"]) for _ in range(30))[15]
            refresh, reuse = [], []
            for cycle in range(args.cycles):
                lengths.copy_(base)
                pool.reset()
                for step in range(args.steps + 1):
                    t = timed(graphs["pool"])
                    if cycle:  # first cycle warms up
                        (refresh if step == 0 else reuse).append(t)
                    lengths.add_(1)
            refresh_us = sorted(refresh)[len(refresh) // 2]
            reuse_us = sorted(reuse)[len(reuse) // 2]
            avg = (refresh_us + args.steps * reuse_us) / (args.steps + 1)
            cells = [rows, f"{context // 1024}K", f"{orig:.1f}", f"{refresh_us:.1f}", f"{reuse_us:.1f}", f"{avg:.1f}"]
            print("| " + " | ".join(map(str, cells)) + f" | {orig / avg:.2f}x |", flush=True)
            del cache, table, graphs, pool
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
