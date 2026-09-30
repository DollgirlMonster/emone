#!/usr/bin/env python3
"""Summarise a TINYTITAN_PREFIX_LOG file: would pinned prefix snapshots pay?

    python3 benchmark/prefix_log_report.py prefix.jsonl
    python3 benchmark/prefix_log_report.py prefix.jsonl --dense-ms-per-token 5.8 --cut-cost-s 17

Each line is one completed request (see PrefixHashLog.swift). The hashes are
chained per block, so a request shares its first k blocks with some earlier
request exactly when its hash k-1 was seen before -- no pairwise comparison.

For every request it reports the longest prefix any earlier request already
had ("reusable"), against what the prompt cache actually served ("cached").
The gap between the two is what better snapshot retention could win.

Where a reusable prefix ends inside the prompt, the request diverged there. For
each such divergence it applies the snapshot-placement rule: re-prefilling the
tokens between the last chunk boundary and the divergence costs
dense-ms-per-token each time the prefix is reused, while cutting a snapshot at
the exact point costs one extra chunk (cut-cost-s) once. The defaults are the
Qwen3.8-Flash 4-bit M1 Max spike figures; pass your own.
"""

import argparse
import json
import sys
from collections import Counter


def load(path):
    rows = []
    with open(path) as handle:
        for number, text in enumerate(handle, 1):
            text = text.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                print(f"skipping malformed line {number}", file=sys.stderr)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("log")
    parser.add_argument("--dense-ms-per-token", type=float, default=5.8)
    parser.add_argument("--cut-cost-s", type=float, default=17.0)
    args = parser.parse_args()

    rows = load(args.log)
    if not rows:
        print("no records")
        return

    seen = set()
    hits_at = Counter()  # hash of a divergence prefix -> times reused
    prompt_total = cached_total = reusable_total = 0
    missed = []  # (reusable - cached) per request, tokens
    divergences = []  # (divergence token, chunk tokens, prefix hash)

    for row in rows:
        hashes = row["hashes"]
        block = row["block_tokens"]
        prompt = row["prompt_tokens"]
        shared = 0
        while shared < len(hashes) and hashes[shared] in seen:
            shared += 1
        reusable = min(prompt, shared * block)
        cached = row["cached_tokens"]
        prompt_total += prompt
        cached_total += cached
        reusable_total += reusable
        missed.append(max(0, reusable - cached))
        if 0 < shared < len(hashes):
            key = hashes[shared - 1]
            hits_at[key] += 1
            divergences.append((shared * block, row["chunk_tokens"], key))
        seen.update(hashes)

    n = len(rows)
    print(f"requests            {n}")
    print(f"prompt tokens       {prompt_total}")
    print(f"cached by server    {cached_total}  ({100 * cached_total / prompt_total:.1f}%)")
    print(f"reusable (ideal)    {reusable_total}  ({100 * reusable_total / prompt_total:.1f}%)")
    lost = sum(missed)
    print(f"missed reuse        {lost} tokens, "
          f"~{lost * args.dense_ms_per_token / 1000:.0f} s of dense prefill "
          f"(at {args.dense_ms_per_token} ms/token)")
    print(f"requests missing >1K reusable tokens: {sum(1 for m in missed if m > 1024)}")

    print()
    print(f"divergence points   {len(divergences)} "
          f"({len(hits_at)} distinct prefixes)")
    hot = [(k, c) for k, c in hits_at.items() if c >= 2]
    print(f"prefixes diverged from 2+ times (pin candidates): {len(hot)}")

    boundary = cut = 0
    for token, chunk, key in divergences:
        gap = token % chunk if chunk > 0 else token
        reuses = hits_at[key]
        # Re-prefilling the gap on every reuse vs one extra chunk, once.
        if gap * args.dense_ms_per_token / 1000 * reuses > args.cut_cost_s:
            cut += 1
        else:
            boundary += 1
    if divergences:
        gaps = sorted(t % c if c > 0 else t for t, c, _ in divergences)
        print(f"gap past last chunk boundary: median {gaps[len(gaps) // 2]} tokens, "
              f"max {gaps[-1]}")
        print(f"placement rule: boundary snapshot {boundary}, exact cut {cut} "
              f"(cut when gap x {args.dense_ms_per_token} ms x reuses > {args.cut_cost_s} s)")


if __name__ == "__main__":
    main()
