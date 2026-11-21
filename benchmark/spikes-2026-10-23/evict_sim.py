#!/usr/bin/env python3
"""9ul.5: replay a EMONE_PREFETCH_TRACE through per-layer expert caches.

Policies: the shipped LFU (cumulative per-layer use count, ties by least
recent), LRU, and Belady's offline optimum (evict the resident expert whose
next use is furthest away). Belady is the ceiling any forecast-driven eviction
(SeqMoE's probabilistic Belady) can approach, so if it barely beats LFU the
idea is closed without building it. Only the router's selections are used;
the engine's own residency and prefetch ring are ignored, so read the
policies against each other, not against the engine's measured hit rate.
"""

import collections
import json
import sys


def load(path):
    by_layer = collections.defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        by_layer[row["layer"]].append((row["position"], tuple(row["experts"])))
    for layer in by_layer:
        by_layer[layer].sort()
    return {layer: [experts for _, experts in rows] for layer, rows in by_layer.items()}


def simulate(steps, slots, policy):
    resident = set()
    count = collections.Counter()
    last = {}
    hits = misses = 0
    # Belady: the step index of each expert's future uses.
    future = collections.defaultdict(collections.deque)
    if policy == "belady":
        for t, experts in enumerate(steps):
            for e in experts:
                future[e].append(t)
    for t, experts in enumerate(steps):
        need = [e for e in experts if e not in resident]
        hits += len(experts) - len(need)
        misses += len(need)
        if policy == "belady":
            for e in experts:
                future[e].popleft()
        protected = set(experts)
        for e in need:
            if len(resident) >= slots:
                candidates = [r for r in resident if r not in protected]
                if policy == "lru":
                    victim = min(candidates, key=lambda r: last[r])
                elif policy == "lfu":
                    victim = min(candidates, key=lambda r: (count[r], last[r]))
                else:
                    victim = max(
                        candidates, key=lambda r: future[r][0] if future[r] else 1 << 60)
                resident.discard(victim)
            resident.add(e)
        for e in experts:
            count[e] += 1
            last[e] = t
    return hits, misses


def main():
    path, slots = sys.argv[1], int(sys.argv[2])
    layers = load(path)
    totals = {p: [0, 0] for p in ("lfu", "lru", "belady")}
    for layer, steps in sorted(layers.items()):
        for p in totals:
            h, m = simulate(steps, slots, p)
            totals[p][0] += h
            totals[p][1] += m
    positions = max(len(s) for s in layers.values())
    print(f"trace={path} layers={len(layers)} positions={positions} slots={slots}")
    base = totals["lfu"][1]
    for p, (h, m) in totals.items():
        print(
            f"  {p:7s} hit={h / (h + m):.4f} misses/token={m / positions:.1f} "
            f"misses_vs_lfu={(m - base) / base * 100:+.1f}%")


if __name__ == "__main__":
    main()
