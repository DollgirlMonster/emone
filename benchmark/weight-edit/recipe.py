"""The MIT CSAIL Exp 8 recipe, rerun on Emone (Emone-bbm.3).

  recipe.py score <out_dir> [patch_dir]   teacher-force both branches of every matched pair
                                          (matched_pairs.json) through EmoneCLI --score,
                                          capturing layers 15 and 19 (MLP and residual)
  recipe.py direction <score_dir> <out.npz> <pair>...
                                          u_L = mean residual after layer L on repetitive
                                          branch tokens minus productive, unit length, from
                                          the named pairs; plus each MLP unit's mean
                                          contribution along u on repetitive tokens
  recipe.py patch <dirs.npz> <out_dir> <alpha> <k|all>
                                          W' = W - alpha * u (mask . u^T W) on mlp.down_proj
                                          of 15 and 19, mask = the k units writing most along u
  recipe.py compare <base_dir> <edited_dir>
                                          mean NLL change per branch kind, per pair

TT_CLI names the binary; the model is the installed 0.8B 8-bit. TT_PAIRS=verbatim
replaces each repetitive branch with the prefix's last line repeated three times
(pure copying in the prefix's own style, as in the MIT pairs).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
from build_patch import FORMAT, safetensors  # noqa: E402
from gturbo_reader import GTurboWeights  # noqa: E402

MODEL = os.environ.get("TT_MODEL", str(ROOT / "models/qwen3.5_0.8B_8Bit"))
CLI = os.environ.get("TT_CLI", str(ROOT / ".build/release/EmoneCLI"))
LAYERS = (15, 19)
HIDDEN, INTER = 1024, 3584
MLP_REC = 12 + 2 * HIDDEN + 2 * INTER
RES_REC = 12 + 2 * HIDDEN


def texts():
    spec = json.loads((HERE / "matched_pairs.json").read_text())
    tok = Tokenizer.from_file(f"{MODEL}/tokenizer/tokenizer.json")
    for i, p in enumerate(spec["pairs"]):
        head = spec["template"].format(q=p["q"]) + p["prefix"]
        base = tok.encode(head, add_special_tokens=False).ids
        branches = dict(productive=p["productive"], repetitive=p["repetitive"])
        if os.environ.get("TT_PAIRS") == "verbatim":
            # Pure copying, same style as the prefix: its last line, three more times.
            last = p["prefix"].rstrip("\n").split("\n")[-1] + "\n"
            branches["repetitive"] = last * 3
        for kind in ("productive", "repetitive"):
            ids = tok.encode(head + branches[kind], add_special_tokens=False).ids
            shared = next(
                (j for j, (a, b) in enumerate(zip(base, ids, strict=False)) if a != b), len(base)
            )
            yield i, kind, head + branches[kind], len(ids) - shared


def score(out, patch=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for i, kind, text, n in texts():
        stem = out / f"p{i}_{kind}"
        env = dict(
            os.environ,
            EMONE_MLP_CAPTURE=f"{stem}.mlp",
            EMONE_MLP_CAPTURE_LAYERS=",".join(map(str, LAYERS)),
        )
        cmd = [
            CLI,
            "--model",
            MODEL,
            "--prompt",
            text,
            "--score",
            str(n),
            "--score-out",
            f"{stem}.nll",
        ]
        if patch:
            cmd += ["--weight-patch", patch]
        run = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if run.returncode != 0:
            sys.exit(f"{stem}: {run.stderr[-400:]}")
        print(
            f"p{i} {kind:10s} {n:3d} tokens  mean NLL {np.loadtxt(f'{stem}.nll').mean():.4f}",
            flush=True,
        )


def load(path, rec):
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw[: raw.size // rec * rec].reshape(-1, rec)
    head = raw[:, :12].copy().view(np.int32).reshape(-1, 3)
    return head, raw


def direction(score_dir, out, pairs):
    d = Path(score_dir)
    w = GTurboWeights(MODEL)
    result = {}
    for L in LAYERS:
        res, act = {}, {}
        for kind in ("productive", "repetitive"):
            rs, ac = [], []
            for i in pairs:
                h, raw = load(d / f"p{i}_{kind}.mlp.resid", RES_REC)
                rs.append(raw[h[:, 1] == L, 12:].copy().view(np.float16).astype(np.float32))
                h, raw = load(d / f"p{i}_{kind}.mlp", MLP_REC)
                ac.append(
                    raw[h[:, 1] == L, 12 + 2 * HIDDEN :].copy().view(np.float16).astype(np.float32)
                )
            res[kind], act[kind] = np.concatenate(rs), np.concatenate(ac)
        u = res["repetitive"].mean(0) - res["productive"].mean(0)
        u /= np.linalg.norm(u)
        W = w.get(f"language_model.model.layers.{L}.mlp.down_proj.weight")
        v = u @ W
        # A unit's write along u on repetitive tokens: act_j * (u . W_j).
        contrib = (act["repetitive"] * v).mean(0)
        result[f"u{L}"], result[f"v{L}"], result[f"c{L}"] = u, v, contrib
        sep = (res["repetitive"] @ u).mean() - (res["productive"] @ u).mean()
        top = np.argsort(-contrib)[:5]
        print(
            f"L{L}: projection gap {sep:.3f}; top writers {top.tolist()} "
            f"({', '.join(f'{contrib[j]:+.3f}' for j in top)})"
        )
    np.savez(out, **result)


def patch(dirs_path, out, alpha, k):
    dirs = np.load(dirs_path)
    tensors, report = {}, {}
    for L in LAYERS:
        u, v, c = dirs[f"u{L}"], dirs[f"v{L}"], dirs[f"c{L}"]
        mask = (
            np.ones_like(v, bool)
            if k == "all"
            else np.isin(np.arange(v.size), np.argsort(-c)[: int(k)])
        )
        tensors[f"layers.{L}.mlp.down_proj.lora_A"] = np.where(mask, v, 0)[None, :]
        tensors[f"layers.{L}.mlp.down_proj.lora_B"] = (-alpha * u)[:, None]
        report[str(L)] = {"columns": int(mask.sum()), "weights": int(mask.sum()) * HIDDEN}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blob = safetensors(tensors)
    (out / "weight_patch.safetensors").write_bytes(blob)
    manifest = {
        "format": FORMAT,
        "base": {"model_id": "qwen3.5-0.8b"},
        "rank": 1,
        "description": f"MIT CSAIL Exp 8 recipe: W - {alpha} u (mask . u^T W) on mlp.down_proj of "
        f"layers 15 and 19, top-{k} writers (Emone-bbm.3).",
        "provenance": {
            "method": "u = residual after the layer, repetitive minus productive "
            "matched branches (matched_pairs.json), teacher-forced",
            "alpha": alpha,
            "columns": k,
            "layers": report,
        },
        "files": {"weight_patch.safetensors": hashlib.sha256(blob).hexdigest()},
    }
    (out / "weight_patch.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(manifest["description"])


def compare(base, edited):
    base, edited = Path(base), Path(edited)
    rows = {"productive": [], "repetitive": []}
    for f in sorted(base.glob("p*_*.nll")):
        kind = f.stem.split("_")[1]
        delta = np.loadtxt(edited / f.name).mean() - np.loadtxt(f).mean()
        rows[kind].append(delta)
        print(
            f"{f.stem:16s} NLL {np.loadtxt(f).mean():.4f} -> "
            f"{np.loadtxt(edited / f.name).mean():.4f}  ({delta:+.4f})"
        )
    for kind, v in rows.items():
        print(
            f"{kind:10s} mean NLL change {np.mean(v):+.4f} nats/token (log-prob {-np.mean(v):+.4f})"
        )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "score":
        score(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "direction":
        direction(sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]])
    elif cmd == "patch":
        patch(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5])
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3])
