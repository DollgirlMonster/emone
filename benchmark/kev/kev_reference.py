"""Kev-0.8B reference fixture for the Emone parity spike (Emone-lmw.3).

Loads Qwen3.5-0.8B-Base + the kev-0.8b v1.0 adapter + head exactly as kev.checkpoint does on the exact path
(fp32, CPU, adapter unmerged), scores a fixed request set three ways, and writes a fixture:

  A  Kev's own serving path (state prefix once, question rows from its cache)  -> the reference
  B  one plain causal row per question (state + branch), unmerged               -> what the engine will run
  C  B on a merged backbone (W += alpha/r * B@A, fp32), plain AutoModel          -> what the converter will install

and exports the merged fp32 backbone as safetensors so the Emone converter can take it.

usage: python -I kev_reference.py <kev-src> <base-dir> <kev-dir> <out-dir>
"""

import json
import sys
from pathlib import Path

KEV_SRC, BASE, KEV, OUT = map(Path, sys.argv[1:5])
sys.path.insert(0, str(KEV_SRC))
import torch  # noqa: E402 (after the sys.path insert)
import torch.nn.functional as F  # noqa: E402 (after the sys.path insert)
from safetensors.torch import save_file  # noqa: E402 (after the sys.path insert)
import model as kev  # kev/model.py, fetched verbatim  # noqa: E402 (after the sys.path insert)
import api as kev_api  # kev/api.py, fetched verbatim  # noqa: E402 (after the sys.path insert)

torch.manual_seed(0)
OUT.mkdir(parents=True, exist_ok=True)

REQUESTS = [
    {
        "state": "User: please run the test suite again and tell me which tests fail",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "What does the user want the agent to do?",
                "criteria": {
                    "run_command": "execute a shell command",
                    "edit_code": "change source files",
                    "answer_question": "explain something",
                    "chit_chat": None,
                },
            },
            "needs_tool": {
                "type": "noul",
                "instructions": "Does answering require running a tool?",
            },
        },
    },
    {
        "state": "The deploy failed again. I have been at this for three hours and nothing works. Fix it NOW.",
        "questions": {
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the user?",
                "criteria": ["Calm", "Annoyed", "Very angry"],
            },
            "escalate": {
                "type": "noul",
                "instructions": "Should this be escalated to a human?",
                "criteria": {"true": "the user is blocked and upset", "false": "routine request"},
            },
        },
    },
    {
        "state": {
            "command": "rm -rf build/ && swift build -c release",
            "cwd": "/Volumes/Chai/TinyTitan",
        },
        "questions": {
            "risk": {
                "type": "choice",
                "instructions": "How risky is running this command?",
                "criteria": {
                    "safe": "read-only or easily undone",
                    "moderate": "deletes generated files",
                    "dangerous": "can destroy user data",
                },
            }
        },
    },
]


def load(merge):
    tok = kev.load_tokenizer(str(BASE))
    m = kev.DecisionModel(
        str(BASE), tok, "cpu", lora=None, head_dim=256, dtype=torch.float32, attn="eager"
    )
    from peft import PeftModel

    m.lm = PeftModel.from_pretrained(m.lm, str(KEV), torch_device="cpu")
    if merge:
        m.lm = m.lm.merge_and_unload()
    meta = torch.load(KEV / "head.pt", map_location="cpu", weights_only=True)
    m.head.load_state_dict(meta["head"])
    m.eval()
    m.head.temperature = float(meta["temperature"])
    return tok, m, meta


@torch.no_grad()
def plain_rows(m, enc):
    """B/C: one causal row per question, no cache; hidden = final-norm output; returns probs and the readout hiddens."""
    S, Spos, rows = kev.rows_of(enc)
    out = []
    for r in rows:
        ids = torch.tensor([S + r["ids"]])
        pos = torch.tensor([Spos + r["pos"]])
        h = m.lm(input_ids=ids, position_ids=pos).last_hidden_state[0].float()
        d = len(S) + r["decide"]
        o = [len(S) + x for x in r["opts"]]
        z = m.head(h[d], h[torch.tensor(o)])
        out.append(
            {
                "ids": S + r["ids"],
                "pos": Spos + r["pos"],
                "decide": d,
                "opts": o,
                "logits": z.tolist(),
                "probs": F.softmax(z, -1).tolist(),
                "h_decide": h[d],
                "h_opts": h[torch.tensor(o)],
            }
        )
    return out


def main():
    tok, m, meta = load(merge=False)
    fixture = {
        "base": "Qwen/Qwen3.5-0.8B-Base",
        "base_revision": meta["base_revision"],
        "adapter": "jaredpalmer/kev-0.8b@v1.0",
        "temperature": float(meta["temperature"]),
        "head_dim": 256,
        "lora_r": 16,
        "lora_alpha": 32,
        "delimiters": kev.SPECIAL,
        "questions": [],
    }
    hidden = {}
    for ri, req in enumerate(REQUESTS):
        rec, qmeta = kev_api.to_record(kev_api.SystemOneRequest(**req))
        enc = kev.encode(
            tok, rec, max_state=kev.SERVE_MAX_STATE, max_branch=kev.SERVE_MAX_BRANCH, strict=True
        )
        ref = [p.tolist() for p in m.probs(enc)]
        rows = plain_rows(m, enc)
        for _, (qm, r, pa) in enumerate(zip(qmeta, rows, ref, strict=True)):
            key = f"r{ri}_{qm['id']}"
            hidden[f"{key}.h_decide"] = r["h_decide"].contiguous()
            hidden[f"{key}.h_opts"] = r["h_opts"].contiguous()
            fixture["questions"].append(
                {
                    "key": key,
                    "type": qm["type"],
                    "keys": qm["keys"],
                    "ids": r["ids"],
                    "pos": r["pos"],
                    "decide": r["decide"],
                    "opts": r["opts"],
                    "probs_kev": pa,
                    "probs_row": r["probs"],
                    "logits_row": r["logits"],
                }
            )
    del m
    _, mm, _ = load(merge=True)
    for _, req in enumerate(REQUESTS):
        rec, _ = kev_api.to_record(kev_api.SystemOneRequest(**req))
        enc = kev.encode(
            tok, rec, max_state=kev.SERVE_MAX_STATE, max_branch=kev.SERVE_MAX_BRANCH, strict=True
        )
        for r in plain_rows(mm, enc):
            q = next(q for q in fixture["questions"] if q["ids"] == r["ids"])
            q["probs_merged"] = r["probs"]
    worst = {
        k: max(
            abs(a - b)
            for q in fixture["questions"]
            for a, b in zip(q["probs_kev"], q[k], strict=True)
        )
        for k in ("probs_row", "probs_merged")
    }
    fixture["max_abs_dp_vs_kev"] = worst
    (OUT / "kev08b_fixture.json").write_text(json.dumps(fixture, indent=1))
    save_file(hidden, str(OUT / "kev08b_hidden.safetensors"))
    save_file(
        {k: v.contiguous() for k, v in meta["head"].items()},
        str(OUT / "kev08b_head.safetensors"),
        metadata={"temperature": repr(float(meta["temperature"])), "head_dim": "256"},
    )
    # the merged fp32 backbone, under the base checkpoint's own tensor names, for the converter
    sd = {f"model.language_model.{k}": v.contiguous() for k, v in mm.lm.state_dict().items()}
    save_file(sd, str(OUT / "merged-backbone-fp32.safetensors"))
    for q in fixture["questions"]:
        print(
            f"{q['key']:22s} kev={[round(p, 4) for p in q['probs_kev']]}  row={[round(p, 4) for p in q['probs_row']]}  merged={[round(p, 4) for p in q['probs_merged']]}"
        )
    print("max |dp| vs kev:", worst)


if __name__ == "__main__":
    main()
