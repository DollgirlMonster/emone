# Handover: emone as the backend for Hermes Agent

**Paste this into the next session:**

> Continue setting up emone (this checkout) as the local model server for Hermes
> Agent. Read `AGENTS.md`, then `docs/handover-hermes.md`. The Mac's checkout is at
> `/Volumes/Chai/TinyTitan`, on branch `claude/practical-hypatia-o9a22c`, which
> carries the `tools/serve.sh` catalog change. The server has **not been started
> yet**, so the three model ids are not known. Next: start the server, read
> `/v1/models`, fill the ids into `~/.hermes/config.yaml` (template below), and
> time a model swap to settle where Hermes's compression runs. Separately, review
> the uncommitted work that was parked on the local-only branch
> `wip/local-server-changes` before anything is merged or deleted.

## The goal

Run Hermes Agent (Nous Research) against emone's loopback OpenAI-compatible
server, with the prompt cache keeping long agent sessions affordable, and switch
between the three installed models from inside Hermes.

## Where it stands

- **Installed models** (`/Volumes/Chai/TinyTitan/models`, all 4-bit):
  - "little Qwen": `qwen3.6_35B_A3B_4Bit` (Qwen 3.6 35B-A3B)
  - "big Qwen": `qwen38flash.gturbo` (Qwen3.8-Flash-Next 125B-A6B)
  - "the other one": `ornith-1.5_35B_A3B_4Bit` (Ornith 1.5 35B-A3B)

  There is no dense Qwen 3.5 install on this Mac.
- **`tools/serve.sh` serves the models folder by default** (commit `9a54f5a`
  on `claude/practical-hypatia-o9a22c`).
  - No argument: serves `<checkout>/models` (or `TINYTITAN_MODELS_DIR`) as the
    catalog.
  - A folder of installs may be named in place of one install.
  - A request naming no model gets `TINYTITAN_DEFAULT_MODEL` (default
    `qwen3.6_35B_A3B_4Bit`), else the first completed install.
  - `benchmark/test_serve_catalog.py` pins the cases with `--dry-run`. It passed
    on Linux; it has not run on the Mac, and the server has not been started
    with the change.
- **The branch switch on the Mac succeeded** after the local edits were parked
  (next section).

## Parked local work: `wip/local-server-changes` (local only, not pushed)

The Mac's checkout had uncommitted changes on `prefill-m1-kernels`. They were
committed as `6e59ec7b` on a new branch, `wip/local-server-changes`, so the
switch could happen without losing anything. Nobody in this session has read
them. That commit holds:

- edits to `sources/TinyTitanServer/Command/main.swift`, `Core/ModelRouter.swift`,
  `Core/ServerArguments.swift` and three server tests (about 150 lines). They
  may be an earlier attempt at catalog serving; compare them with `main` before
  reviving any of it.
- **deletions of `docs/deepseek-v41-flash-port.md` and
  `docs/deepseek-v41-flash-reference.md`** (about 1,800 lines), which look
  accidental. Do not merge the deletions without asking.
- `.beads/interactions.jsonl`.

Still untracked in the checkout: `benchmark/m1-spike/`, `benchmark/surprisal/`,
and `prompt-cache/` at the root. `serve.sh` writes to `.prompt-cache/` (dot,
gitignored), so `prompt-cache/` is probably from an older run; confirm before
removing it.

## Next steps

1. **Start the server** (terminal 1). The first run builds the server:
   `tools/serve.sh -- --max-context 65536`
2. **Read the ids** (terminal 2). Run the command without a trailing `#`
   comment, because the Mac's zsh does not treat `#` as a comment:
   `curl -s 127.0.0.1:8089/v1/models | jq -r '.data[].id'`
3. **Write `~/.hermes/config.yaml`** from the template below.
4. **Time a swap.** With the 125B loaded, time a one-token request to the little
   id, then one back to the big id. The two times together are the swap cost.
5. **Choose where compression runs** (the reasoning is in the next section).

## Hermes config template

```yaml
model:
  provider: custom
  base_url: http://127.0.0.1:8089/v1
  api_key: local-no-auth
  default: <little-qwen id>
  context_length: 65536            # equal to serve.sh --max-context

providers:
  emone:
    api: http://127.0.0.1:8089/v1
    transport: chat_completions
    models:
      <little-qwen id>: { context_length: 65536 }
      <big-qwen id>:    { context_length: 65536 }
      <ornith id>:      { context_length: 65536 }

auxiliary:
  compression:
    provider: custom
    base_url: http://127.0.0.1:8089/v1
    model: <little-qwen id>
```

Switch in a session with `/model custom:emone:<id>`. The key names come from
Hermes's published docs and have not been checked against a live Hermes; run
`hermes doctor` once the file is in place.

## Where compression should run

The server keeps one model resident at a time. A request naming another model
swaps it, so a compression call that names the little model while you chat with
the 125B costs two swaps. Numbers for this M1 Max (README fork section and
`docs/m1-prefill-spike.md`):

| | 125B | 35B |
| --- | ---: | ---: |
| Prefill | ~390 tok/s | ~550 tok/s |
| Decode | 5.6 tok/s | ~21-25 tok/s at 33-57K context |

**Example job:** 30K tokens in, a 2K-token summary out, with no cache hit
because the compression prompt is laid out differently from the chat.

- On the 125B: 77 s + 357 s, **about 7 min**.
- On the 35B: 55 s + 85 s, **about 2.3 min**, plus two swaps.

So the little model wins unless the two swaps together take over about 5 min.
The weights are memory-mapped. The only measured switches are 1.5-5 s, on dense
models, and nothing has been recorded for the MoE models on this Mac. After a
swap, the 125B's 16 GiB expert cache starts cold.

**Short, frequent side tasks** (titles, small checks) favour the opposite. A
1K-token prompt with a 30-token answer is about 8 s on the 125B, which is
cheaper than two swaps unless a load takes only a few seconds. Keep those on the
resident model, or on the cloud option.

**The cloud alternative:** Gemini through Ollama's cloud models, at
`http://localhost:11434/v1`. It runs on Ollama's compute servers, not Google's,
and needs `ollama signin`. It never causes a swap. Hermes authenticates to it
with `OPENAI_API_KEY`, which Ollama ignores, so a placeholder in
`~/.hermes/.env` is enough.

## Traps already hit

- **Uncommitted edits block `git switch`.** Check `git status` before changing
  branches on the Mac.
- **The Mac's zsh does not treat `#` as a comment.** A note after `#` on a
  pasted command line is passed on as arguments.
- **Pin no example path that is not installed.** An earlier reply pointed at a
  `qwen3.5_9B_4Bit` that does not exist here.
- **The server refuses forced tool calls and image inputs.** It serves only
  `tool_choice` `auto` and `none`, and refuses image inputs. Point Hermes's
  vision side task elsewhere, or turn it off.
- **Keep Hermes's `context_length` equal to `--max-context`.** The compression
  model's window must be at least the main model's.
