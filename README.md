# This fork: what differs from upstream

This is [DollgirlMonster/TinyTitan](https://github.com/DollgirlMonster/TinyTitan),
a private-use fork of [Pummelchen/TinyTitan](https://github.com/Pummelchen/TinyTitan).
Everything below this section is upstream's README, unchanged. Its benchmarks were
measured on upstream's 24 GB M3, not here. The fork's own changes:

**Prefill for Qwen3.8-Flash-Next 125B-A6B 4-bit, tuned on an M1 Max** (32-core
GPU, 64 GB, model on an external Thunderbolt NVMe). On a 16,931-token prompt:

| step | prefill s | tok/s | output |
| --- | ---: | ---: | --- |
| upstream engine (estimated) | ~485 | ~35 | reference |
| + grouped-query QSA attention kernel | 419-451 | 39 | identical |
| + PLE n-gram rows read concurrently | 353-389 | 46 | identical |
| + 8,192-token chunks, parallel host key selection | 297-301 | 57 | identical |
| + remaining scalar GEMMs and the shared expert on the MPP tensor ops | 246-251 | 68 | rounding |
| + routed-expert tiles as grouped MPP GEMMs, run concurrently | 213-215 | 79 | rounding |
| + QSA indexer and QSA attention on the simdgroup matrix units, 16,384-token chunks | **156** | **108** | rounding |

About **3.1x** faster than the upstream engine; 2.8x against the first
measured baseline. The steps that change rounding passed a paired surprisal
test against the switch-free engine: +0.011 nats per token over 512
teacher-forced tokens, t = +0.92, which is no measurable change. The full
record is [`docs/m1-prefill-spike.md`](docs/m1-prefill-spike.md).

- **Where the settings live.** The two QSA matrix-unit kernels are on by
  default, and only Qwen3.8 uses QSA. The MPP switches and the 16K chunk are
  set in Qwen3.8 4-bit's `ModelProfile` row, so no other model's output
  changes. `TINYTITAN_PREFILL_MPP_WIDE`, `TINYTITAN_PREFILL_ROUTED_MPP`,
  `TINYTITAN_QSA_SCORE_MMA` and `TINYTITAN_PREFILL_QSA_MMA` (each `0` or `1`)
  override them. The profile's chunk is lowered to what a YaRN context allows.
- **Prefill chunks of 8,192 and 16,384 tokens are allowed.** Each chunk
  streams nearly the whole routed-expert corpus, so the number of chunks is
  the cost.
- **`tools/serve.sh <install>` is a one-line launcher** for an install
  anywhere, including an external drive. It loads the model on the first
  request, releases it after 15 idle minutes, and keeps the prefix cache on
  disk.
- **Server additions.** Token-prefix frontier checkpoints restore the deepest
  cached prefix after a system-prompt edit. `--prompt-cache-memory-ttl-seconds`
  releases idle RAM snapshots, and `GET /v1/prefill-progress` reports prefill
  progress.
- **Measurement tools.** `tools/m1_spike.sh` builds, tests and runs
  interleaved A/B prefill arms, reporting GPU work in gigacycles.
  `TinyTitanCLI --score` and `tools/prefill_surprisal_ab.sh` run the paired
  surprisal test for any prefill numerics change.

---

<p align="center">
  ✨ NVMAI is now TinyTitan ! ✨
</p>

<p align="center">
  <img width="1254" height="1254" alt="1" src="https://github.com/user-attachments/assets/2a990de5-af24-48cd-b673-e875755741cc" />
</p>

# TinyTitan

[![Stars](https://img.shields.io/github/stars/Pummelchen/TinyTitan?style=flat-square&logo=github&label=Stars&color=e3b341)](https://github.com/Pummelchen/TinyTitan/stargazers)
[![Views (14d)](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Pummelchen/TinyTitan/main/.github/traffic.json)](https://github.com/Pummelchen/TinyTitan)
[![Last Commit](https://img.shields.io/github/last-commit/Pummelchen/TinyTitan?style=flat-square&logo=git&label=Last%20Commit&color=2ea44f)](https://github.com/Pummelchen/TinyTitan/commits/main)
[![Contact](https://img.shields.io/badge/Contact-0xa0b1%40gmail.com-blue?style=flat-square&logo=gmail&logoColor=white)](mailto:0xa0b1@gmail.com)

TinyTitan is the fastest SSD streamer for AI models on a Mac.

What is new in each release lives in the
[Changelog](https://github.com/Pummelchen/TinyTitan/wiki/Changelog).

<br>

✨ Full DeepSeek Harness Support with Plugins: <br>
TinyTitan now is a first class member of [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) - including a Qwen 3.8 plugin with specific optimizations for TinyTitan.
Awesome: You can also manage dsh workspaces and sessions which are distributed over many Mac's in your LAN/Tailscale network using our dsh plugin "dsh-lan-manager" and the manager cli. 


<br>

✨ New Research Project: <br>
[TinyTitan Datacenter](https://github.com/Pummelchen/TinyTitan_Datacenter) - Run large MOE LLM models on a distributed cluster of Mac Mini/Studio's
while keeping them on SSD/NVMe for near linear scale of decode tok/s.



## Benchmarks

Peak decode on a base 8-core M3 MacBook Pro with 24 GB. 
`NA` means the CPU engine
does not serve that model: the MoE families stream their experts on the GPU + ANE path,
and only the dense Qwen 3.5 models run on either engine.

| Model | Quantization | GPU | CPU |
| --- | --- | ---: | ---: |
| Qwen 3.5 2B (dense) | 4-bit | **53.73 tok/s** | **15.42 tok/s** |
| Qwen 3.5 2B (dense) | 8-bit | **32.77 tok/s** | **15.83 tok/s** |
| Qwen 3.5 4B (dense) | 4-bit | **26.18 tok/s** | **7.71 tok/s** |
| Qwen-AgentWorld 35B-A3B | 4-bit | **21.74 tok/s** | NA |
| Ornith 1.5 35B-A3B | 4-bit | **21.65 tok/s** | NA |
| Qwen 3.6 35B-A3B | 4-bit | **21.41 tok/s** | NA |
| KAT-Coder-V2.5-Dev 35B-A3B | 4-bit | **17.86 tok/s** | NA |
| Qwen 3.5 4B (dense) | 8-bit | **16.14 tok/s** | **7.04 tok/s** |
| Qwen 3.5 9B (dense) | 4-bit | **14.93 tok/s** | **4.07 tok/s** |
| Qwen 3.6 35B-A3B | 8-bit | **12.37 tok/s** | NA |
| Qwen-AgentWorld 35B-A3B | 8-bit | **12.28 tok/s** | NA |
| Ornith 1.5 35B-A3B | 8-bit | **11.93 tok/s** | NA |
| Qwen 3.5 9B (dense) | 8-bit | **8.90 tok/s** | **4.51 tok/s** |
| KAT-Coder-V2.5-Dev 35B-A3B | 8-bit | **6.91 tok/s** | NA |
| Qwen3.8-Flash-Next 125B-A6B | 4-bit | **5.46 tok/s** | NA |
| Qwen3.8-Flash-Next 125B-A6B | 8-bit | **2.10 tok/s** | NA |



### Supported LLMs

Every model installs at **4-bit and 8-bit**:

- **Qwen3.8-Flash-Next 125B-A6B**
- **KAT-Coder-V2.5-Dev 35B-A3B**
- **Qwen-AgentWorld 35B-A3B**
- **Ornith 1.5 35B-A3B**
- **Qwen 3.6 35B-A3B**
- **Qwen 3.5 9B**
- **Qwen 3.5 4B**
- **Qwen 3.5 2B**



### Usage

Already installed? These three commands cover what most people want, and the
[wiki Cookbook](https://github.com/Pummelchen/TinyTitan/wiki/Cookbook) has a
copy-paste recipe for each task — the API, a coding client, JSON output, memory,
long context, the CPU engine — with the output you should expect.

```bash
tools/install_models.sh                                         # what is installed
.build/release/TinyTitanCLI --model models/qwen3.5_2B_4Bit \
  --prompt "The capital of France is" --max-new 32 --temperature 0
tools/server_launcher.sh --client codex --model qwen38 --bits 4   # API + Codex
```

Generated text goes to **stdout**; the timing footer goes to **stderr**, so a
pipeline sees only the answer. Nothing installed yet? Start with
[Getting Started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started).

- **Easiest install:** one command checks the Mac, downloads TinyTitan's
  prebuilt arm64 binaries, optionally downloads a model, and leaves a
  `tinytitan` command that starts the server — then offers to start it, so you
  finish with a base URL to point a client at. Safe to re-run; a second run
  updates what is already installed instead of fetching it twice.
  ```bash
  bash -c "$(curl -fsSL https://raw.githubusercontent.com/Pummelchen/TinyTitan/main/tools/install_tinytitan.sh)"
  ```
  **Use that form, not `curl … | bash`.** A pipe makes the script's stdin the
  pipe, so it cannot ask anything and takes the default at every step — including
  the model download and the browser window. `bash -c "$(curl …)"` downloads the
  script first and runs it with your terminal still on stdin, so the questions
  work. From a clone or an unzipped download, `bash tools/install_tinytitan.sh`
  does the same and never needs `chmod +x`, and `--help` lists its flags.
  **Nothing needs Homebrew, Xcode, git or Node**, and no Python is needed to
  install or to start the server: the engine arrives built for `arm64`. Only the
  model *converters* use Python, and for the models the repacker streams itself
  the installer prints the command that needs none.
- **A chat window, if you want one:** the installer can also set up TinyTitan's
  own [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) — a
  local page in your browser with a prompt box, already pointed at the model you
  installed. `--web` does it in one step, or answer yes when the installer asks.
  It opens with `~/.local/bin/tinytitan-web` (or
  `tools/server_launcher.sh --web`). It is installed under `~/.tinytitan`, pinned
  to a tested DeepSeek Harness version, and kept isolated from any DeepSeek
  Harness you run yourself, so our copy and yours never share a config, a
  session or a port; see [tools/dsh_local.sh](tools/dsh_local.sh). This is a
  client in the browser, not a bundled app — the engine and its server remain the
  product.
- **OpenAI-compatible server:** A loopback Chat Completions and Responses API
  for starting TinyTitan and connecting supported coding clients.
- **One server, one port, one launcher:** `tools/server_launcher.sh` starts the
  API on its own, or starts it and opens one of the supported clients — Codex,
  Claude Code, Qwen Code, OpenCode or the Zed editor — wiring that client's
  provider config to the model the server advertises. It asks what to launch
  from one list of every installed model and quantization (GPU and CPU), the
  thinking level that model supports, and an optional RAM target for the server
  process (any whole GB from 4 up — the interactive question offers 4, 8, 16 or
  32; anything over 30% of the Mac's physical memory is
  warned about in red and used anyway, and the default is the install's own
  measured profile, which the runtime holds to a third of physical memory).
  It serves on `127.0.0.1:8080` by default — the launcher asks for the port,
  and `--port` or `TINYTITAN_PORT` sets it — and every other
  installed model stays available by name through the API; the server switches
  on demand, keeping one model resident at a time.

```bash
tools/server_launcher.sh                                       # interactive
tools/server_launcher.sh --client codex --model ornith --bits 4  # server + Codex
tools/server_launcher.sh --client zed --model qwen38 --bits 4 --ram 8
```

- **Persistent agent memory (optional):** With `TINYTITAN_MEMORY=1` the model gets
  memory that outlives a conversation, scoped per repository, with six memory
  tools the engine answers itself. It runs inside the server process, so there
  is no database to install and nothing to start. Off by default; see
  [docs/agent-memory.md](docs/agent-memory.md).
- **Three client protocols on one server:** OpenAI Chat Completions, the
  OpenAI Responses API (stored responses, `previous_response_id`, the full
  event grammar) and the Anthropic Messages API (`/v1/messages`,
  `count_tokens`, streaming), so Codex, Claude Code and the OpenAI and
  Anthropic SDKs all talk to the same model; see
  [docs/server-api.md](docs/server-api.md).
- **Enforced structured output:** a request may ask for JSON — `response_format`
  on Chat Completions, `text.format` on the Responses API,
  `output_config.format` on Messages — and the server compiles the schema into a
  byte-level grammar that masks the sampler on both engines, so the model can
  only emit a document the schema allows rather than being asked nicely for one.
  The supported schema subset is small and explicit, and everything outside it
  is refused by name; see [docs/structured-output.md](docs/structured-output.md).
- **Tested coding CLIs:** The launch workflow supports Codex, Claude Code, Qwen
  Code, OpenCode and the Zed editor against the local server; the coder benchmark
  scores the four that can be prompted (Claude Code through a loopback Anthropic
  shim) and checks every client's wiring without a model
  (`--round clients`); DeepSeek Harness reaches the server through its own
  `llm-pi-ai` provider route, which `tools/dsh_route.sh` generates from the
  installed models (and `plugins/dsh-tinytitan` keeps current inside the harness,
  adding a compaction backend that does not think) — see
  [Connect a client](https://github.com/Pummelchen/TinyTitan/wiki/OpenAI-Compatible-Server#connect-a-client).
- **CLI and tools:** TinyTitan also provides direct CLI generation, streaming
  responses, and client-authorized function-tool calls.


### Core Benefits

- TinyTitan streams LLM's faster than any other similar project.
- Run large MOE AI models on low RAM Apple Silicon Macs by keeping the AI model on SSD/NVMe. 
- A 125B model on 8 GB of RAM. TinyTitan streams experts straight from SSD, so model size is bounded by your disk space, not your memory.
- You set the RAM budget. TinyTitan stays inside it. Give it 4 GB or 8 GB — it holds the line, so your Mac stays responsive while the model runs.
- Apple Neural Engine acceleration for prompt processing - 2.3× faster than the GPU cores.
- Our own Metal kernels, our own engine. Purpose-built for Apple silicon and engineered to use your Mac at the physical limit.
- No MLX. No GGUF. TinyTitan ships its own high-speed model format and a converter that builds it straight from the original weights.
  

### Special Features

- **Bounded expert RAM:** The resident expert cache is sized per family from
  the model's own expert stride and clamped to a third of physical memory, so a
  smaller Mac is not handed a budget tuned on a larger one. `--ram` names a
  target for the whole server process, not the cache: the runtime subtracts the
  weight file and a measured ~0.5 GB runtime reserve, then steps down the slot
  ladder to the largest cache that fits, so an 8 GB target lands at about a 4 GB
  cache (32 slots) on the 125B install. It is wired, so it cannot be paged out
  and everything else the Mac is running has to fit beside it: the launcher
  recommends **30% of physical memory** and warns in red above it — swapping, a
  less stable system and slower tokens — but a larger target is your call and is
  passed on. The target covers the resident weights, the runtime floor and the
  cache for a short request; a long context's KV grows on top of it.
- **Long context:** Native RoPE supports up to 262K tokens, while optional YaRN
  extends the context to 512K or 1M tokens.
- **Compressed KV cache:** Live attention state can use 16-bit, 8-bit, or 4-bit
  storage independently of the installed model quantization.
- **Thinking mode:** Ornith and Qwen support truthful Off/On reasoning control;
  their chat templates do not define Low, Medium, or High effort levels.
- **MTP off by default:** Native speculative decoding remains experimental and
  disabled because measured Ornith runs showed no speed benefit and it
  currently requires greedy decoding, native RoPE, and prompt-cache reuse off.

### Performance Improvements

- **Tiled Top-K sampling:** Production sampling (Top-K 1–64) runs a
  three-stage tiled GPU reduction, cutting per-token sampling cost from
  15.5 ms to 1.4 ms with a token-for-token identical stream — the main
  source of the v4.6 decode gain.
- **ANE prefill:** `TINYTITAN_PREFILL_ANE=on` runs
  full-attention prefill blocks on the Neural Engine from a one-time
  exported Core ML sidecar, roughly halving long-prompt time to first
  token; short prompts and decode are untouched.
- **Follow-up cache:** Exact live and multi-prefix prompt-state reuse avoids
  repeating compatible prefill work across conversation turns.
- **Concise mode:** An optional terse system prompt reduces generated text for
  workloads that benefit from it; standard responses are the default because
  they generalized more reliably in the coding/tooling qualification.
- **Fast alias:** The chat-only `-fast` model alias strips coding-agent
  boilerplate before prefill for quicker direct answers, while the base alias
  preserves tools and agent loops.


## Core Links

- [Getting started](https://github.com/Pummelchen/TinyTitan/wiki/Getting-Started)
- [Features](https://github.com/Pummelchen/TinyTitan/wiki/Features)
- [Local server and launchers](https://github.com/Pummelchen/TinyTitan/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/TinyTitan/wiki/Runtime-Controls)
- [Benchmarks](https://github.com/Pummelchen/TinyTitan/wiki/Benchmarks)
- [Changelog](https://github.com/Pummelchen/TinyTitan/wiki/Changelog)
- [Repository layout](docs/repository-layout.md) — where everything lives, and
  the naming and file-size conventions

## Credits

TinyTitan is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), whose
bounded-memory runtime, installer, CLI and local server this project builds on.
The Qwen 3.6 integration was created by
[NeelM0906](https://github.com/NeelM0906) in
[upstream PR #29](https://github.com/drumih/turbo-fieldfare/pull/29). Concise
mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright (c) 2026 André Borchert.

## Contact

Questions, bug reports and suggestions are always welcome. You can contact André Borchert by email at [0xa0b1@gmail.com](mailto:0xa0b1@gmail.com).
