# Repository layout

What lives where, why it is arranged this way, and the conventions a new file
has to follow. Written after a structural pass over the tree (2026-09-11) that
split the oversized files and flattened one inconsistent test path; every claim
here was checked against the tree, and the counts are from that pass.

## Top level

| Path | What it is | Committed |
| --- | --- | --- |
| `sources/` | The Swift package's targets, one directory per target | yes |
| `tests/` | Test targets, mirroring `sources/` path for path | yes |
| `docs/` | Engineering documentation: plans, profiles, the findings register, the runbooks | yes |
| `tools/` | Build, install, verification and conversion drivers (`*.sh`, `*.py`) | yes |
| `benchmark/` | Benchmark scripts, the golden outputs the baseline compares against, launch helpers | yes |
| `plugins/` | Client-side bundles for tools that drive the server; `plugins/dsh-emone/` is the DeepSeek Harness one | yes |
| `assets/` | Brand assets (wordmark, slogans) | yes |
| `.build/`, `models/` | SwiftPM's build directory and the installed models | **no** — ignored, and never a source of truth |

`plugins/dsh-emone/` is a Node package (the DeepSeek Harness bundle), the one
non-Swift, non-Python deliverable here: it ships a `cordis.patch.yml`, a plugin
entry, a compaction backend and its own `node --test` suite, and it is installed
into a DSH profile as a `file:` dependency. It is plain ESM JavaScript because
that is what the harness loads.

The two ignored directories are the only large ones (`models/` is the whole
point of the project and is hundreds of GB; `.build/` is disposable). Nothing
else in the tree is generated, so a clean checkout is the tree plus whatever
models you installed.

## Targets

`sources/` holds one directory per SwiftPM target, and the name of the
directory is the name of the target:

- **`Emone`** — the runtime: the model, the forward runner, the kernels, the
  tokenizer. Subdirectories are *concerns*, not layers:
  `Kernels/` (Swift dispatch over the shaders), `Metal/` (the `.metal`
  sources), `Runtime/{Inference,Prefill,KVCache,Generation,Configuration,Family}`,
  `Infrastructure/{ModelIO,Streaming,Metal}`, `CPUEngine/`, `Tokenization/`.
- **`EmoneServer`, `EmoneCLI`, `EmoneRepack`, `EmoneMemoryTool`** — the four
  executables, each split into a SwiftPM-invisible `Command/` subdirectory
  (the `@main`/top-level entry) and a `Core/` library part that the tests
  import. `Package.swift` declares them as two targets each, with
  `exclude: ["Command"]` on the library half.
- **`EmoneFormat`, `EmoneMemory`, `ContinuityCore`** — the `.gturbo` format
  types, the memory layer, and the session/continuity engine. Each is a
  standalone library with its own README where its contract needs prose.
- **`EmoneBench`, `EmoneValidation`, `EmoneKernelsC`** — the benchmark
  harness, the validation/reference target, and the C kernels.

Two naming conventions follow from this and are worth stating, because both
were violated by exactly one file each and both violations were fixed in the
2026-09-11 pass:

1. **`main.swift` means top-level code.** A file with top-level statements is
   named `main.swift` (`EmoneCLI/Command`, `EmoneServer/Command`,
   `EmoneRepack/Command`, `EmoneMemoryTool`). A file whose entry point is
   `@main` is named after its type (`EmoneBench/EmoneBench.swift`,
   `ContinuityDemo/ContinuityDemo.swift`). `@main` in a `main.swift` happens to
   compile while the target is a single file and stops compiling the moment a
   second file joins the target — which is how `EmoneBench` was caught.
2. **Feature files are named for the type or the axis they extend**:
   `Model.swift` + `Model+Loading.swift`, `HTTPServerHandler.swift` +
   `HTTPServerHandler+{Routes,Chat,Responses,Anthropic,Plumbing}.swift`. A
   `+` name means "another file's type, one concern".

## Tests mirror sources

`tests/<Target>/...` mirrors the target's own directory shape, so a test is
found the same way the code is: `sources/Emone/Runtime/Prefill/X.swift` is
tested by `tests/Emone/Runtime/Prefill/...`. Where a target has a `Core/`
library half, the test path repeats it (`sources/EmoneServer/Core` ↔
`tests/EmoneServer/Core`).

One path was inconsistent and is now fixed: the `Emone` runtime's test target
sat at `tests/Emone/Core` while the runtime itself has no `Core/` level. It is
`tests/Emone` now, with the target renamed `EmoneTestsCore` → `EmoneTests`.

`tests/Emone/Runtime/qwen38_tensor_names.txt` and
`tests/EmoneRepack/Core/Support/qwen38_tensor_names.txt` are byte-identical
200 KB fixtures. **That duplication is required, not an oversight:** SwiftPM
resources belong to one target, the two files are resources of two different
test targets, and neither target may read the other's `Bundle.module`.
Removing one means one suite silently loses its fixture, so leave them.

## File size

There is no line-count ceiling on a *file*; the gate is on functions
(`tools/lint.sh`, 120 lines, with an inline `lint:allow-long <reason>`
exemption for orchestrators that are genuinely one sequence). File size is a
readability question, and the convention that came out of this pass is:

- **A file should hold one type, or one type plus the value types it speaks
  in.** When a file held a type and its supporting value types, those moved
  out (`ExpertCacheTypes.swift` out of `PreadExpertStreamer.swift`;
  `HTTPServerSupport.swift` out of `HTTPServer.swift`).
- **A file should hold one API surface or one phase.** `HTTPServerHandler` had
  grown to 2,111 lines covering routing, three API surfaces and the response
  plumbing; it is now six files, the largest 604 lines. The forward runner was
  already split by phase (`+Decode`, `+Prefill`, `+Residual`, `+MTP`), which is
  why those files stay large: each *is* one phase, and splitting a pipeline
  mid-sequence trades one long read for several functions with unwieldy
  signatures — the same argument its `lint:allow-long` comments already make.
- **Extracting a method to another file widens its access.** `private` is
  file-scoped in Swift, so a member reached from a new file of the same module
  becomes `internal`. That is the price of the split and the reason it is done
  only where the read improves; 95 members of `ServerHTTPHandler` and 4 members
  around `Model` were widened this pass, and nothing else changed.

Files still above 600 lines are, in order: `ServerInference.swift` (1,664),
`RealForwardRunner+Decode.swift` (1,572), `RealForwardRunner+Prefill.swift`
(1,524), `PreadExpertStreamer.swift` (1,391, one class),
`RealForwardRunner.swift` (1,357), `RemoteStreamingRepacker.swift` (1,234),
`Model.swift` (1,024; the test files below this line
are sized by case count, not by design). Each is one cohesive type or one phase
of a pipeline; the next structural gain there is a *design* change (a type doing
two jobs), not a move, and none is currently doing two jobs.

Three files were split out of that list on 2026-09-15, each as pure code motion
after the seam was checked to be self-contained — no `private` member reached
across the new boundary, so no access was widened:

| New file | Lines | Out of |
| --- | ---: | --- |
| `EmoneServer/Core/StructuredOutputDiagnostics.swift` | 245 | `ServerInference.swift` (1,897 → 1,664) |
| `Runtime/Inference/RealForwardRunner+DecodeAttention.swift` | 268 | `+Decode.swift` (1,829 → 1,572) |
| `Runtime/Inference/RealForwardRunner+PrefillAttention.swift` | 362 | `+Prefill.swift` (1,875 → 1,524) |

The rule for the next split is the one the pass above followed: move a *cluster*
— an entry point with its own helpers — never half of one pipeline, and check
for `private` members on both sides of the seam first, because a file-scoped
`private` reached from another file has to become `internal`.

## Generated and local files

`models/`, `.build/`, `.swiftpm/`, `benchmark/mock/`,
`benchmark/benchmark-results/`, `.qwen/` (the wiki clone), `.claude/` and
`memory/` are ignored. `benchmark/__pycache__` and `tools/__pycache__` are
Python's, ignored by the same rule as any `__pycache__`. `.DS_Store` is
ignored and should not be committed anywhere; the structural pass removed the
stray ones that had accumulated outside `.build/`.

## Where to start reading

- The runtime's entry point is `Model+Loading.swift` (`Model.load`) →
  `RealForwardRunner` → the phase files.
- The server's is `HTTPServer.swift` (the actor) → `HTTPServerHandler.swift`
  (the per-connection handler) → `HTTPServerHandler+Routes.swift`.
- The format is `docs/gturbo-format.md`; the memory layer is
  `sources/ContinuityCore/README.md` and `docs/agent-memory.md`.
- The state of the tree, including what is verified and what is not, is the
  project tracker in the wiki (`Project-Tracker`).
