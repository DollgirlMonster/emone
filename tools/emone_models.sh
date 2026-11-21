#!/usr/bin/env bash
# Shared model and client catalogue for every launcher and start script.
#
# One place decides which installs exist, what they are called and where
# they live, so the server launcher, the CLI launcher and the eight
# per-model start scripts cannot drift apart. A launcher that resolved the
# model itself and a start script that hardcoded a port is exactly how the
# "-fast" alias and the width-suffixed model id got out of step before.
#
# The client list lives here for the same reason: the launcher launches them
# and the coder benchmark measures them, so both read one list and a test
# (benchmark/test_coder_clients.py) fails if either drifts from it.
#
# Every model serves on one port. The server switches models on demand by
# name, one resident at a time, so a port per install only left client
# configs to keep in step. EMONE_PORT still overrides it: the memory
# harness runs its own server on 8096 that way.
EMONE_DEFAULT_PORT=8080

# --- Clients ---------------------------------------------------------------
#
# Every client this checkout launches or measures, in menu order:
#
#   id|label|kind|binaries
#
# `coder` clients are the ones a benchmark can ask a question and score; each
# has a command builder in benchmark/coder_cli_benchmark.py. `editor` clients
# are configured and opened instead: Zed's CLI has no non-interactive prompt
# mode (it opens windows, diffs files and pipes stdin), so it can be wired and
# checked, never prompted.
EMONE_CLIENTS=(
  "codex|Codex|coder|codex"
  "claude|Claude Code|coder|claude"
  "qwen|Qwen Code|coder|qwen qwen-code"
  "opencode|OpenCode|coder|opencode"
  "zed|Zed editor|editor|zed"
)

# --- What a person is offered to install ------------------------------------
#
# The install menu, in the order it is shown:
#
#   key|label|bits|installed GB|what it is for
#
# `key` is an `install_models.sh` target and a `EmoneRepack --model` selector.
# `label` is what the menu prints; it is spelled out here rather than derived,
# because `emone_resolve_model` takes the base keys (ornith15, qwen38) and
# not the `-8bit` targets or `qwen38flash`, and a menu that fell back to printing
# `qwen35-4b-8bit` at a person is not a menu.
#
# The sizes are **installed** sizes, not downloads: a MoE install is built from a
# ~70 GB bf16 fetch and the widths share it, so the number a person has to have
# free is this one. The first row is the default, and it is the model every
# release is verified against.
#
# Kept here rather than in the installer so the installer, the model installer
# and any future menu cannot disagree about what exists or how big it is.
# shellcheck disable=SC2034  # consumed by install_models.sh and the installer's menu
EMONE_MODEL_CHOICES=(
  "qwen36|Qwen 3.6 35B-A3B|4|19.5|Recommended. The general-purpose default."
  "qwen36-8bit|Qwen 3.6 35B-A3B|8|37.8|The same model at 8-bit: twice the disk, slower decode."
  "ornith15-8bit|Ornith 1.5 35B-A3B|8|36.9|Coding fine-tune; best measured coding and tooling results."
  "ornith15|Ornith 1.5 35B-A3B|4|19.5|The coding fine-tune at 4-bit."
  "agentworld-8bit|Qwen-AgentWorld 35B-A3B|8|41.0|Tuned for agentic tool use."
  "agentworld|Qwen-AgentWorld 35B-A3B|4|20.0|Agentic tool use at 4-bit."
  "katcoder-8bit|KAT-Coder-V2.5-Dev 35B-A3B|8|38.0|Coding-specialised Qwen 3.6 fine-tune."
  "katcoder|KAT-Coder-V2.5-Dev 35B-A3B|4|20.0|Coding-specialised, at 4-bit."
  "qwen35-4b|Qwen 3.5 4B|4|3.7|Dense and small: the quickest way to see it work, and it runs on the CPU too."
  "qwen35-4b-8bit|Qwen 3.5 4B|8|5.3|Dense 4B at 8-bit."
  "qwen35-9b|Qwen 3.5 9B|4|7.5|Dense 9B: more capable, still light on disk."
  "qwen35-9b-8bit|Qwen 3.5 9B|8|11.0|Dense 9B at 8-bit."
  "qwen35-2b|Qwen 3.5 2B|4|2.3|Dense 2B: seconds to load. Weak at arithmetic and long instructions."
  "qwen35-2b-8bit|Qwen 3.5 2B|8|2.2|Dense 2B at 8-bit."
  "qwen35-0.8b-8bit|Qwen 3.5 0.8B|8|1.0|Side model for session titles and one-word answers, not a chat model."
  "qwen35-0.8b|Qwen 3.5 0.8B|4|0.7|The side model at 4-bit."
  "qwen38-27b|Qwen3.8-27B|4|17.0|Dense 27B, Qwen3.8 generation: a general assistant, all in RAM."
  "qwen38-27b-8bit|Qwen3.8-27B|8|29.0|Dense 27B at 8-bit."
  "qwen38flash|Qwen3.8-Flash-Next 125B-A6B|4|162.0|The largest model, and the most disk."
  "qwen38flash-8bit|Qwen3.8-Flash-Next 125B-A6B|8|220.0|The largest model at 8-bit."
)

# emone_client_ids -> one client id per line, in menu order.
emone_client_ids() {
  local entry
  for entry in "${EMONE_CLIENTS[@]+"${EMONE_CLIENTS[@]}"}"; do printf '%s\n' "${entry%%|*}"; done
}

# emone_client_field <id> <2 label | 3 kind | 4 binaries> -> that field.
# Returns 1 for an id the list does not carry.
emone_client_field() {
  local entry rest
  for entry in "${EMONE_CLIENTS[@]+"${EMONE_CLIENTS[@]}"}"; do
    [[ "${entry%%|*}" == "$1" ]] || continue
    rest="${entry#*|}"
    case "$2" in
      2) printf '%s\n' "${rest%%|*}" ;;
      3) rest="${rest#*|}"; printf '%s\n' "${rest%%|*}" ;;
      4) printf '%s\n' "${rest#*|}" ;;
      *) return 1 ;;
    esac
    return 0
  done
  return 1
}

emone_client_label()    { emone_client_field "$1" 2; }
emone_client_kind()     { emone_client_field "$1" 3; }
emone_client_binaries() { emone_client_field "$1" 4; }

# emone_client_ids_csv <separator> -> "codex<sep>claude<sep>…", for help text.
emone_client_ids_csv() {
  local separator="${1:-|}" first=1 id
  while IFS= read -r id; do
    (( first )) || printf '%s' "$separator"
    printf '%s' "$id"
    first=0
  done < <(emone_client_ids)
  printf '\n'
}

# emone_coder_client_ids -> just the clients a benchmark can prompt.
emone_coder_client_ids() {
  local id
  while IFS= read -r id; do
    [[ "$(emone_client_kind "$id")" == "coder" ]] && printf '%s\n' "$id"
  done < <(emone_client_ids)
}

# emone_resolve_model <key> -> EMONE_MODEL_{KEY,STEM,LABEL,ENGINES,THINKING,FAMILY}
#
# ENGINES, THINKING and FAMILY are what the built-in fallback list needs to
# describe an install truthfully when the server cannot report its catalog:
# which engines can serve the family, default first, the thinking levels its
# chat template renders, and the family's own name as the runtime spells it
# (the engine message names it). The catalog is still the better source -- it
# reads all three off the install -- and the fallback is only reached without
# it.
emone_resolve_model() {
  case "${1:-}" in
    ornith|ornith15|ornith1.5)
      EMONE_MODEL_KEY=ornith
      EMONE_MODEL_STEM="ornith-1.5_35B_A3B"
      EMONE_MODEL_LABEL="Ornith 1.5 35B-A3B"
      EMONE_MODEL_FAMILY=qwen36
      EMONE_MODEL_ENGINES=gpu
      EMONE_MODEL_THINKING="off,on" ;;
    qwen36|qwen3.6)
      EMONE_MODEL_KEY=qwen36
      EMONE_MODEL_STEM="qwen3.6_35B_A3B"
      EMONE_MODEL_LABEL="Qwen 3.6 35B-A3B"
      EMONE_MODEL_FAMILY=qwen36
      EMONE_MODEL_ENGINES=gpu
      EMONE_MODEL_THINKING="off,on" ;;
    agentworld|aw)
      EMONE_MODEL_KEY=agentworld
      EMONE_MODEL_STEM="qwen-agentworld_35B_A3B"
      EMONE_MODEL_LABEL="Qwen-AgentWorld 35B-A3B"
      EMONE_MODEL_FAMILY=qwen36
      EMONE_MODEL_ENGINES=gpu
      EMONE_MODEL_THINKING="off,on" ;;
    katcoder|kat|kat-coder)
      EMONE_MODEL_KEY=katcoder
      EMONE_MODEL_STEM="kat-coder-v2.5_35B_A3B"
      EMONE_MODEL_LABEL="KAT-Coder-V2.5-Dev 35B-A3B"
      EMONE_MODEL_FAMILY=qwen36
      EMONE_MODEL_ENGINES=gpu
      EMONE_MODEL_THINKING="off,on" ;;
    qwen38|qwen3.8)
      EMONE_MODEL_KEY=qwen38
      EMONE_MODEL_STEM="qwen3.8-flash-next_125B_A6B"
      EMONE_MODEL_LABEL="Qwen3.8-Flash-Next 125B-A6B"
      EMONE_MODEL_FAMILY=qwen38flash
      EMONE_MODEL_ENGINES=gpu
      EMONE_MODEL_THINKING="off,low,medium,xhigh" ;;
    # The dense models: the one shape both engines implement, GPU by default.
    qwen35-0.8b|qwen3.5-0.8b)
      EMONE_MODEL_KEY=qwen35-0.8b
      EMONE_MODEL_STEM="qwen3.5_0.8B"
      EMONE_MODEL_LABEL="Qwen 3.5 0.8B"
      EMONE_MODEL_FAMILY=qwen3_5_dense
      EMONE_MODEL_ENGINES="gpu,cpu"
      EMONE_MODEL_THINKING="off,on" ;;
    qwen35-2b|qwen3.5-2b)
      EMONE_MODEL_KEY=qwen35-2b
      EMONE_MODEL_STEM="qwen3.5_2B"
      EMONE_MODEL_LABEL="Qwen 3.5 2B"
      EMONE_MODEL_FAMILY=qwen3_5_dense
      EMONE_MODEL_ENGINES="gpu,cpu"
      EMONE_MODEL_THINKING="off,on" ;;
    qwen35-4b|qwen3.5-4b)
      EMONE_MODEL_KEY=qwen35-4b
      EMONE_MODEL_STEM="qwen3.5_4B"
      EMONE_MODEL_LABEL="Qwen 3.5 4B"
      EMONE_MODEL_FAMILY=qwen3_5_dense
      EMONE_MODEL_ENGINES="gpu,cpu"
      EMONE_MODEL_THINKING="off,on" ;;
    qwen35-9b|qwen3.5-9b)
      # shellcheck disable=SC2034  # read by callers of emone_resolve_model
      EMONE_MODEL_KEY=qwen35-9b
      EMONE_MODEL_STEM="qwen3.5_9B"
      EMONE_MODEL_LABEL="Qwen 3.5 9B"
      EMONE_MODEL_FAMILY=qwen3_5_dense
      EMONE_MODEL_ENGINES="gpu,cpu"
      EMONE_MODEL_THINKING="off,on" ;;
    # Qwen3.8-27B: the dense family at 27B. Offered on the GPU only here; the
    # CPU engine implements the family but reads 16.5 GB per token.
    qwen38-27b|qwen3.8-27b)
      # shellcheck disable=SC2034  # read by callers of emone_resolve_model
      EMONE_MODEL_KEY=qwen38-27b
      EMONE_MODEL_STEM="qwen3.8_27B"
      EMONE_MODEL_LABEL="Qwen3.8-27B"
      EMONE_MODEL_FAMILY=qwen3_5_dense
      EMONE_MODEL_ENGINES=gpu
      EMONE_MODEL_THINKING="off,on" ;;
    *)
      echo "unknown AI model: ${1:-} (ornith|qwen36|agentworld|katcoder|qwen38|qwen35-0.8b|qwen35-2b|qwen35-4b|qwen35-9b|qwen38-27b)" >&2
      return 2 ;;
  esac
}

# emone_resolve_quant <4|8|4bit|8bit> -> EMONE_QUANT ("4bit"/"8bit"), EMONE_QUANT_DIR ("4Bit"/"8Bit")
emone_resolve_quant() {
  # shellcheck disable=SC2034  # EMONE_QUANT/QUANT_DIR are read by the caller
  case "${1:-}" in
    4|4bit) EMONE_QUANT=4bit; EMONE_QUANT_DIR=4Bit ;;
    8|8bit) EMONE_QUANT=8bit; EMONE_QUANT_DIR=8Bit ;;
    *) echo "unknown quantization: ${1:-} (4|8)" >&2; return 2 ;;
  esac
}

# emone_model_port -> the one port every model is served on. Kept as a
# function because callers and notes still ask for it by name.
emone_model_port() {
  echo "$EMONE_DEFAULT_PORT"
}

# Every model this checkout knows about, for help text and for the fallback
# list when the server cannot report its catalog. One list: the catalog is
# still the source of what is installed.
EMONE_ALL_MODELS=(ornith qwen36 agentworld katcoder qwen38 qwen35-0.8b qwen35-2b qwen35-4b qwen35-9b qwen38-27b)

# --- Installed models, from the server's catalog -----------------------
#
# The server knows what is installed -- GPU installs and CPU snapshots,
# their ids, widths and the thinking levels each one supports -- so the
# launcher asks it instead of keeping a second list that goes stale:
# `EmoneServer --catalog --models-dir <dir>` prints JSON. EMONE_CATALOG_JSON
# reads the same JSON from a file instead, so the launcher can be exercised
# against tools/testdata/catalog-example.json without a build.
#
# emone_load_catalog <binary> <models-dir> fills parallel arrays, one entry
# per model and quantization, GPU entries first:
#   EMONE_CAT_ID  EMONE_CAT_NAME  EMONE_CAT_QUANT (4|8)  EMONE_CAT_BACKEND (gpu|cpu)
#   EMONE_CAT_PATH  EMONE_CAT_THINKING (comma-separated, off first)  EMONE_CAT_SIZE (GB, or -)
#   EMONE_CAT_FAMILY (the manifest's family, e.g. qwen3_5_dense)
#   EMONE_CAT_ENGINES (comma-separated engines that can serve it, default first)
# and returns 1 with the reason in EMONE_CATALOG_ERROR when there is no
# usable catalog.

# Levels in the order they are offered; anything newer the server adds is
# kept, after these.
EMONE_CATALOG_PARSER='
import json, sys
LEVELS = ["off", "on", "minimal", "low", "medium", "high", "xhigh", "max"]
def field(value):
    text = str(value)
    if not text or any(c in text for c in "\t\r\n"):
        raise ValueError("empty or multi-line field")
    return text
try:
    models = json.load(sys.stdin)["models"]
except (ValueError, KeyError, TypeError) as error:
    sys.exit("catalog: not the expected JSON (%s)" % error)
if not isinstance(models, list):
    sys.exit("catalog: \"models\" is not a list")
gpu, cpu = [], []
for model in models:
    try:
        backend = field(model["backend"])
        if backend not in ("gpu", "cpu"):
            raise ValueError("backend %r" % backend)
        listed = [str(level) for level in (model.get("thinking") or ["off"])]
        levels = [l for l in LEVELS if l in listed] + [l for l in listed if l not in LEVELS]
        size = model.get("size_gb")
        family = field(model.get("family") or "-")
        engines = field(model.get("engines") or backend)
        row = [field(model["id"]), field(model["name"]), field(int(model["quant"])),
               backend, field(model["path"]), field(",".join(levels)),
               "%.1f" % float(size) if size is not None else "-", family, engines]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        name = model.get("id") if isinstance(model, dict) else model
        print("catalog: skipping %r (%s)" % (name, error), file=sys.stderr)
        continue
    (gpu if backend == "gpu" else cpu).append(row)
for row in gpu + cpu:
    print("\t".join(row))
'

emone_reset_catalog() {
  EMONE_CAT_ID=(); EMONE_CAT_NAME=(); EMONE_CAT_QUANT=(); EMONE_CAT_BACKEND=()
  EMONE_CAT_PATH=(); EMONE_CAT_THINKING=(); EMONE_CAT_SIZE=(); EMONE_CAT_FAMILY=()
  EMONE_CAT_ENGINES=(); EMONE_CATALOG_MISSING=()
}

emone_load_catalog() {
  local binary="$1" models_dir="$2" json rows
  local id name quant backend path thinking size family engines
  EMONE_CATALOG_ERROR=""
  emone_reset_catalog
  if [[ -n "${EMONE_CATALOG_JSON:-}" ]]; then
    if ! json="$(cat "$EMONE_CATALOG_JSON" 2>/dev/null)"; then
      EMONE_CATALOG_ERROR="cannot read EMONE_CATALOG_JSON=$EMONE_CATALOG_JSON"
      return 1
    fi
  elif [[ ! -x "$binary" ]]; then
    EMONE_CATALOG_ERROR="no server binary at $binary"
    return 1
  elif ! json="$("$binary" --catalog --models-dir "$models_dir" 2>/dev/null)"; then
    EMONE_CATALOG_ERROR="$(basename "$binary") --catalog failed; this build may predate it"
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    EMONE_CATALOG_ERROR="python3 is needed to read it"
    return 1
  fi
  if ! rows="$(printf '%s' "$json" | python3 -c "$EMONE_CATALOG_PARSER")"; then
    EMONE_CATALOG_ERROR="its output could not be read"
    return 1
  fi
  # Tab-separated with no empty fields (the parser writes "-"), because
  # read collapses runs of tabs and an empty column would shift the rest.
  while IFS=$'\t' read -r id name quant backend path thinking size family engines; do
    [[ -n "$id" ]] || continue
    EMONE_CAT_ID+=("$id"); EMONE_CAT_NAME+=("$name"); EMONE_CAT_QUANT+=("$quant")
    EMONE_CAT_BACKEND+=("$backend"); EMONE_CAT_PATH+=("$path")
    EMONE_CAT_THINKING+=("$thinking"); EMONE_CAT_SIZE+=("$size")
    EMONE_CAT_FAMILY+=("$family"); EMONE_CAT_ENGINES+=("$engines")
  done <<< "$rows"
  if (( ${#EMONE_CAT_ID[@]} == 0 )); then
    EMONE_CATALOG_ERROR="it lists no installed models"
    return 1
  fi
}

# emone_catalog_keep_installed: drop every entry whose install directory is
# gone, so what the launcher presents is what the disk holds whatever the
# catalog came from. EMONE_CATALOG_JSON can name an older build's output and a
# model can be deleted between two runs; neither may leave a model or a width
# on the menu that cannot be loaded. Returns 1 when nothing is left.
emone_catalog_keep_installed() {
  local i
  local ids=() names=() quants=() backends=() paths=()
  local thinking=() sizes=() families=() engines=()
  for (( i = 0; i < ${#EMONE_CAT_ID[@]}; i++ )); do
    [[ -e "${EMONE_CAT_PATH[$i]}" ]] || continue
    ids+=("${EMONE_CAT_ID[$i]}"); names+=("${EMONE_CAT_NAME[$i]}")
    quants+=("${EMONE_CAT_QUANT[$i]}"); backends+=("${EMONE_CAT_BACKEND[$i]}")
    paths+=("${EMONE_CAT_PATH[$i]}"); thinking+=("${EMONE_CAT_THINKING[$i]}")
    sizes+=("${EMONE_CAT_SIZE[$i]}"); families+=("${EMONE_CAT_FAMILY[$i]}")
    engines+=("${EMONE_CAT_ENGINES[$i]}")
  done
  emone_reset_catalog
  (( ${#ids[@]} > 0 )) || return 1
  EMONE_CAT_ID=("${ids[@]+"${ids[@]}"}"); EMONE_CAT_NAME=("${names[@]+"${names[@]}"}")
  EMONE_CAT_QUANT=("${quants[@]+"${quants[@]}"}"); EMONE_CAT_BACKEND=("${backends[@]+"${backends[@]}"}")
  EMONE_CAT_PATH=("${paths[@]+"${paths[@]}"}"); EMONE_CAT_THINKING=("${thinking[@]+"${thinking[@]}"}")
  EMONE_CAT_SIZE=("${sizes[@]+"${sizes[@]}"}"); EMONE_CAT_FAMILY=("${families[@]+"${families[@]}"}")
  EMONE_CAT_ENGINES=("${engines[@]+"${engines[@]}"}")
}

# emone_static_catalog <models-dir>: the same arrays from the built-in list,
# for a server that cannot report its catalog. No ids: such a server reports
# its one id once up.
#
# Only installs that are on disk are listed: this menu is what the person is
# about to load, so a model or a width that is not in models/ is not offered.
# The labels left out are collected in EMONE_CATALOG_MISSING, so the caller can
# say in one line what Emone supports but this checkout does not have. Returns
# 1 with EMONE_CATALOG_ERROR set when models/ holds none of them.
emone_static_catalog() {
  local models_dir="$1" key bits dir added
  local missing=()
  emone_reset_catalog
  for key in "${EMONE_ALL_MODELS[@]+"${EMONE_ALL_MODELS[@]}"}"; do
    emone_resolve_model "$key"
    added=0
    for bits in 8 4; do
      emone_resolve_quant "$bits"
      dir="$models_dir/${EMONE_MODEL_STEM}_${EMONE_QUANT_DIR}"
      [[ -e "$dir" ]] || continue
      EMONE_CAT_ID+=("-"); EMONE_CAT_NAME+=("$EMONE_MODEL_LABEL"); EMONE_CAT_QUANT+=("$bits")
      EMONE_CAT_BACKEND+=("${EMONE_MODEL_ENGINES%%,*}")
      EMONE_CAT_PATH+=("$dir"); EMONE_CAT_THINKING+=("$EMONE_MODEL_THINKING")
      EMONE_CAT_SIZE+=("-"); EMONE_CAT_FAMILY+=("$EMONE_MODEL_FAMILY")
      EMONE_CAT_ENGINES+=("$EMONE_MODEL_ENGINES")
      added=1
    done
    (( added )) || missing+=("$EMONE_MODEL_LABEL")
  done
# shellcheck disable=SC2034  # read by server_launcher.sh and install_models.sh
  (( ${#missing[@]} > 0 )) && EMONE_CATALOG_MISSING=("${missing[@]+"${missing[@]}"}")
  if (( ${#EMONE_CAT_ID[@]} == 0 )); then
    # shellcheck disable=SC2034  # read by dsh_route.sh and server_launcher.sh
    EMONE_CATALOG_ERROR="no install under $models_dir matches the built-in list"
    return 1
  fi
}

# emone_catalog_find_id <id> -> echoes the index of the entry with that id.
emone_catalog_find_id() {
  local i
  for (( i = 0; i < ${#EMONE_CAT_ID[@]}; i++ )); do
    if [[ "${EMONE_CAT_ID[$i]}" != "-" && "${EMONE_CAT_ID[$i]}" == "$1" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

# emone_catalog_find_dir <install directory name> -> echoes the index of the
# entry installed there. This is how an install key and a width (ornith 8)
# find their entry: the directory name is fixed by the installer, while
# the id is the server's to choose.
emone_catalog_find_dir() {
  local i
  for (( i = 0; i < ${#EMONE_CAT_PATH[@]}; i++ )); do
    if [[ "$(basename "${EMONE_CAT_PATH[$i]}")" == "$1" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

# --- Persistent memory -------------------------------------------------
#
# Off unless EMONE_MEMORY=1. Memory runs inside the server process, so there
# is no database to install or start, and it takes what it needs: measured,
# a hundred-chapter novel over ten sessions was about 100 KB. There is no
# RAM ceiling by default; EMONE_MEMORY_CACHE_MIB sets one for anyone who
# wants it. The files live in a dedicated folder under the checkout,
# <Emone>/memory, beside models/ (override with EMONE_MEMORY_DIR). One file
# per project; a project untouched for 30 days is deleted, and at most 100
# are kept (EMONE_MEMORY_RETENTION_DAYS, EMONE_MEMORY_MAX_WORKSPACES).
#
# The workspace defaults to the directory the launcher was run from, which
# is the repository being worked on, so two checkouts never share memory.

emone_export_memory_environment() {
  local workspace_dir="${1:-$PWD}"
  export EMONE_MEMORY="${EMONE_MEMORY:-0}"
  [[ "$EMONE_MEMORY" == "1" ]] || return 0
  # The home directory, its parent and the root are not projects. A server
  # launched from one and used for everything would put a novel and a
  # codebase in one fact store, so this refuses to start rather than mix.
  # The server applies the same rule on its own; this just says it earlier
  # and in the terminal the person is looking at.
  if [[ -z "${EMONE_MEMORY_WORKSPACE:-}" ]]; then
    local resolved home_dir
    resolved="$(cd "$workspace_dir" 2>/dev/null && pwd -P)"
    home_dir="$(cd "$HOME" 2>/dev/null && pwd -P)"
    if [[ "$resolved" == "$home_dir" || "$resolved" == "$(dirname "$home_dir")" || "$resolved" == "/" ]]; then
      echo "ERROR: EMONE_MEMORY=1 but this is launched from $resolved, which is not a project directory." >&2
      echo "       Memory would collect every project into one store. Either:" >&2
      echo "         cd <your project>   and run the start script again" >&2
      echo "       or name the workspace explicitly:" >&2
      echo "         EMONE_MEMORY_WORKSPACE=my-project <start script>" >&2
      exit 2
    fi
  fi
  export EMONE_WORKSPACE_DIR="${EMONE_WORKSPACE_DIR:-$workspace_dir}"
  export EMONE_MEMORY_NAMESPACE="${EMONE_MEMORY_NAMESPACE:-tinytitan}"
  local emone_root
  emone_root="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  export EMONE_MEMORY_DIR="${EMONE_MEMORY_DIR:-$emone_root/memory}"
}
