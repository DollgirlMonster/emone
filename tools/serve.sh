#!/usr/bin/env bash
# Serve one installed model on 127.0.0.1, tuned by its own profile row.
#
#   tools/serve.sh /Volumes/Chai/TinyTitan/models/qwen38flash.gturbo
#   TINYTITAN_MODEL=/path/to/install tools/serve.sh --idle 600
#   tools/serve.sh <install> --reasoning xhigh -- --kv-bits 4
#
# The short path for a model that lives anywhere, including an external
# drive; tools/server_launcher.sh is the full, interactive one for models/.
# Everything tuned per model -- the prefill chunk, the prefill kernel
# switches, the expert-cache budget, sampling -- comes from the install's
# ModelProfile row, so this script sets no TINYTITAN_* switch of its own.
# The one server default it does set is reasoning: on, so a thinking model
# thinks (the model's template default effort -- extra high on Qwen3.8).
#
# By default the server binds at once, loads the model on the first request
# and releases it after --idle seconds without one, so a model does not sit
# resident between sessions. The prefix cache is kept on disk
# (<checkout>/.prompt-cache/<install>, beside the models on the external
# drive rather than the internal SSD), so an unload does not cost the next
# request its whole prompt. Its budget is 100 GiB by default (--disk-cache-gib):
# a Qwen3.8 snapshot at 40K tokens is ~2.8 GB, so the server's own 8 GiB
# default keeps only about three conversations.
#
# Nothing here downloads or installs a model, and nothing is killed: another
# model process, or an install without its receipt, stops it.
set -Eeuo pipefail
trap 'echo "serve: stopped at line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="${TINYTITAN_MODEL:-}"
PORT="${TINYTITAN_PORT:-8089}"
IDLE=900
CACHE_ROOT="${TINYTITAN_PROMPT_CACHE_ROOT:-$ROOT/.prompt-cache}"
REASONING=on
DISK_GIB=100
DRY_RUN=0
EXTRA=()

usage() {
  sed -n '2,/^set -/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  cat <<'USAGE'
Options:
  <install>            model directory (or TINYTITAN_MODEL)
  --port <n>           default 8089 (TINYTITAN_PORT)
  --idle <seconds>     release the model after this long idle (default 900;
                       0 keeps it loaded from the first request on)
  --disk-cache-gib <n> SSD prompt-cache budget in GiB (default 100; 0 disables)
  --reasoning <level>  off, on, minimal, low, medium, high, xhigh or max
                       (default on); skipped when the server args after --
                       carry --reasoning, --thinking or --reasoning-effort
  --dry-run            print the server command and start nothing
  -- <server args>     passed to TinyTitanServer unchanged, last, so they win
USAGE
}

die() {
  echo "serve: $*" >&2
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --idle) IDLE="$2"; shift 2 ;;
    --reasoning) REASONING="$2"; shift 2 ;;
    --disk-cache-gib) DISK_GIB="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    -*) die "unknown option: $1 (see --help)" ;;
    *) MODEL="$1"; shift ;;
  esac
done

[ -n "$MODEL" ] || die "name an install: tools/serve.sh <model-dir> (or set TINYTITAN_MODEL)"
case "$MODEL" in
  /*) ;;
  *) MODEL="$PWD/$MODEL" ;;
esac
MODEL="${MODEL%/}"
case "$PORT$IDLE$DISK_GIB" in
  *[!0-9]*) die "--port, --idle and --disk-cache-gib take whole numbers" ;;
esac

[ -d "$MODEL" ] || die "$MODEL is not a directory (is the drive mounted?)"
[ -f "$MODEL/verified-install.json" ] || die "$MODEL is not a completed install (no verified-install.json)"

name="$(basename "$MODEL")"
cache_dir="$CACHE_ROOT/${name%.gturbo}"
server_args=(--model "$MODEL" --port "$PORT" --prompt-cache-disk "$cache_dir"
  --prompt-cache-disk-mib "$((DISK_GIB * 1024))")
if [ "$IDLE" -gt 0 ]; then
  server_args+=(--idle-unload-seconds "$IDLE")
fi
# The server refuses --reasoning alongside --thinking/--reasoning-effort, so
# a reasoning flag passed through after -- replaces this default outright.
for arg in ${EXTRA[@]+"${EXTRA[@]}"}; do
  case "$arg" in
    --reasoning | --thinking | --reasoning-effort) REASONING="" ;;
  esac
done
if [ -n "$REASONING" ]; then
  server_args+=(--reasoning "$REASONING")
fi

cd "$ROOT"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "TinyTitanServer ${server_args[*]+"${server_args[*]}"} ${EXTRA[*]+"${EXTRA[*]}"}"
  exit 0
fi

[ "$(uname -s)" = "Darwin" ] || die "needs macOS on Apple Silicon"
busy="$(pgrep -fl 'TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' || true)"
[ -z "$busy" ] || die "another model or test process is running; stop it first:
$busy"

# Always build: incremental, seconds when nothing changed, and it keeps a pull
# from serving on a stale binary.
swift build -c release --product TinyTitanServer >&2
BIN_DIR="$(swift build -c release --show-bin-path)"
mkdir -p "$cache_dir"
# Prefix-reuse measurement: hashes only, one line per request, read by
# benchmark/prefix_log_report.py. Set TINYTITAN_PREFIX_LOG= (empty) to turn it off.
export TINYTITAN_PREFIX_LOG="${TINYTITAN_PREFIX_LOG-$CACHE_ROOT/prefix-log/${name%.gturbo}.jsonl}"
if [ -n "$TINYTITAN_PREFIX_LOG" ]; then
  mkdir -p "$(dirname "$TINYTITAN_PREFIX_LOG")"
  echo "serve: prefix log $TINYTITAN_PREFIX_LOG" >&2
fi

echo "serve: $name on http://127.0.0.1:$PORT (idle unload: ${IDLE}s, prompt cache: $cache_dir)" >&2
exec "$BIN_DIR/TinyTitanServer" ${server_args[@]+"${server_args[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
