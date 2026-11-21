#!/usr/bin/env bash
# Runs the memory-value benchmarks arm by arm against Qwen 3.6 35B 4-bit.
#
#   benchmark/memval_run.sh smoke           # placement + one fact, ~3 minutes
#   benchmark/memval_run.sh pong            # control, auto, minimal, full
#   benchmark/memval_run.sh book            # summary, auto, minimal, full
#   benchmark/memval_run.sh pong full       # one arm
#
# The install: EMONE_MEMVAL_MODEL=ornith|qwen36|agentworld (default qwen36)
# and EMONE_MEMVAL_QUANT=4|8 (default 4). EMONE_MEMVAL_ARMS="summary auto"
# limits the arms. Results go under .build/benchmark-logs/memory-<bench>-
# <label>/ where the label is EMONE_MEMVAL_LABEL or "<model>-<quant>bit".
#
# Each arm gets a freshly started server with its own configuration and its
# own empty memory directory under the scratch root, so nothing an arm writes
# can reach another arm or the user's real ~/.emone/memory. The server is
# stopped between arms. Never build while this is running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${1:?pong|book}"
ONLY="${2:-}"
PORT="${EMONE_PORT:-8096}"
MODEL="${EMONE_MEMVAL_MODEL:-qwen36}"
QUANT="${EMONE_MEMVAL_QUANT:-4}"
case "$MODEL" in
  ornith|qwen36|agentworld) : ;;
  *) echo "EMONE_MEMVAL_MODEL must be ornith, qwen36 or agentworld" >&2; exit 2 ;;
esac
# One launcher for every model and client now; `codex` keeps the agent loop,
# which is what these benchmarks drove before.
LAUNCH=("$ROOT/tools/server_launcher.sh" codex full "$MODEL" "$QUANT" default off)
[[ -x "${LAUNCH[0]}" ]] || { echo "no launcher at ${LAUNCH[0]}" >&2; exit 2; }
LABEL="${EMONE_MEMVAL_LABEL:-$MODEL-${QUANT}bit}"
SCRATCH="${EMONE_MEMVAL_SCRATCH:-$ROOT/.build/benchmark-logs/memval-scratch-$LABEL}"
# One results directory per benchmark and install. `pong` and `book` keep the
# names their recorded runs already carry, because tools that read those runs
# -- the simulator, the watchdog calibration -- glob for them.
case "$BENCH" in
  pong)  BENCH_DIR=value ;;
  book|smoke) BENCH_DIR=book ;;
  master) BENCH_DIR="${EMONE_MASTER_SCENARIO:?set EMONE_MASTER_SCENARIO}" ;;
  *)     BENCH_DIR="$BENCH" ;;
esac
LOGS="$ROOT/.build/benchmark-logs/memory-$BENCH_DIR-$LABEL"
mkdir -p "$LOGS" "$SCRATCH"

case "$BENCH" in
  smoke)   SCRIPT="$ROOT/benchmark/memory_smoke.py";    ARMS=(auto) ;;   # no tools: consolidation is the only writer
  pong)    SCRIPT="$ROOT/benchmark/memory_value.py";    ARMS=(control auto minimal full) ;;
  book)    SCRIPT="$ROOT/benchmark/memory_book.py";     ARMS=(summary auto minimal full) ;;
  correct) SCRIPT="$ROOT/benchmark/memory_correct.py";  ARMS=(control auto) ;;
  projects) SCRIPT="$ROOT/benchmark/memory_projects.py"; ARMS=(control auto) ;;
  volume)  SCRIPT="$ROOT/benchmark/memory_volume.py";   ARMS=(control auto full) ;;
  # The ten master prompts: one scenario per invocation, named by
  # EMONE_MASTER_SCENARIO (see benchmark/memval_master.sh).
  master)  SCRIPT="$ROOT/benchmark/memory_master.py";   ARMS=(summary auto) ;;
  *) echo "usage: $0 smoke|pong|book|correct|projects|volume|master [arm]" >&2; exit 2 ;;
esac
[[ -n "$ONLY" ]] && ARMS=("$ONLY")
if [[ -n "${EMONE_MEMVAL_ARMS:-}" && "$BENCH" != smoke ]]; then
  read -r -a ARMS <<< "$EMONE_MEMVAL_ARMS"
fi
# Repeats. Only meaningful with sampling on: at temperature 0 a repeat is the
# same output, so the default leaves temperature to the server, which is what
# a real client does. EMONE_MEMVAL_TEMPERATURE=0 pins it for a determinism
# check.
RUNS="${EMONE_MEMVAL_RUNS:-3}"
# Where the run numbering starts. Repeats of one configuration have to be
# interleaved with the others to be worth anything -- three of A then three
# of B measures the order as much as the arms -- and interleaving means
# invoking this script once per run, which would otherwise overwrite run 1
# every time.
FIRST_RUN="${EMONE_MEMVAL_FIRST_RUN:-1}"
[[ "$BENCH" == smoke ]] && RUNS=1
# The server distils a session after this much quiet. Two minutes in
# production; here the harness waits for the log line, so keep it short.
IDLE="${EMONE_MEMVAL_CONSOLIDATION_IDLE:-5}"

BINARY="$ROOT/.build/release/EmoneServer"
if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: no release binary at $BINARY; run: swift build -c release --product EmoneServer" >&2
  exit 1
fi
# The release binary must be newer than every source file, or the arms
# measure whatever was last built. This is the check that was missing when
# three arms of numbers turned out to be the same arm.
# EmoneMemoryTool is the CLI; the server does not link it, so SwiftPM will
# not relink the server when it changes, and it is not measured here.
newest_source="$(find "$ROOT/sources" -path "$ROOT/sources/EmoneMemoryTool" -prune -o -name '*.swift' -newer "$BINARY" -print | head -1)"
if [[ -n "$newest_source" ]]; then
  echo "ERROR: $newest_source is newer than the release binary; rebuild first." >&2
  exit 1
fi

# --- provenance --------------------------------------------------------------
# The mtime check above catches a *stale* binary; it cannot catch a binary built
# from a *different* tree. An uncommitted edit compiled into a rebuild passes it
# silently, and a suite then measures two engines while claiming one -- which is
# what happened on 2026-09-20, when another session's expert-cache change was
# compiled into a rebuild halfway through a ten-world run. So refuse a dirty
# tree unless that is deliberate, and record what was actually measured.
DIRTY="$(git -C "$ROOT" status --porcelain)"
if [[ -n "$DIRTY" && "${EMONE_MEMVAL_ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "ERROR: the working tree is dirty, so a build from it is not the committed engine:" >&2
  printf '%s\n' "$DIRTY" | sed 's/^/       /' >&2
  echo "       commit or stash the changes and rebuild, or set EMONE_MEMVAL_ALLOW_DIRTY=1" >&2
  echo "       to measure a dirty tree deliberately." >&2
  exit 1
fi
{
  echo "commit: $(git -C "$ROOT" rev-parse HEAD)"
  echo "dirty: $([[ -n "$DIRTY" ]] && echo yes || echo no)"
  echo "dirty_allowed: ${EMONE_MEMVAL_ALLOW_DIRTY:-0}"
  echo "binary: $BINARY"
  echo "binary_sha256: $(shasum -a 256 "$BINARY" | awk '{print $1}')"
  echo "recorded_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$LOGS/run-meta.txt"
echo "  provenance: $(git -C "$ROOT" rev-parse --short HEAD) binary $(shasum -a 256 "$BINARY" | cut -c1-12) dirty=$([[ -n "$DIRTY" ]] && echo yes || echo no)"

# The PIDs actually listening on the port -- the server binary, never the
# launcher shell around it. Killing the launcher leaves the model running:
# bash does not forward a signal to the child it is waiting on, and an
# orphaned server then answers the next run's readiness poll with the
# previous run's configuration. That is exactly what the first smoke test
# did, and its "empty replies in 0 s" were the orphan shutting down under
# the requests.
listening_pids() { lsof -ti :"$PORT" -sTCP:LISTEN 2>/dev/null || true; }

stop_server() {
  local pid
  for pid in $(listening_pids); do
    ps -p "$pid" -o command= | grep -q EmoneServer && kill -TERM "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 90); do
    [[ -z "$(listening_pids)" ]] && break
    sleep 1
  done
  for pid in $(listening_pids); do kill -KILL "$pid" 2>/dev/null || true; done
  [[ -n "${LAUNCHER_PID:-}" ]] && kill "$LAUNCHER_PID" 2>/dev/null || true
  LAUNCHER_PID=""
}

# Ready means a completion answers, not that the port is open. The port is
# bound before the first token can be produced, and a poll on /v1/models
# has no way to tell a loaded server from one that is still reading
# weights.
wait_ready() {
  local deadline=$(( $(date +%s) + 900 )) model
  until [[ -n "$(listening_pids)" ]]; do
    if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
      echo "ERROR: launcher exited before the server bound; see $SERVER_LOG" >&2; return 1
    fi
    (( $(date +%s) > deadline )) && { echo "ERROR: no server after 15 min; see $SERVER_LOG" >&2; return 1; }
    sleep 3
  done
  until model="$(curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" \
                 | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)" \
        && [[ -n "$model" ]]; do
    (( $(date +%s) > deadline )) && { echo "ERROR: /v1/models never answered" >&2; return 1; }
    sleep 3
  done
  local body status
  body="$(printf '{"model":"%s","messages":[{"role":"user","content":"Say OK."}],"max_completion_tokens":4,"temperature":0}' "$model")"
  until status="$(curl -s -o /dev/null -w '%{http_code}' --max-time 600 \
                  -H 'Content-Type: application/json' -d "$body" \
                  "http://127.0.0.1:$PORT/v1/chat/completions")" && [[ "$status" == 200 ]]; do
    (( $(date +%s) > deadline )) && { echo "ERROR: server never answered a completion (last status $status)" >&2; return 1; }
    sleep 5
  done
  echo "ready: $model answers completions"
}

trap stop_server EXIT

for RUN in $(seq "$FIRST_RUN" $(( FIRST_RUN + RUNS - 1 ))); do
for ARM in "${ARMS[@]+"${ARMS[@]}"}"; do
  case "$ARM" in
    control|summary) MEMORY=0; TOOLS=off ;;
    auto)            MEMORY=1; TOOLS=off ;;      # memory on, no tools: the engine writes
    minimal)         MEMORY=1; TOOLS=minimal ;;
    full)            MEMORY=1; TOOLS=full ;;
    *) echo "unknown arm $ARM" >&2; exit 2 ;;
  esac
  MEMDIR="$SCRATCH/$BENCH-$ARM-r$RUN"
  rm -rf "$MEMDIR"; mkdir -p "$MEMDIR"
  SERVER_LOG="$LOGS/server-$ARM-r$RUN.log"

  echo "=== $BENCH / $LABEL / $ARM / run $RUN  (memory=$MEMORY memory_tools=$TOOLS consolidation_idle=${IDLE}s dir=$MEMDIR port=$PORT)"
  # Conditions, not decoration: this machine's synthetic speeds and its
  # generation rate both move with background load (dasd, in particular), and a
  # wall clock without the conditions it was measured under is not comparable.
  echo "    host: load $(uptime | sed 's/.*load averages: //') | $(ps -Ao pcpu,comm | awk '/dasd/{printf "dasd %s%%", $1}')"
  # Never let the launcher find a server to "stop": that path races the
  # readiness poll. The port is free before every arm, or the arm does not
  # start.
  stop_server
  if [[ -n "$(listening_pids)" ]]; then
    echo "ERROR: port $PORT is still held by $(listening_pids); refusing to start" >&2; exit 1
  fi
  (
    cd "$ROOT"
    EMONE_PORT="$PORT" EMONE_MEMORY="$MEMORY" EMONE_MEMORY_TOOLS="$TOOLS" \
    EMONE_MEMORY_DIR="$MEMDIR" EMONE_MEMORY_JOURNAL=1 \
    EMONE_MEMORY_GUARD="${EMONE_MEMORY_GUARD:-1}" \
    EMONE_MEMORY_CONSOLIDATION=1 EMONE_MEMORY_CONSOLIDATION_IDLE_SECONDS="$IDLE" \
      exec "${LAUNCH[@]+"${LAUNCH[@]}"}"
  ) >"$SERVER_LOG" 2>&1 &
  LAUNCHER_PID=$!
  wait_ready
  grep -m1 "memory enabled" "$SERVER_LOG" || echo "(memory line: none, as expected for $ARM)"

  EMONE_PORT="$PORT" EMONE_MEMVAL_MEMDIR="$MEMDIR" EMONE_MEMVAL_RUN="$RUN" \
  EMONE_MEMVAL_SERVER_LOG="$SERVER_LOG" EMONE_MEMVAL_RESULTS="$LOGS" \
    python3 "$SCRIPT" "$ARM" 2>&1 | tee "$LOGS/run-$ARM-r$RUN.log"
  stop_server
  echo "--- consolidations for $ARM run $RUN:"
  grep -cE "memory (consolidated session=|consolidation skipped session=)" "$SERVER_LOG" || true
done
done

[[ "$BENCH" == smoke ]] || { echo; echo "=== report ($LABEL)"; EMONE_MEMVAL_RESULTS="$LOGS" python3 "$SCRIPT" report; }
