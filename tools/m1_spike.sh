#!/usr/bin/env bash
# Build this checkout, run the model-free tests, then measure which prefill
# levers matter on *this* Mac. Written for an M1 Max with 64 GB running
# Qwen3.8-Flash-Next 4-bit, but nothing in it is specific to that pair.
#
#   tools/m1_spike.sh                          # build, tests, 2 rounds of every arm
#   tools/m1_spike.sh --model <dir> --rounds 3
#   tools/m1_spike.sh --skip-build --skip-tests --arms "base s256 c2048"
#   tools/m1_spike.sh --dry-run                # print what would run, run nothing
#
# Why these arms. Every Qwen3.8 tuning verdict in docs/ was measured on a 24 GiB
# M3, where prefill ran at 90% GPU occupancy, so cutting expert reads could win
# at most ~10% and the read levers were closed. A 64 GB M1 Max has ~2.5x the GPU
# and ~4x the memory bandwidth but a similar SSD, and room for half the expert
# corpus in RAM. The arms test whether that moves the balance:
#
#   base         the install's profile defaults. On Qwen3.8 4-bit that is
#                now spike 10's winner: chunk 16384, the MPP wide and routed
#                switches, and the QSA kernels on the matrix units. The arms
#                below pin their own env and chunk; an arm that sets a switch
#                to 1 which the row already ships is the same run as base.
#
# Spike 4 (the default set, ~16K-token prompt): the chunk ceiling
#   c8192        --prefill-chunk 8192: half the chunks, so half the sweeps of
#                the routed-expert corpus (~57 GB each on Qwen3.8 4-bit)
#   c16384       --prefill-chunk 16384: the whole default prompt in one chunk
#   wide16k      c16384 + TINYTITAN_PREFILL_MPP_WIDE=1
#   pergqa       TINYTITAN_PREFILL_QSA_GQA=0: the per-head QSA kernel the grouped
#                one replaced, for a before/after on this build
#   c8192qsa     c8192 + TINYTITAN_QSA_GPU_SELECT=1. Decode only: the switch
#                never reaches the prefill selection, so for prefill this is a
#                second c8192 (spike 5 measured exactly that)
#   c8192wide    c8192 + TINYTITAN_PREFILL_MPP_WIDE=1
#   c8192sg      c8192 + TINYTITAN_PREFILL_SG_QMM=1: every batched prefill
#                projection and the shared expert on the simdgroup-matrix QMM
#                instead of MPP or the scalar QMM; may differ in output (sum
#                order), judged by tools/prefill_surprisal_ab.sh. Measured
#                slower than MPP on an M1 Max (spike 7)
#   c8192routed  c8192 + TINYTITAN_PREFILL_ROUTED_MPP=1: routed-expert tiles
#                as grouped MPP GEMMs instead of one output per thread
#   c8192widerouted  c8192wide + TINYTITAN_PREFILL_ROUTED_MPP=1
#   c8192fast    c8192widerouted + TINYTITAN_QSA_SCORE_MMA=1 (indexer scores
#                on the matrix units)
#   c16384fast   c8192fast at chunk 16384: one routed-expert sweep fewer on a
#                ~16K prompt, now that the tiles no longer hide the drive
#   c8192attn    c8192fast + TINYTITAN_PREFILL_QSA_MMA=1 (QSA attention on the
#                matrix units: rounding-level change, needs the surprisal A/B)
#   c16384attn   c8192attn at chunk 16384
# Chunking can change the output (the chunk boundaries move), so c8192/c16384
# may legitimately differ from base; they are judged on speed and on staying
# coherent, then on benchmark/quant_perplexity_ab.py before any default moves.
#
# Spike 3: the scalar prefill GEMMs onto MPP
#   mppwide      TINYTITAN_PREFILL_MPP_WIDE=1: hyper-connection, QSA-indexer and
#                PLE projections on the MPP tensor-op QMM, and the shared expert
#                as three GEMMs over the chunk; may change the output
#   gqa          TINYTITAN_PREFILL_QSA_GQA=1: the QSA attention kernel with one
#                threadgroup per (token, KV head), each selected K/V row read
#                once instead of once per query head; now the default, so this
#                arm is a second base
#   wide         mppwide + gqa
#   all          combo + mppwide
#   splitwide    split + mppwide + gqa (diagnostic, one round is enough)
# Run it with --gpu-clock: the clock moves more than most arms do.
#
# Spike 2: per-token dispatch and switches already in the tree
#   coalesce     TINYTITAN_PREFILL_COALESCE=1: one encoder per per-token loop;
#                must be bit-identical to base (output "same as base")
#   qqmm         TINYTITAN_PREFILL_Q_QMM=1: batched QMM for q-family projections;
#                may change the output
#   hcfused      TINYTITAN_HC_FUSED=1: fused hyper-connection gates
#   qsagpu       TINYTITAN_QSA_GPU_SELECT=1: QSA key selection on the GPU
#   combo        coalesce + hcfused + qsagpu
#   split        TINYTITAN_PREFILL_SPLIT=1: diagnostic; times each layer stage
#                (prefill_split_* roles). Its wall clock is not comparable.
#
# Spike 1 (settled on an M1 Max 64 GB, kept for other machines)
#   s128, s256   more routed-expert cache (256 slots ~ 34 GiB, half the corpus)
#   nobound      TINYTITAN_BOUNDED_IO=0: let the page cache hold experts too
#   s256nobound  both
#   c2048        half the prefill chunk: if this is much slower than base, the
#                cost tracks the chunk count and a >4096 chunk is worth building
#
# Arms are interleaved (the order rotates each round), greedy with a fixed seed,
# one model process at a time. Results land in benchmark/m1-spike/<stamp>/:
# every raw log, results.tsv, and summary.txt. Nothing here downloads, converts
# or re-installs a model, and nothing is purged or killed: a failed precondition
# stops the script with the reason.
set -Eeuo pipefail
# Under -e a failed command ends the script; say where, rather than vanishing.
trap 'echo "m1_spike: stopped at line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="$ROOT/models/qwen3.8-flash-next_125B_A6B_4Bit"
ROUNDS=2
PROMPT_CHARS=56000
MAX_NEW=64
COOLDOWN=20
ARMS="base c8192 c16384"
SKIP_BUILD=0
GPU_CLOCK=0
SKIP_TESTS=0
FULL_TESTS=0
DRY_RUN=0
MIN_FREE_PCT=20
OUT=""

usage() {
  sed -n '2,/^set -/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  cat <<'USAGE'
Options:
  --model <dir>        installed .gturbo model (default models/qwen3.8-flash-next_125B_A6B_4Bit)
  --rounds <n>         interleaved rounds per arm (default 2)
  --arms "<list>"      any of: base c8192 c16384 c8192qsa c8192wide c8192sg
                       c8192routed c8192widerouted c8192fast c16384fast
                       c8192attn c16384attn
                       wide16k pergqa
                       mppwide gqa wide splitwide all combo
                       coalesce qqmm hcfused qsagpu split
                       s128 s256 nobound s256nobound c2048
  --prompt-chars <n>   prompt size in characters (default 56000, ~16K tokens;
                       spikes 1-3 used 28000, ~7.9K)
  --max-new <n>        generated tokens per run (default 64)
  --cooldown <s>       pause between runs (default 20)
  --out <dir>          results directory (default benchmark/m1-spike/<stamp>)
  --skip-build         reuse the existing release build
  --skip-tests         skip the model-free tests
  --full-tests         run the whole suite instead of the suites this branch touched
  --dry-run            print the plan and the commands, run nothing
  --gpu-clock          sample the GPU clock and residency with powermetrics
                       during every run (asks for your password once) and
                       report GPU gigacycles: busy time x clock, the kernel
                       work independent of throttling
USAGE
}

die() {
  echo "m1_spike: $*" >&2
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --prompt-chars) PROMPT_CHARS="$2"; shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --cooldown) COOLDOWN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --full-tests) FULL_TESTS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --gpu-clock) GPU_CLOCK=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

# The script changes into the checkout before running anything, so a relative
# --model has to be anchored to where it was typed. No symlink resolution: the
# install receipt is bound to the exact path the model was installed at.
case "$MODEL" in
  /*) ;;
  *) MODEL="$PWD/$MODEL" ;;
esac
MODEL="${MODEL%/}"

case "$ROUNDS$PROMPT_CHARS$MAX_NEW$COOLDOWN" in
  *[!0-9]*) die "--rounds, --prompt-chars, --max-new and --cooldown take whole numbers" ;;
esac
[ "$ROUNDS" -ge 1 ] || die "--rounds must be at least 1"

# name|environment|extra CLI arguments
arm_spec() {
  case "$1" in
    base) echo "base||" ;;
    s128) echo "s128||--expert-cache-slots 128" ;;
    s256) echo "s256||--expert-cache-slots 256" ;;
    nobound) echo "nobound|TINYTITAN_BOUNDED_IO=0|" ;;
    s256nobound) echo "s256nobound|TINYTITAN_BOUNDED_IO=0|--expert-cache-slots 256" ;;
    c2048) echo "c2048||--prefill-chunk 2048" ;;
    c8192) echo "c8192||--prefill-chunk 8192" ;;
    c16384) echo "c16384||--prefill-chunk 16384" ;;
    wide16k) echo "wide16k|TINYTITAN_PREFILL_MPP_WIDE=1|--prefill-chunk 16384" ;;
    pergqa) echo "pergqa|TINYTITAN_PREFILL_QSA_GQA=0|" ;;
    c8192qsa) echo "c8192qsa|TINYTITAN_QSA_GPU_SELECT=1|--prefill-chunk 8192" ;;
    c8192wide) echo "c8192wide|TINYTITAN_PREFILL_MPP_WIDE=1|--prefill-chunk 8192" ;;
    c8192sg) echo "c8192sg|TINYTITAN_PREFILL_SG_QMM=1|--prefill-chunk 8192" ;;
    c8192routed) echo "c8192routed|TINYTITAN_PREFILL_ROUTED_MPP=1|--prefill-chunk 8192" ;;
    c8192widerouted) echo "c8192widerouted|TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_ROUTED_MPP=1|--prefill-chunk 8192" ;;
    c8192fast) echo "c8192fast|TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_ROUTED_MPP=1 TINYTITAN_QSA_SCORE_MMA=1|--prefill-chunk 8192" ;;
    c16384fast) echo "c16384fast|TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_ROUTED_MPP=1 TINYTITAN_QSA_SCORE_MMA=1|--prefill-chunk 16384" ;;
    c8192attn) echo "c8192attn|TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_ROUTED_MPP=1 TINYTITAN_QSA_SCORE_MMA=1 TINYTITAN_PREFILL_QSA_MMA=1|--prefill-chunk 8192" ;;
    c16384attn) echo "c16384attn|TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_ROUTED_MPP=1 TINYTITAN_QSA_SCORE_MMA=1 TINYTITAN_PREFILL_QSA_MMA=1|--prefill-chunk 16384" ;;
    coalesce) echo "coalesce|TINYTITAN_PREFILL_COALESCE=1|" ;;
    qqmm) echo "qqmm|TINYTITAN_PREFILL_Q_QMM=1|" ;;
    hcfused) echo "hcfused|TINYTITAN_HC_FUSED=1|" ;;
    qsagpu) echo "qsagpu|TINYTITAN_QSA_GPU_SELECT=1|" ;;
    combo) echo "combo|TINYTITAN_PREFILL_COALESCE=1 TINYTITAN_HC_FUSED=1 TINYTITAN_QSA_GPU_SELECT=1|" ;;
    split) echo "split|TINYTITAN_PREFILL_SPLIT=1|" ;;
    mppwide) echo "mppwide|TINYTITAN_PREFILL_MPP_WIDE=1|" ;;
    all) echo "all|TINYTITAN_PREFILL_COALESCE=1 TINYTITAN_HC_FUSED=1 TINYTITAN_QSA_GPU_SELECT=1 TINYTITAN_PREFILL_MPP_WIDE=1|" ;;
    gqa) echo "gqa|TINYTITAN_PREFILL_QSA_GQA=1|" ;;
    wide) echo "wide|TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_QSA_GQA=1|" ;;
    splitwide) echo "splitwide|TINYTITAN_PREFILL_SPLIT=1 TINYTITAN_PREFILL_MPP_WIDE=1 TINYTITAN_PREFILL_QSA_GQA=1|" ;;
    *) return 1 ;;
  esac
}

arm_list=()
for arm in $ARMS; do
  arm_spec "$arm" >/dev/null || die "unknown arm '$arm' (see --help for the list)"
  arm_list+=("$arm")
done
[ "${#arm_list[@]}" -gt 0 ] || die "--arms is empty"

run() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

# --- preconditions (AGENTS.md "Test rules") ---------------------------------
[ "$(uname -s)" = "Darwin" ] || die "needs macOS on Apple Silicon"
[ "$(uname -m)" = "arm64" ] || die "needs an arm64 (Apple Silicon) shell, not Rosetta"
macos_major="$(sw_vers -productVersion | cut -d. -f1)"
[ "$macos_major" -ge 26 ] || die "needs macOS 26 or newer (found $(sw_vers -productVersion))"
swift_version="$( (swift --version 2>&1 || true) | sed -n 's/.*Swift version \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
[ -n "$swift_version" ] || die "swift not found; install Xcode 27 (Swift 6.4+) and run xcode-select"
swift_major="${swift_version%%.*}"
swift_minor="${swift_version#*.}"
if [ "$swift_major" -lt 6 ] || { [ "$swift_major" -eq 6 ] && [ "$swift_minor" -lt 4 ]; }; then
  die "needs Swift 6.4+ (found $swift_version)"
fi

free_pct() {
  (memory_pressure -Q 2>/dev/null || true) | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'
}

check_quiet_machine() {
  local busy
  busy="$(pgrep -fl 'TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' || true)"
  [ -z "$busy" ] || die "another model or test process is running; stop it first:
$busy"
  local free
  free="$(free_pct)"
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_PCT" ]; then
    die "memory is under pressure (${free}% free, need ${MIN_FREE_PCT}%); close apps and retry"
  fi
}

cd "$ROOT"
stamp="$(date +%Y%m%d-%H%M%S)"
[ -n "$OUT" ] || OUT="$ROOT/benchmark/m1-spike/$stamp"
[ "$DRY_RUN" -eq 1 ] || mkdir -p "$OUT"

# --- machine record ---------------------------------------------------------
record_machine() {
  echo "commit      $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)') on $(git rev-parse --abbrev-ref HEAD)"
  echo "chip        $(sysctl -n machdep.cpu.brand_string)"
  echo "memory      $(($(sysctl -n hw.memsize) / 1073741824)) GiB"
  echo "gpu cores   $(system_profiler SPDisplaysDataType 2>/dev/null | sed -n 's/.*Total Number of Cores: *//p' | head -1)"
  echo "macOS       $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "swift       $swift_version"
  echo "model       $MODEL"
  # Experts stream from the model's own drive, so where it sits bounds every
  # result: an external Thunderbolt enclosure is a fraction of an internal SSD.
  # diskutil wants the volume, not a folder on it; df names the device.
  echo "model disk  $(diskutil info "$(df "$MODEL" 2>/dev/null | awk 'NR == 2 { print $1 }')" 2>/dev/null \
    | awk -F': *' '/Device Location|Protocol|Solid State/ { gsub(/^ +/, "", $1); printf "%s=%s  ", $1, $2 }')"
  echo "arms        ${arm_list[*]+"${arm_list[*]}"}"
  echo "rounds      $ROUNDS, prompt ${PROMPT_CHARS} chars, max-new $MAX_NEW, cooldown ${COOLDOWN}s"
}
if [ "$DRY_RUN" -eq 1 ]; then record_machine; else record_machine | tee "$OUT/machine.txt"; fi

# --- build -----------------------------------------------------------------
if [ "$SKIP_BUILD" -eq 0 ]; then
  echo
  echo "== build (release) =="
  run swift build -c release
fi
BIN_DIR="$(swift build -c release --show-bin-path)"
CLI="$BIN_DIR/TinyTitanCLI"
if [ "$DRY_RUN" -eq 0 ]; then
  [ -x "$CLI" ] || die "no release TinyTitanCLI at $CLI; run without --skip-build"
  archs="$(lipo -archs "$CLI")"
  [ "$archs" = "arm64" ] || die "release binary is '$archs', expected exactly arm64"
fi

# --- tests (never load a model) ---------------------------------------------
if [ "$SKIP_TESTS" -eq 0 ]; then
  echo
  check_quiet_machine
  if [ "$FULL_TESTS" -eq 1 ]; then
    echo "== tests: full suite, serial =="
    run swift test --no-parallel
  else
    echo "== tests: the suites this branch touched, serial =="
    run swift test --no-parallel --filter \
      'FrontierTracker|PrefillProgress|RawCompletionCapture|ServerPromptStateStore|ServerArgument|HTTPServer|PrefillAttentionQSAGrouped|PrefillSharedExpertBatched|GEMVRows|PrefillChunkScratch|PrefillRuntimeConfig|RuntimeConfiguration|ModelProfile|ModelIdentity|CLIArguments|OpenAIValidation|NgramTableReader|QSAPrefillSelection|ContinuationScore|PrefillAffineSimdgroupQMM|PrefillGroupedRoutedMoE|QSABlockScoresMMA'
  fi
fi

# --- model precondition -----------------------------------------------------
[ -d "$MODEL" ] || die "no model at $MODEL (pass --model <dir>; this script never installs one)"
[ -f "$MODEL/verified-install.json" ] \
  || die "$MODEL has no verified-install.json; it is not a completed install"

# --- the prompt: fixed repository prose, ASCII only --------------------------
prompt_file="$OUT/prompt.txt"
if [ "$DRY_RUN" -eq 0 ]; then
  # Filter to a whole file first, then cut it. Piping straight into `head -c`
  # lets head exit early, the writer die of SIGPIPE, and pipefail end the
  # script -- depending on timing, so it only fails some of the time.
  # shellcheck disable=SC2046 # the doc list is word-split on purpose
  cat $(ls "$ROOT"/docs/qwen38-*.md "$ROOT"/docs/adding-a-model.md | sort) \
    | LC_ALL=C tr -cd '\11\12\15\40-\176' >"$prompt_file.full"
  head -c "$PROMPT_CHARS" "$prompt_file.full" >"$prompt_file"
  rm -f "$prompt_file.full"
fi

results="$OUT/results.tsv"
[ "$DRY_RUN" -eq 1 ] \
  || printf 'round\tarm\tprefill_tok\tprefill_s\tprefill_tps\tdecode_tps\toccupancy_pct\tdecode_hit_pct\tdecode_gib\tmax_rss_gib\tswap_mb_delta\toutput_sha\texit\tgpu_mhz\tgpu_residency_pct\tgpu_busy_s\tgpu_gcycles\n' >"$results"

# --- GPU clock sampling (--gpu-clock) ----------------------------------------
# The GPU steps between 972 and 1,296 MHz under sustained load, which moved an
# unchanged arm 12% between two runs. Busy time x clock is the work itself.
SUDO_KEEPALIVE=""
stop_keepalive() {
  if [ -n "$SUDO_KEEPALIVE" ]; then kill "$SUDO_KEEPALIVE" 2>/dev/null || true; fi
}
if [ "$GPU_CLOCK" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "powermetrics needs root; sudo asks once, then stays warm for the run."
  sudo -v || die "--gpu-clock needs sudo for powermetrics"
  (while true; do sudo -n true 2>/dev/null || true; sleep 50; done) &
  SUDO_KEEPALIVE=$!
  trap stop_keepalive EXIT
fi

# Mean clock and residency over the samples where the GPU was mostly busy,
# i.e. the prefill, not the model load or the cooldown around it.
gpu_clock_stats() {
  awk '/GPU HW active frequency/ { f = $5 }
       /GPU HW active residency/ { r = $5; sub("%", "", r)
                                   if (r + 0 >= 50) { sf += f; sr += r; n++ } }
       END { if (n) printf "%.0f %.1f", sf / n, sr / n }' "$1" 2>/dev/null || true
}

swap_used_mb() {
  (sysctl -n vm.swapusage || true) | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p'
}

run_arm() {
  local round="$1" arm="$2" spec env_part args_part log out code
  spec="$(arm_spec "$arm")"
  env_part="$(echo "$spec" | cut -d'|' -f2)"
  args_part="$(echo "$spec" | cut -d'|' -f3)"
  log="$OUT/r${round}-${arm}.log"
  out="$OUT/r${round}-${arm}.out"

  # TURBO_FIELDFARE_PHASES adds the per-chunk host-time breakdown (stderr).
  local envs=(TINYTITAN_KERNEL_STATS=1 TINYTITAN_RUNNER_STATS=1 TURBO_FIELDFARE_PHASES=1)
  local assignment
  for assignment in $env_part; do envs+=("$assignment"); done
  local extra=()
  # shellcheck disable=SC2206 # arm arguments are simple words
  if [ -n "$args_part" ]; then extra=($args_part); fi

  echo
  echo "== round $round, arm $arm =="
  echo "+ env ${envs[*]+"${envs[*]}"} TinyTitanCLI --model $MODEL --prompt <prompt.txt> --max-new $MAX_NEW --temperature 0 --seed 1 --max-context 65536 ${extra[*]+"${extra[*]}"}"
  [ "$DRY_RUN" -eq 0 ] || return 0

  check_quiet_machine
  local swap_before swap_after
  swap_before="$(swap_used_mb)"
  # A failed run is recorded, not fatal; `|| code=$?` keeps both -e and the
  # ERR trap out of it.
  local gpufile="$OUT/r${round}-${arm}.gpu.txt" pm_pid=""
  if [ "$GPU_CLOCK" -eq 1 ]; then
    sudo -n powermetrics --samplers gpu_power -i 2000 -o "$gpufile" >/dev/null 2>&1 &
    pm_pid=$!
  fi
  code=0
  /usr/bin/time -l env ${envs[@]+"${envs[@]}"} "$CLI" \
    --model "$MODEL" --prompt "$(cat "$prompt_file")" \
    --max-new "$MAX_NEW" --temperature 0 --seed 1 --max-context 65536 \
    ${extra[@]+"${extra[@]}"} >"$out" 2>"$log" || code=$?
  swap_after="$(swap_used_mb)"
  if [ -n "$pm_pid" ]; then
    sudo -n pkill -INT -f "powermetrics --samplers gpu_power -i 2000 -o $gpufile" || true
    wait "$pm_pid" 2>/dev/null || true
  fi

  # The loader says "trusted install receipt invalid: model directory
  # mismatch"; match the part that names the cause, not the exact wording.
  if grep -q 'model directory mismatch' "$log"; then
    die "the model's install receipt names a different path (the model was moved).
$(grep -m1 '^error:' "$log" || true)
Either move it back to the path above, or re-issue the receipt in place:
  swift run -c release TinyTitanRepack --verify-install --input-gturbo \"$MODEL\""
  fi

  local footer prefill_tok prefill_s decode_tps prefill_tps occ hit gib rss sha swap_delta
  footer="$(grep -o '\[stop=[^]]*\]' "$log" | tail -1 || true)"
  prefill_tok="$(echo "$footer" | sed -n 's/.*prefill=\([0-9]*\)tok.*/\1/p')"
  prefill_s="$(echo "$footer" | sed -n 's/.*prefill=[0-9]*tok\/\([0-9.]*\)s.*/\1/p')"
  decode_tps="$(echo "$footer" | sed -n 's/.*tok\/s=\([0-9.]*\).*/\1/p')"
  # A rate over a handful of tokens is noise (one token over ~1 ms printed
  # 993 tok/s), so decode is only reported past 16 generated tokens.
  local new_tok
  new_tok="$(echo "$footer" | sed -n 's/.* new=\([0-9]*\)tok.*/\1/p')"
  if [ "${new_tok:-0}" -lt 16 ]; then decode_tps="n/a(${new_tok:-0}tok)"; fi
  prefill_tps="$(awk -v t="${prefill_tok:-0}" -v s="${prefill_s:-0}" 'BEGIN { if (s > 0) printf "%.1f", t / s; else print "" }')"
  occ="$(sed -n 's/.*(\([0-9]*\)% occupied).*/\1/p' "$log" | tail -1)"
  hit="$(sed -n 's/.*\[decode expert io\].*(\([0-9.]*\)% hit).*/\1/p' "$log" | tail -1)"
  gib="$(sed -n 's/.*\[decode expert io\].*hit) \([0-9.]*\) GiB.*/\1/p' "$log" | tail -1)"
  rss="$(awk '/maximum resident set size/ { printf "%.1f", $1 / 1073741824 }' "$log")"
  sha="$(shasum -a 256 "$out" | cut -c1-12)"
  swap_delta="$(awk -v a="${swap_before:-0}" -v b="${swap_after:-0}" 'BEGIN { printf "%+.0f", b - a }')"
  local busy_ms gpu_mhz="" gpu_res="" busy_s gcycles=""
  busy_ms="$(sed -n 's/.*busy \([0-9]*\) ms of.*/\1/p' "$log" | tail -1)"
  busy_s="$(awk -v b="${busy_ms:-0}" 'BEGIN { if (b > 0) printf "%.1f", b / 1000 }')"
  if [ -n "$pm_pid" ]; then
    read -r gpu_mhz gpu_res <<<"$(gpu_clock_stats "$gpufile")" || true
    if [ -n "$gpu_mhz" ] && [ -n "$busy_ms" ]; then
      gcycles="$(awk -v b="$busy_ms" -v f="$gpu_mhz" 'BEGIN { printf "%.1f", b * f / 1000000 }')"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$round" "$arm" "${prefill_tok:--}" "${prefill_s:--}" "${prefill_tps:--}" \
    "${decode_tps:--}" "${occ:--}" "${hit:--}" "${gib:--}" "${rss:--}" \
    "$swap_delta" "$sha" "$code" \
    "${gpu_mhz:--}" "${gpu_res:--}" "${busy_s:--}" "${gcycles:--}" >>"$results"
  echo "   exit $code  ${footer:-no footer (see $log)}  occupancy ${occ:-?}%  rss ${rss:-?} GiB  swap ${swap_delta} MB"
  if [ -n "$pm_pid" ]; then
    echo "   gpu ${gpu_mhz:-?} MHz at ${gpu_res:-?}% residency, busy ${busy_s:-?} s = ${gcycles:-?} Gcycles"
  fi
  if [ "$code" -ne 0 ]; then
    echo "   arm failed: $(grep -m1 '^error:' "$log" || echo "no error line; see $log")" >&2
    # The first run failing means the model or the binary is the problem, and
    # every later run would fail the same way.
    [ "$RUNS_DONE" -gt 0 ] || die "the first run failed, so the matrix stops here. Log: $log"
  fi
  RUNS_DONE=$((RUNS_DONE + 1))
}

# --- the matrix: arms interleaved, the order rotating each round -------------
RUNS_DONE=0
count="${#arm_list[@]}"
round=1
while [ "$round" -le "$ROUNDS" ]; do
  i=0
  while [ "$i" -lt "$count" ]; do
    arm="${arm_list[$(((i + round - 1) % count))]}"
    run_arm "$round" "$arm"
    if [ "$DRY_RUN" -eq 0 ] && [ "$COOLDOWN" -gt 0 ]; then sleep "$COOLDOWN"; fi
    i=$((i + 1))
  done
  round=$((round + 1))
done

[ "$DRY_RUN" -eq 0 ] || exit 0

# --- the golden check, informational ----------------------------------------
golden="not checked (model is not under models/)"
if [ "$MODEL" = "$ROOT/models/qwen3.8-flash-next_125B_A6B_4Bit" ]; then
  check_quiet_machine
  if tools/golden-baseline.sh --check qwen38-4 >"$OUT/golden.log" 2>&1; then
    golden="matches benchmark/golden"
  else
    golden="differs (see golden.log). The stored baseline is from another machine, so a
             difference is expected until one is captured here; it is not by itself a bug"
  fi
fi

# --- summary ----------------------------------------------------------------
{
  cat "$OUT/machine.txt"
  echo "golden      $golden"
  echo
  awk -F'\t' '
    NR == 1 { next }
    {
      arm = $2
      if (!(arm in seen)) { seen[arm] = 1; order[++n] = arm }
      ps[arm] = ps[arm] (ps[arm] == "" ? "" : " ") $4
      pt[arm] = pt[arm] (pt[arm] == "" ? "" : " ") $5
      dt[arm] = dt[arm] (dt[arm] == "" ? "" : " ") $6
      oc[arm] = oc[arm] (oc[arm] == "" ? "" : " ") $7
      hr[arm] = hr[arm] (hr[arm] == "" ? "" : " ") $8
      rs[arm] = rs[arm] (rs[arm] == "" ? "" : " ") $10
      sw[arm] = sw[arm] (sw[arm] == "" ? "" : " ") $11
      sh[arm] = sh[arm] (sh[arm] == "" ? "" : " ") $12
      if ($13 != 0) failed[arm]++
      if ($4 + 0 > 0) { sum[arm] += $4; k[arm]++ }
      mz[arm] = mz[arm] (mz[arm] == "" ? "" : " ") $14
      gc[arm] = gc[arm] (gc[arm] == "" ? "" : " ") $17
      if ($17 + 0 > 0) { gsum[arm] += $17; gk[arm]++; anyclock = 1 }
    }
    END {
      printf "%-12s %-18s %-14s %-13s %-9s %-11s %-10s %-9s %s\n", \
        "arm", "prefill s", "prefill tok/s", "decode tok/s", "occ %", "dec hit %", "rss GiB", "swap MB", "output"
      for (i = 1; i <= n; i++) {
        a = order[i]
        split(sh[a], hs, " "); split(sh["base"], bs, " ")
        same = (a == "base") ? "reference" : ((hs[1] == bs[1]) ? "same as base" : "DIFFERS from base")
        if (failed[a]) same = same ", " failed[a] " failed"
        printf "%-12s %-18s %-14s %-13s %-9s %-11s %-10s %-9s %s\n", \
          a, ps[a], pt[a], dt[a], oc[a], hr[a], rs[a], sw[a], same
      }
      print ""
      print "Values are per round, in round order; read the spread before the mean."
      if (k["base"] > 0) {
        b = sum["base"] / k["base"]
        for (i = 1; i <= n; i++) {
          a = order[i]
          if (a == "base" || k[a] == 0) continue
          printf "  %-12s prefill %+.1f%% vs base\n", a, 100 * (sum[a] / k[a] - b) / b
        }
      }
      if (anyclock) {
        print ""
        print "GPU work, independent of the clock (--gpu-clock): busy s x MHz = Gcycles"
        printf "%-12s %-18s %s\n", "arm", "GPU MHz", "Gcycles"
        for (i = 1; i <= n; i++) {
          a = order[i]
          printf "%-12s %-18s %s\n", a, mz[a], gc[a]
        }
        if (gk["base"] > 0) {
          gb = gsum["base"] / gk["base"]
          for (i = 1; i <= n; i++) {
            a = order[i]
            if (a == "base" || gk[a] == 0) continue
            printf "  %-12s GPU work %+.1f%% vs base\n", a, 100 * (gsum[a] / gk[a] - gb) / gb
          }
        }
        print "Compare kernels by Gcycles: prefill seconds also move with the clock."
      }
      print ""
      print "How to read it:"
      print "  occ % on base well under 90 -> prefill is waiting on the SSD here, unlike the M3;"
      print "     s256 and a >4096 chunk are the levers to build on."
      print "  c2048 much slower than base -> prefill cost tracks the chunk count."
      print "  s256 faster with swap +0 -> run with --expert-cache-slots 256 (or --ram-budget) now."
      print "  nobound faster -> the page-cache trade pays on this machine (TINYTITAN_BOUNDED_IO=0)."
      print "  c2048 may legitimately differ in output (different chunking); the others should not."
      print "Spike 2:"
      print "  coalesce MUST read \"same as base\"; if it does not, that is a bug, not a trade."
      print "  qqmm/hcfused/qsagpu may differ; a faster arm that differs needs a quality check"
      print "     (benchmark/quant_perplexity_ab.py) before it becomes a default."
      print "  split is a diagnostic: read its prefill_split_* roles in r*-split.log, not its time."
      print "Spike 3:"
      print "  mppwide/all may differ from base (different summation order); faster AND"
      print "     close is the bar, then benchmark/quant_perplexity_ab.py before a default."
      print "  gqa MUST read \"same as base\": it keeps the per-head arithmetic exactly."
      print "  Read GPU work (Gcycles) first; prefill seconds move with the clock."
      print "Spike 4:"
      print "  c8192/c16384 read the routed experts 2x/4x fewer times than base on a ~16K"
      print "     prompt; the win is in prefill seconds and the \"expert fetch + tiles\""
      print "     phase (r*-*.log), not in Gcycles. They may differ in output (chunking)."
      print "  pergqa MUST read \"same as base\" and should be slower: the old kernel."
    }' "$results"
  # The host-side split of each prefill (TURBO_FIELDFARE_PHASES, on in every
  # arm): the chunk count, then the dense GPU half and the routed-expert half.
  # A bigger chunk should shrink the expert half, which Gcycles cannot show.
  echo
  echo "Prefill phases per round (s): chunks / route readback + GPU / expert fetch + tiles"
  for arm in ${arm_list[@]+"${arm_list[@]}"}; do
    line=""
    r=1
    while [ "$r" -le "$ROUNDS" ]; do
      log="$OUT/r$r-$arm.log"
      if [ -f "$log" ]; then
        line="$line  $(awk '
          /^\[prefill phases over/ { c++ }
          /route readback \+ GPU:/ { g += $5 }
          /expert fetch \+ tiles:/ { e += $5 }
          END { printf "%d / %.1f / %.1f", c, g / 1000, e / 1000 }' "$log")"
      fi
      r=$((r + 1))
    done
    printf "  %-12s%s\n" "$arm" "$line"
  done
} | tee "$OUT/summary.txt"

echo
echo "results: $OUT"
