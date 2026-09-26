#!/usr/bin/env bash
# Does a prefill switch change what the model understands? A paired surprisal
# A/B for prefill numerics, such as TINYTITAN_PREFILL_MPP_WIDE.
#
#   tools/prefill_surprisal_ab.sh --model <dir>
#   tools/prefill_surprisal_ab.sh --model <dir> --b-env "TINYTITAN_PREFILL_MPP_WIDE=1" --score 512
#
# The same text runs twice through `TinyTitanCLI --score`: all but the last
# N tokens are prefilled (the path the switch changes), then those N are
# teacher-forced through decode (which it does not change) and each one's
# negative log-likelihood is recorded. Both arms read identical tokens, so the
# per-token difference B - A is a paired number: its mean, standard error and
# t say whether the switch moved surprisal, and by how much in perplexity.
# A mean within ~2 standard errors of zero is "no measurable change".
#
# Greedy decoding is not involved, so one pass per arm is the measurement;
# there is no sampling noise to average away. Nothing here downloads or
# installs a model, and nothing is killed: a failed precondition stops it.
set -Eeuo pipefail
trap 'echo "prefill_surprisal_ab: stopped at line $LINENO: $BASH_COMMAND (exit $?)" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL=""
TEXT_CHARS=60000
TEXT_FILE=""
SCORE=512
CHUNK=8192
A_CHUNK=""
B_CHUNK=""
A_ENV=""
B_ENV="TINYTITAN_PREFILL_MPP_WIDE=1"
OUT=""

usage() {
  sed -n '2,/^set -/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  cat <<'USAGE'
Options:
  --model <dir>        installed .gturbo model (required)
  --text-chars <n>     characters of repository prose to use (default 60000,
                       ~17K tokens; the last --score tokens are scored)
  --text <file>        score this text instead (its first --text-chars
                       characters). The default prose is this repository's
                       docs, which change between commits, so pass an earlier
                       run's text.txt to compare runs across commits
  --score <n>          continuation tokens to score (default 512; decode speed
                       sets the time: ~2 min at 4 tok/s)
  --chunk <n>          --prefill-chunk for both arms (default 8192)
  --a-chunk <n>        --prefill-chunk for arm A only (default --chunk)
  --b-chunk <n>        --prefill-chunk for arm B only (default --chunk), to
                       judge a whole launch configuration against a reference
  --a-env "<K=V ...>"  environment for arm A (default: none, the baseline)
  --b-env "<K=V ...>"  environment for arm B (default TINYTITAN_PREFILL_MPP_WIDE=1)
  --out <dir>          results directory (default benchmark/surprisal/<stamp>)
USAGE
}

die() {
  echo "prefill_surprisal_ab: $*" >&2
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --text-chars) TEXT_CHARS="$2"; shift 2 ;;
    --text) TEXT_FILE="$2"; shift 2 ;;
    --score) SCORE="$2"; shift 2 ;;
    --chunk) CHUNK="$2"; shift 2 ;;
    --a-chunk) A_CHUNK="$2"; shift 2 ;;
    --b-chunk) B_CHUNK="$2"; shift 2 ;;
    --a-env) A_ENV="$2"; shift 2 ;;
    --b-env) B_ENV="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

[ -n "$MODEL" ] || die "--model is required"
case "$MODEL" in
  /*) ;;
  *) MODEL="$PWD/$MODEL" ;;
esac
MODEL="${MODEL%/}"
case "$TEXT_FILE" in
  "" | /*) ;;
  *) TEXT_FILE="$PWD/$TEXT_FILE" ;;
esac
A_CHUNK="${A_CHUNK:-$CHUNK}"
B_CHUNK="${B_CHUNK:-$CHUNK}"
case "$TEXT_CHARS$SCORE$A_CHUNK$B_CHUNK" in
  *[!0-9]*) die "--text-chars, --score and the chunks take whole numbers" ;;
esac

[ "$(uname -s)" = "Darwin" ] || die "needs macOS on Apple Silicon"
[ -f "$MODEL/verified-install.json" ] || die "$MODEL is not a completed install"
busy="$(pgrep -fl 'TinyTitanServer|TinyTitanCLI|TinyTitanPackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm' || true)"
[ -z "$busy" ] || die "another model or test process is running; stop it first:
$busy"

cd "$ROOT"
CLI="$(swift build -c release --show-bin-path)/TinyTitanCLI"
[ -x "$CLI" ] || die "no release TinyTitanCLI at $CLI; run swift build -c release first"

[ -n "$OUT" ] || OUT="$ROOT/benchmark/surprisal/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"

# The same prose m1_spike.sh uses, filtered to a file first and then cut (see
# its note on SIGPIPE under pipefail).
text="$OUT/text.txt"
if [ -n "$TEXT_FILE" ]; then
  [ -f "$TEXT_FILE" ] || die "--text $TEXT_FILE is not a file"
  LC_ALL=C tr -cd '\11\12\15\40-\176' <"$TEXT_FILE" >"$text.full"
else
  # shellcheck disable=SC2046 # the doc list is word-split on purpose
  cat $(ls "$ROOT"/docs/qwen38-*.md "$ROOT"/docs/adding-a-model.md "$ROOT"/docs/m1-prefill-spike.md | sort) \
    | LC_ALL=C tr -cd '\11\12\15\40-\176' >"$text.full"
fi
head -c "$TEXT_CHARS" "$text.full" >"$text"
rm -f "$text.full"

run_arm() {
  local name="$1" envs="$2" chunk="$3" code=0
  local -a pairs=()
  local pair
  for pair in $envs; do pairs+=("$pair"); done
  echo "== arm $name: ${envs:-(no environment)}, chunk $chunk =="
  env ${pairs[@]+"${pairs[@]}"} "$CLI" --model "$MODEL" --prompt "$(cat "$text")" \
    --score "$SCORE" --score-out "$OUT/$name.nll" \
    --prefill-chunk "$chunk" --max-context 65536 >"$OUT/$name.log" 2>&1 || code=$?
  [ "$code" -eq 0 ] || die "arm $name exited $code; see $OUT/$name.log"
  grep '^\[score' "$OUT/$name.log" || die "arm $name printed no score line; see $OUT/$name.log"
}

run_arm A "$A_ENV" "$A_CHUNK"
sleep 20
run_arm B "$B_ENV" "$B_CHUNK"

hash_of() { sed -n 's/.*token_hash=\([0-9a-f]*\).*/\1/p' "$OUT/$1.log"; }
[ "$(hash_of A)" = "$(hash_of B)" ] || die "the arms scored different tokens; not a comparison"

{
  echo
  echo "model   $MODEL"
  echo "commit  $(git rev-parse --short HEAD)"
  echo "A env   ${A_ENV:-(none)}, chunk $A_CHUNK"
  echo "B env   ${B_ENV:-(none)}, chunk $B_CHUNK"
  paste "$OUT/A.nll" "$OUT/B.nll" | awk '
    { a += $1; b += $2; d = $2 - $1; s += d; ss += d * d; n++ }
    END {
      if (n < 2) { print "too few tokens scored"; exit 1 }
      mean = s / n
      var = (ss - n * mean * mean) / (n - 1)
      if (var < 0) var = 0
      se = sqrt(var / n)
      t = (se > 0) ? mean / se : 0
      printf "tokens  %d\n", n
      printf "A       mean nll %.5f  perplexity %.4f\n", a / n, exp(a / n)
      printf "B       mean nll %.5f  perplexity %.4f\n", b / n, exp(b / n)
      printf "B - A   mean dNLL %+.5f  (se %.5f, t %+.2f)  perplexity %+.3f%%\n", \
        mean, se, t, 100 * (exp(b / n) / exp(a / n) - 1)
      if (t > -2 && t < 2) print "verdict no measurable change (|t| < 2)"
      else if (mean > 0) print "verdict B is MORE surprised: a real loss; size it before accepting"
      else print "verdict B is less surprised"
    }'
} | tee "$OUT/summary.txt"
echo "results: $OUT"
