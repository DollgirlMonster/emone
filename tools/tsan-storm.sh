#!/usr/bin/env bash
# Reproduce the ThreadSanitizer report TT-001 is closed against, on demand.
#
# The `thread-sanitizer` CI job is a single `swift test --sanitize=thread`, and a
# single instrumented run is almost always clean: the report needs the drainer
# task to be resumed on one thread while its frame is written on another, and it
# is contention that makes that land. Six instrumented processes at once
# reproduce it within a round or two, though the rate varies with how loaded the
# machine already is (measured between 0 and 7 reports per 12 runs on one Mac).
#
# That is the point of this script: it runs the same sanitized server bundle
# many times in parallel and fails if **any** run reports. With the suppression
# in `tools/tsan-suppressions.txt` it stays clean (and TSan prints "Matched 1
# suppressions" on the runs that would otherwise have fired); pass
# `--no-suppressions` to see the raw report, which is how the suppression is
# re-justified when the toolchain or swift-nio moves.
#
#   tools/tsan-storm.sh                       # 2 rounds of 6, with suppressions
#   tools/tsan-storm.sh --no-suppressions     # the raw report, for re-checking
#   tools/tsan-storm.sh --runs 5 --parallel 4
#   tools/tsan-storm.sh --filter ResponsesAPIHTTPTests
#
# Exits 0 when no process reported, 1 when at least one did, 2 on a setup error.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE="$ROOT/.build/out/Products/Debug/EmoneServerTests.xctest/Contents/MacOS/EmoneServerTests"
TSAN_LIB="$(dirname "$BUNDLE")/../Frameworks/libclang_rt.tsan_osx_dynamic.dylib"
SDK_ROOT="$(xcode-select -p)"
HELPER="$SDK_ROOT/Toolchains/XcodeDefault.xctoolchain/usr/libexec/swift/pm/swiftpm-testing-helper"
FRAMEWORKS="$SDK_ROOT/Platforms/MacOSX.platform/Developer/Library/Frameworks"
XCODE_LIBS="$SDK_ROOT/Platforms/MacOSX.platform/Developer/usr/lib"
SUPPRESSIONS="$ROOT/tools/tsan-suppressions.txt"

runs=2
parallel=6
filter=""
use_suppressions=1

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --runs) runs="${2:?--runs needs a number}"; shift 2 ;;
    --parallel) parallel="${2:?--parallel needs a number}"; shift 2 ;;
    --filter) filter="${2:?--filter needs a pattern}"; shift 2 ;;
    --suppressions) SUPPRESSIONS="${2:?--suppressions needs a path}"; shift 2 ;;
    --no-suppressions) use_suppressions=0; shift ;;
    --help|-h) sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

[ -x "$HELPER" ] || die "no swiftpm-testing-helper under $SDK_ROOT; is Xcode installed?"
[ -d "$FRAMEWORKS/XCTest.framework" ] || die "no XCTest.framework under $FRAMEWORKS"
[ -f "$TSAN_LIB" ] || die "no ThreadSanitizer runtime at $TSAN_LIB"

if [ ! -f "$BUNDLE" ]; then
  say "Building the instrumented test bundle (one time)"
  (cd "$ROOT" && swift build --build-tests --sanitize=thread) || die "the sanitized build failed"
  [ -f "$TSAN_LIB" ] || die "the sanitized build produced no ThreadSanitizer runtime"
fi

tsan_options="halt_on_error=0"
if [ "$use_suppressions" = 1 ]; then
  if [ -f "$SUPPRESSIONS" ]; then
    tsan_options="$tsan_options:suppressions=$SUPPRESSIONS"
  else
    die "no suppression file at $SUPPRESSIONS (pass --no-suppressions to run raw)"
  fi
fi

extra=()
[ -n "$filter" ] && extra=(--filter "$filter")

work="$(mktemp -d "${TMPDIR:-/tmp}/tt-tsan-storm.XXXXXX")"
say "Running $((runs * parallel)) instrumented runs, $parallel at a time"
[ "$use_suppressions" = 1 ] && echo "  suppressions: $SUPPRESSIONS" || echo "  suppressions: OFF (raw report expected)"

round=1
while [ "$round" -le "$runs" ]; do
  pids=()
  slot=1
  while [ "$slot" -le "$parallel" ]; do
    env DYLD_INSERT_LIBRARIES="$TSAN_LIB" \
        DYLD_FRAMEWORK_PATH="$FRAMEWORKS" \
        DYLD_LIBRARY_PATH="$XCODE_LIBS" \
        TSAN_OPTIONS="$tsan_options" \
        "$HELPER" --test-bundle-path "$BUNDLE" --no-parallel "$BUNDLE" \
        --testing-library swift-testing "${extra[@]+"${extra[@]}"}" \
        > "$work/run-$round-$slot.log" 2>&1 &
    pids+=($!)
    slot=$((slot + 1))
  done
  wait "${pids[@]+"${pids[@]}"}"
  reported="$(grep -l 'WARNING: ThreadSanitizer' "$work"/run-"$round"-*.log 2>/dev/null | wc -l | tr -d ' ')"
  echo "  round $round: $reported of $parallel reported"
  round=$((round + 1))
done

hits="$(grep -l 'WARNING: ThreadSanitizer' "$work"/*.log 2>/dev/null)"
if [ -n "$hits" ]; then
  say "REPRODUCED"
  for log in $hits; do
    echo "  ${log##*/}:"
    grep -A3 'WARNING: ThreadSanitizer' "$log" | sed 's/^/    /'
    grep 'SUMMARY: ThreadSanitizer' "$log" | head -1 | sed 's/^/    /'
  done
  echo "  logs kept in $work"
  exit 1
fi

say "Clean: no report in $((runs * parallel)) instrumented runs"
rm -rf "$work"
exit 0
