#!/bin/bash
# 9ul.1: EMONE_EARLY_HITS off/on, interleaved pairs, greedy, fixed seed.
set -u
S="$1"; ROOT=/Volumes/Chai/TinyTitan; CLI=$ROOT/.build/out/Products/Release/EmoneCLI
PROMPT="$(cat "$S/prompt.txt")"
run() { # row model max_new extra pair arm
  local row=$1 model=$2 n=$3 extra=$4 pair=$5 arm=$6 tag="$S/ab/${1}_p${5}_${6}"
  local env=(EMONE_RUNNER_STATS=1)
  [ "$arm" = on ] && env+=(EMONE_EARLY_HITS=1)
  env "${env[@]}" "$CLI" --model "$ROOT/models/$model" --prompt "$PROMPT" --max-new "$n" \
    --temperature 0 --seed 1 --ignore-eos $extra >"$tag.out" 2>"$tag.err"
  echo "$row pair=$pair arm=$arm exit=$? $(grep -o 'tok/s=[0-9.]*' "$tag.out" | tail -1) text_md5=$(grep -v "^\[stop=" "$tag.out" | md5 -q)"
}
mkdir -p "$S/ab"
for pair in 1 2 3; do
  run q36 qwen3.6_35B_A3B_4Bit 512 "" $pair off; run q36 qwen3.6_35B_A3B_4Bit 512 "" $pair on
done
for pair in 1 2 3; do
  run ornith ornith-1.5_35B_A3B_4Bit 512 "" $pair off; run ornith ornith-1.5_35B_A3B_4Bit 512 "" $pair on
done
for pair in 1 2 3; do
  run q38 qwen38flash.gturbo 256 "--expert-cache-slots 128" $pair off; run q38 qwen38flash.gturbo 256 "--expert-cache-slots 128" $pair on
done
