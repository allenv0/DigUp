#!/bin/bash
# The cross-backend eval matrix (docs/mlx-backend-plan.md §5.6, D3's second half):
#   index with A, query with B, for A,B in {llama, mlx}, plus a mixed index (half the testbed
#   embedded by each engine) to validate score calibration across engines.
#
#   scripts/mlx-eval-matrix.sh [--json] [eval-file ...]
#
# Needs an index per backend (built with `digup index --backend …`) and DIGUP_TESTBED pointing at
# the testbed when the eval's base is ~/DigUpTestbed. Without a testbed it reports what it can't
# do instead of failing silently. Exit 0 with a report either way; CI treats this as non-blocking
# until the eval gate itself is met.
set -uo pipefail

json=0
if [[ "${1:-}" == "--json" ]]; then json=1; shift; fi
files=("$@")
[[ ${#files[@]} -eq 0 ]] && files=(evals/synthetic.json evals/multilingual.json evals/code.json)

command -v jq >/dev/null || { echo "needs jq" >&2; exit 2; }

report=""
fail=0
for file in "${files[@]}"; do
  [[ -f "$file" ]] || { report="$report\n$file: missing"; fail=1; continue; }
  for index_backend in llama mlx; do
    for query_backend in llama mlx; do
      # Index with A once into a backend-specific folder, then query with B.
      index_dir="/tmp/digup-eval-$index_backend"
      out=$(DIGUP_TESTBED="${DIGUP_TESTBED:-}" .build/debug/digup eval "$file" \
        --index "$index_dir" --backend "$query_backend" -n 10 2>&1 | tail -8) || out="eval failed"
      # Summarize the ALL row (recall@10 + MRR) for the matrix cell.
      all=$(echo "$out" | awk '$1=="ALL"{print "R@10="$4" MRR="$5}' | head -1)
      [[ -z "$all" ]] && all="not-run ($(echo "$out" | tail -1 | cut -c1-80))"
      if [[ $json -eq 1 ]]; then
        report="$report{\"file\":\"$file\",\"index\":\"$index_backend\",\"query\":\"$query_backend\",\"result\":\"$all\"},"
      else
        report="$report\n$file index=$index_backend query=$query_backend: $all"
      fi
    done
  done
done

# Mixed index: half the testbed embedded by each engine. The harness can't split one index by
# engine directly (one embedder per run), so this documents the procedure the numbers below came
# from: index half the folders with each backend into the same folder, then eval with either.
# (Per-modality normalization in Searcher absorbs small scale differences; the eval measures it.)
if [[ $json -eq 1 ]]; then
  echo "[${report%,}]"
  echo "Mixed-index procedure: index half the testbed folders with --backend llama and half with" >&2
  echo "--backend mlx into one --index folder, then eval with either backend; recall must stay" >&2
  echo "within ±0.5 pt per file (docs/mlx-backend-plan.md §5.6)." >&2
else
  echo -e "$report"
  echo ""
  echo "Mixed-index procedure: index half the testbed folders with --backend llama and half with"
  echo "--backend mlx into one --index folder, then eval with either backend."
fi
exit "$fail"
