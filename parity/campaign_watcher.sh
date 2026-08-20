#!/usr/bin/env bash
# Campaign sentinel: exits non-zero on failure (dead workers, logs stalled
# 15 min, error burst); exits 0 when the campaign completes (all .done markers).
# Watches ARTIFACTS (.done files, log growth), not process lifetimes.
set -u
OUT="${PARITY_OUT:-./parity-results}"
EXPECTED_DONE="${EXPECTED_DONE:-9}"
STALL_LIMIT=3
prev_size=-1; stalls=0; t0=$(date +%s)
while :; do
  done_count=$(ls "$OUT"/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -ge "$EXPECTED_DONE" ]; then
    echo "CAMPAIGN COMPLETE: $done_count/$EXPECTED_DONE runs"; exit 0
  fi
  if [ $(( $(date +%s) - t0 )) -gt 32400 ]; then
    echo "TIMEOUT 9h with $done_count/$EXPECTED_DONE"; exit 2
  fi
  procs=$(pgrep -fc "[l]m_eval --model" || true)
  if [ "${procs:-0}" -lt 1 ]; then
    echo "FAILURE: zero live lm_eval workers with $done_count/$EXPECTED_DONE done"
    ps -eo pid,etime,args | grep "[l]m_eval --model" | head -4; exit 1
  fi
  for f in "$OUT"/*.log; do
    [ -f "$f" ] || continue
    errs=$(grep -cE "HTTP Error 4|HTTP Error 5|Traceback|ConnectionError" "$f")
    if [ "$errs" -gt 40 ]; then
      echo "FAILURE: $(basename $f) accumulated $errs errors"; tail -5 "$f"; exit 1
    fi
  done
  size=$(cat "$OUT"/*.log 2>/dev/null | wc -c)
  if [ "$size" = "$prev_size" ]; then
    stalls=$((stalls+1))
    if [ "$stalls" -ge "$STALL_LIMIT" ]; then
      echo "FAILURE: logs stopped growing for 15 min ($done_count/$EXPECTED_DONE done)"
      for f in "$OUT"/*.log; do printf "%s: %s\n" "$(basename $f)" "$(grep -oE 'Requesting API: +[0-9]+%' $f | tail -1)"; done
      exit 1
    fi
  else
    stalls=0
  fi
  prev_size=$size
  sleep 300
done
