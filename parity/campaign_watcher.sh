#!/usr/bin/env bash
# Centinela de la campaña AA-Index: sale ≠0 ante fallo (proceso muerto, logs
# estancados 15 min, ráfaga de errores); sale 0 al completarse (18 .done).
set -u
OUT=/home/ludwid/lm-eval/aa-index
EXPECTED_DONE=24
STALL_LIMIT=3
prev_size=-1; stalls=0; t0=$(date +%s)
while :; do
  done_count=$(ls "$OUT"/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -ge "$EXPECTED_DONE" ]; then
    echo "CAMPAÑA COMPLETA: $done_count/$EXPECTED_DONE runs"; exit 0
  fi
  if [ $(( $(date +%s) - t0 )) -gt 32400 ]; then
    echo "TIMEOUT 9h con $done_count/$EXPECTED_DONE"; exit 2
  fi
  procs=$(pgrep -fc "[l]m_eval --model" || true)
  if [ "${procs:-0}" -lt 1 ]; then
    echo "FALLO: cero workers lm_eval vivos con $done_count/$EXPECTED_DONE done"
    ps -eo pid,etime,args | grep "[l]m_eval --model" | head -4; exit 1
  fi
  for f in "$OUT"/*.log; do
    [ -f "$f" ] || continue
    errs=$(grep -cE "HTTP Error 4|HTTP Error 5|Traceback|ConnectionError" "$f")
    if [ "$errs" -gt 40 ]; then
      echo "FALLO: $(basename $f) acumula $errs errores"; tail -5 "$f"; exit 1
    fi
  done
  size=$(cat "$OUT"/*.log 2>/dev/null | wc -c)
  if [ "$size" = "$prev_size" ]; then
    stalls=$((stalls+1))
    if [ "$stalls" -ge "$STALL_LIMIT" ]; then
      echo "FALLO: logs sin crecer 15 min ($done_count/$EXPECTED_DONE done)"
      for f in "$OUT"/*.log; do printf "%s: %s\n" "$(basename $f)" "$(grep -oE 'Requesting API: +[0-9]+%' $f | tail -1)"; done
      exit 1
    fi
  else
    stalls=0
  fi
  prev_size=$size
  sleep 300
done
