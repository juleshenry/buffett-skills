#!/usr/bin/env bash
# Batch-runs `python -m src.run TICKER` across a list of tickers, tolerating
# local Ollama flakiness (this has been observed to OOM-crash under load on
# 16GB machines, especially with other Ollama usage happening concurrently).
#
# Strategy: before each attempt, poll until Ollama responds. After each
# attempt, check the per-ticker log for connection failures; if any occurred,
# retry the same ticker (cheap: successful LLM calls are disk-cached, so a
# retry only re-does the calls that actually failed).
#
# Usage:
#   scripts/run_universe.sh <ticker_list_file> <log_dir>
set -uo pipefail

TICKER_LIST="${1:?usage: run_universe.sh <ticker_list_file> <log_dir>}"
LOG_DIR="${2:?usage: run_universe.sh <ticker_list_file> <log_dir>}"

OLLAMA_URL="http://localhost:11434/api/tags"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
HEALTH_POLL_INTERVAL="${HEALTH_POLL_INTERVAL:-5}"
HEALTH_MAX_WAIT="${HEALTH_MAX_WAIT:-180}"
INTER_TICKER_SLEEP="${INTER_TICKER_SLEEP:-30}"
RETRY_BACKOFF="${RETRY_BACKOFF:-15}"
# Free pages * 16384 bytes/page; ~65536 pages =~ 1GB. This machine has been
# observed to OOM-crash the shared Ollama server when free pages get very
# low, so pause and let memory recover before piling on more requests rather
# than just checking "is the HTTP port up" (which can be true right up until
# the crash).
MIN_FREE_PAGES="${MIN_FREE_PAGES:-65536}"
MEMORY_POLL_INTERVAL="${MEMORY_POLL_INTERVAL:-10}"
MEMORY_MAX_WAIT="${MEMORY_MAX_WAIT:-120}"

mkdir -p "$LOG_DIR"

wait_for_ollama() {
  local waited=0
  while true; do
    if curl -s -m 3 -o /dev/null -w '%{http_code}' "$OLLAMA_URL" 2>/dev/null | grep -q '^200$'; then
      return 0
    fi
    if [ "$waited" -ge "$HEALTH_MAX_WAIT" ]; then
      return 1
    fi
    sleep "$HEALTH_POLL_INTERVAL"
    waited=$((waited + HEALTH_POLL_INTERVAL))
  done
}

free_pages_now() {
  memory_pressure 2>/dev/null | awk -F': ' '/Pages free/ {gsub(/ /,"",$2); print $2}'
}

wait_for_memory() {
  local waited=0
  local free
  while true; do
    free="$(free_pages_now)"
    if [ -z "$free" ] || [ "$free" -ge "$MIN_FREE_PAGES" ] 2>/dev/null; then
      return 0
    fi
    if [ "$waited" -ge "$MEMORY_MAX_WAIT" ]; then
      echo "  (memory still tight after ${MEMORY_MAX_WAIT}s wait: ${free} free pages; proceeding cautiously)"
      return 0
    fi
    echo "  (memory tight: ${free} free pages < ${MIN_FREE_PAGES}, waiting ${MEMORY_POLL_INTERVAL}s...)"
    sleep "$MEMORY_POLL_INTERVAL"
    waited=$((waited + MEMORY_POLL_INTERVAL))
  done
}

clean_count=0
recovered_count=0
failed_tickers=()
recovered_tickers=()

while IFS= read -r ticker; do
  ticker="$(echo "$ticker" | tr -d '[:space:]')"
  [ -z "$ticker" ] && continue

  attempt=1
  ok=0
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    if ! wait_for_ollama; then
      echo "[$ticker] WARNING: ollama unresponsive after ${HEALTH_MAX_WAIT}s wait, trying anyway"
    fi
    wait_for_memory

    log_file="$LOG_DIR/${ticker}_attempt${attempt}.log"
    echo "=== [$ticker] attempt $attempt/$MAX_ATTEMPTS ==="
    python3 -m src.run "$ticker" > "$log_file" 2>&1
    exit_code=$?
    fail_count=$(grep -c "Connection refused\|RemoteDisconnected\|ConnectionResetError" "$log_file" 2>/dev/null || true)
    fail_count="${fail_count:-0}"

    if [ "$exit_code" -eq 0 ] && [ "$fail_count" -eq 0 ]; then
      ok=1
      break
    fi

    echo "[$ticker] attempt $attempt: exit=$exit_code ollama_connection_failures=$fail_count"
    attempt=$((attempt + 1))
    [ "$attempt" -le "$MAX_ATTEMPTS" ] && sleep "$RETRY_BACKOFF"
  done

  if [ "$ok" -eq 1 ]; then
    if [ "$attempt" -gt 1 ]; then
      echo "[$ticker] OK (recovered on attempt $attempt)"
      recovered_tickers+=("$ticker")
      recovered_count=$((recovered_count + 1))
    else
      echo "[$ticker] OK (clean)"
      clean_count=$((clean_count + 1))
    fi
  else
    echo "[$ticker] FAILED after $MAX_ATTEMPTS attempts -- some LLM-backed metrics may be missing/degraded"
    failed_tickers+=("$ticker")
  fi

  sleep "$INTER_TICKER_SLEEP"
done < "$TICKER_LIST"

echo
echo "==================== SUMMARY ===================="
echo "Clean:                  $clean_count"
echo "Recovered after retry:  $recovered_count  -> ${recovered_tickers[*]:-none}"
echo "Failed (max attempts):  ${#failed_tickers[@]}  -> ${failed_tickers[*]:-none}"
