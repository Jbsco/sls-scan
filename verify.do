#!/bin/bash
# The increment gate: every mechanical check, cheapest first.
exec >&2
fail=0
run () {
  name="$1"; shift
  echo "=========================================="
  echo "verify: $name"
  echo "=========================================="
  if "$@"; then
    echo "verify: $name OK"
  else
    echo "verify: $name FAILED"
    fail=1
  fi
}
run "style + sanitization" bash style.do
run "tests"                bash test.do
run "reference"            bash ref.do
run "application"          bash app.do
echo "=========================================="
if [ "$fail" -ne 0 ]; then
  echo "verify: one or more gates failed"
  exit 1
fi
echo "verify: all gates green"
echo "=========================================="
[ -n "${3:-}" ] && touch "$3"
exit 0
