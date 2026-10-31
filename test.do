#!/bin/bash
# Test gate: the pytest suite of the pure functions.
exec >&2
if ! python3 -c 'import pytest' 2>/dev/null; then
    echo "test: pytest not found - install the dev extras (pip install -e '.[dev]')"
    exit 1
fi
python3 -m pytest -q tests || exit 1
[ -n "${3:-}" ] && touch "$3"
exit 0
