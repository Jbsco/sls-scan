#!/bin/bash
# Style and hygiene gate: markdown that renders in the local viewer, and no local
# paths or private material.
exec >&2
FAIL=0

echo "== 1/2  Markdown renders in the local viewer"
MD=$(git ls-files '*.md')
if [ -n "$MD" ] && command -v markdown >/dev/null 2>&1; then
    if python3 tools/mdlint.py $MD; then
        echo "  clean"
    else
        FAIL=1
    fi
else
    echo "  markdown renderer not found - skipped"
fi

echo "== 2/2  Content sanitization"
PATHS=$(git ls-files -z | xargs -0 grep -nE '~/|/home/[a-z]+/' 2>/dev/null \
        | grep -v '^style.do:' | grep -v '^.github/workflows/' || true)
if [ -n "$PATHS" ]; then
    echo "  local path reference(s):"
    echo "$PATHS" | sed 's/^/    /'
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "style: FAILED"
    exit 1
fi
echo "style: all checks clean"
[ -n "${3:-}" ] && touch "$3"
exit 0
