#!/bin/bash
set -e

# Code quality checks for the RAG chatbot project
# Usage: ./scripts/quality.sh [command]
#   format  - Auto-format code with black
#   check   - Check formatting without making changes
#   test    - Run the test suite
#   all     - Run all checks (default)

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

format() {
    echo "=== Formatting code with black ==="
    uv run black backend/ main.py
    echo ""
}

check_format() {
    echo "=== Checking code formatting ==="
    uv run black --check backend/ main.py
    echo ""
}

run_tests() {
    echo "=== Running tests ==="
    cd backend && uv run pytest tests/ -v
    cd "$PROJECT_ROOT"
    echo ""
}

run_all() {
    check_format
    run_tests
    echo "=== All checks passed ==="
}

case "${1:-all}" in
    format)
        format
        ;;
    check)
        check_format
        ;;
    test)
        run_tests
        ;;
    all)
        run_all
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 {format|check|test|all}"
        exit 1
        ;;
esac
