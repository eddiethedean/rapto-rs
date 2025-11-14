#!/usr/bin/env bash
set -euo pipefail

# Generate flamegraph from perf data
# Usage: generate_flamegraph.sh <perf_data_file> [output_svg]

PERF_DATA="${1:-}"
OUTPUT_SVG="${2:-${PERF_DATA%.*}.flamegraph.svg}"

if [ -z "$PERF_DATA" ] || [ ! -f "$PERF_DATA" ]; then
    echo "Usage: generate_flamegraph.sh <perf_data_file> [output_svg]"
    exit 1
fi

echo "Generating flamegraph from $PERF_DATA..."

# Use perf script to extract data
perf script -i "$PERF_DATA" | \
    /workspace/.venv/bin/flamegraph.pl \
    --title "$(basename "$OUTPUT_SVG")" \
    --width 1600 \
    > "$OUTPUT_SVG" 2>/dev/null || {
    echo "Error: Failed to generate flamegraph"
    echo "Attempting to install flamegraph if missing..."
    pip3 install --break-system-packages flamegraph
    perf script -i "$PERF_DATA" | \
        /workspace/.venv/bin/flamegraph.pl > "$OUTPUT_SVG" || {
        echo "Error: flamegraph generation failed"
        exit 1
    }
}

echo "Flamegraph generated: $OUTPUT_SVG"

