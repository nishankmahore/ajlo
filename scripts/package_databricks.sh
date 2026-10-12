#!/usr/bin/env bash
# Build a clean zip of the code needed on Databricks (no data, results, venv, git).
# Output: dist/ajlo_databricks.zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/dist"
STAGE="$(mktemp -d)/ajlo"
mkdir -p "$STAGE" "$OUT"

cd "$ROOT"
for p in ajlo experiments tests databricks pyproject.toml README.md LICENSE; do
  cp -R "$p" "$STAGE/"
done
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$STAGE" -name "*.pyc" -delete
find "$STAGE" -name ".DS_Store" -delete

rm -f "$OUT/ajlo_databricks.zip"
(cd "$(dirname "$STAGE")" && zip -qr "$OUT/ajlo_databricks.zip" ajlo)
rm -rf "$(dirname "$STAGE")"

echo "Package: $OUT/ajlo_databricks.zip"
unzip -l "$OUT/ajlo_databricks.zip" | tail -1
