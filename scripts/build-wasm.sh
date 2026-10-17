#!/usr/bin/env bash
set -euo pipefail

extra_features="${1:-}"
features="wasm"
if [[ -n "$extra_features" ]]; then
  features="wasm,$extra_features"
fi

wasm-pack build \
  --target web \
  --release \
  --out-dir web/pkg \
  --out-name indra_ocr \
  --no-default-features \
  --features "$features"
