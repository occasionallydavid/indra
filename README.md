# Indra OCR Rust

Inference-only Rust port of Indra OCR. At build time,
[`burn-onnx`](https://github.com/tracel-ai/burn-onnx) converts all ten exported
ONNX graphs into native [Burn](https://burn.dev) code and Burnpack weights. The
release binary embeds those weights and no longer needs ONNX Runtime, `ort`, or
the model directory at runtime. PNG and JPEG are enabled by default; CJK, math,
WebP, TIFF, BMP, and GIF are optional Cargo features.

The port includes line detection, CTC recognition, orientation and deskew,
reading order, layout blocks, and Markdown tables. CJK routing and LaTeX math
recognition are optional features. It contains no training code or
model-authoring paths.

PDF rendering and PDF text-layer extraction are not included in this port.

## Quick start

```shell
./scripts/download-models.sh models
cargo build --release
target/release/indra-ocr scan.png
```

The ONNX files are only needed to regenerate the native models during the build.
The downloader uses plain `curl` requests to versioned model URLs and verifies
each file with SHA-256. It does not use `hf-hub`.

## CLI

```shell
target/release/indra-ocr scan.png
target/release/indra-ocr scan.jpg --format text
target/release/indra-ocr page.png --format json --timings
cargo build --release --features math
target/release/indra-ocr page.png --math
target/release/indra-ocr page.png --table-heads
target/release/indra-ocr *.png --out results/
cargo build --release --features webp,tiff
cargo build --release --features cjk
```

Output formats are Markdown, plain text, and JSON.

The default PNG/JPEG build embeds only the base detection, recognition, order,
and orientation models and is about 18 MiB. Enable `cjk` for the 9.8 MiB Chinese
recognizer or `math` for the roughly 21 MiB math models.

See [RUST.md](RUST.md) for library usage and implementation details.

## Browser / WebAssembly

The WASM target does not include the `image` crate. JavaScript decodes the file
with Canvas and passes an RGBA byte buffer to Rust:

```shell
./scripts/download-models.sh models
./scripts/build-wasm.sh
python3 -m http.server 8000 -d web
```

Open `http://localhost:8000`. CJK or math can be added to the WASM build with:

```shell
./scripts/build-wasm.sh cjk,math
```

The wrapper is [web/index.html](web/index.html), and the generated module is
written to `web/pkg/`.

## License

The code is Apache-2.0. The model weights have their own license; see the
upstream model repository for details.
