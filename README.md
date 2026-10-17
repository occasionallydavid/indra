# Indra OCR Rust

Inference-only Rust port of Indra OCR. At build time,
[`burn-onnx`](https://github.com/tracel-ai/burn-onnx) converts all ten exported
ONNX graphs into native [Burn](https://burn.dev) code and Burnpack weights. The
release binary embeds those weights and no longer needs ONNX Runtime, `ort`, or
the model directory at runtime.

The port includes line detection, CTC recognition, CJK routing, orientation and
deskew, reading order, layout blocks, Markdown tables, and optional LaTeX math
recognition. It contains no training code or model-authoring paths.

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
target/release/indra-ocr page.png --math
target/release/indra-ocr page.png --table-heads
target/release/indra-ocr *.png --out results/
```

Output formats are Markdown, plain text, and JSON.

See [RUST.md](RUST.md) for library usage and implementation details.

## License

The code is Apache-2.0. The model weights have their own license; see the
upstream model repository for details.
