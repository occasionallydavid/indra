# Indra OCR Rust

Inference-only Rust port of Indra OCR. It runs the existing exported ONNX
models through [`ort`](https://crates.io/crates/ort) and reads image files with
the `image` crate.

The port includes line detection, CTC recognition, CJK routing, orientation and
deskew, reading order, layout blocks, Markdown tables, and optional LaTeX math
recognition. It contains no training code or model-authoring paths.

PDF rendering and PDF text-layer extraction are not included in this port.

## Quick start

```shell
./scripts/download-models.sh models
cargo build --release
target/release/indra-ocr scan.png --model-dir models
```

The downloader uses plain `curl` requests to versioned model URLs and verifies
each file with SHA-256. It does not use `hf-hub`.

## CLI

```shell
target/release/indra-ocr scan.png --model-dir models
target/release/indra-ocr scan.jpg --model-dir models --format text
target/release/indra-ocr page.png --model-dir models --format json --timings
target/release/indra-ocr page.png --model-dir models --math
target/release/indra-ocr page.png --model-dir models --table-heads
target/release/indra-ocr *.png --model-dir models --out results/
```

`--model-dir` can be omitted when `INDRA_OCR_MODEL_DIR` is set or the model
files are in `./models`. Output formats are Markdown, plain text, and JSON.

See [RUST.md](RUST.md) for library usage and implementation details.

## License

The code is Apache-2.0. The model weights have their own license; see the
upstream model repository for details.
