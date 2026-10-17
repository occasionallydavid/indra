# Rust inference port

This repository includes an inference-only Rust implementation of Indra OCR.
[`burn-onnx`](https://github.com/tracel-ai/burn-onnx) converts all ten exported
ONNX graphs into native Burn modules during the build. The resulting binary
contains no ONNX parser or ONNX Runtime and has no training, targets, losses, or
model-authoring paths.

The Rust CLI accepts image input. PNG and JPEG are enabled by default. CJK, WebP,
TIFF, BMP, and GIF can be enabled with Cargo features. PDF rendering and the PDF
text-layer path are intentionally not included.

## Build and download models

The ONNX weights are not committed. Download the versioned inference files
manually with the included script, then build:

```shell
./scripts/download-models.sh models
cargo build --release
```

The script uses plain `curl` requests to the versioned model repository, checks
every file's SHA-256, and does not use `hf-hub`. The ONNX files are compile-time
inputs: the build writes generated Rust and Burnpack weight modules to Cargo's
`OUT_DIR`, and those weights are embedded in the final executable.

## CLI

```shell
target/release/indra-ocr scan.png
target/release/indra-ocr scan.jpg --format text
target/release/indra-ocr page.png --format json --timings
cargo build --release --features math
target/release/indra-ocr page.png --math
target/release/indra-ocr page.png --table-heads
target/release/indra-ocr *.png --out results/
cargo build --release --features cjk
```

Output formats are `md`, `text`, and `json`. JSON includes line boxes,
confidences, layout blocks, Markdown, page rotation, skew, and stage timings.

## Library

```rust
use indra_ocr::pipeline::{Ocr, load_image};

fn main() -> anyhow::Result<()> {
    let image = load_image("page.png").map_err(anyhow::Error::msg)?;
    let mut ocr = Ocr::new()?;
    let page = ocr.ocr_image(&image, true)?;
    println!("{}", page.markdown.unwrap_or_else(|| page.text()));
    Ok(())
}
```

`Ocr::enable_math()` loads the classifier and attention decoder when the crate
is built with the `math` feature. This avoids embedding roughly 21 MiB of math
weights in binaries that do not use them. CJK routing is likewise opt-in with
the `cjk` feature, avoiding its 9.8 MiB recognizer in the default build. Page
orientation is always included.

## Native model generation

`build.rs` registers the ten ONNX graphs with `ModelGen`, uses Burn's embedding
loader for the `.bpk` weights, and includes the generated files through
`src/model/mod.rs`. A `Clip` operator override handles one upstream Burn ONNX
codegen type mismatch in `math_enc`.

The compile-time input folder contains:

- `det.onnx`, `rec.onnx`, `order.onnx`, `meta.json`
- `orient.onnx`, plus `script.onnx` and `rec_cjk.onnx` when building with `cjk`
- `mathcls.onnx`, `math_enc.onnx`, `math_dec_init.onnx`,
  `math_dec_step.onnx`, `math_meta.json`, only when building with `math`

The model version and filenames are pinned in `scripts/download-models.sh`.
