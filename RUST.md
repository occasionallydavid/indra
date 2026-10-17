# Rust inference port

This repository now includes an inference-only Rust implementation of Indra OCR.
It runs the existing exported ONNX weights through [`ort`](https://crates.io/crates/ort)
and contains no training code, targets, losses, or model-authoring paths.

The Rust CLI accepts image input. The `image` crate supplies PNG, JPEG, WebP,
TIFF, BMP, GIF, PNG, PNM, QOI, and other common image formats. PDF rendering and
the PDF text-layer path are intentionally not included.

## Build and download models

The ONNX weights are not committed. Download the versioned inference files
manually with the included script:

```shell
./scripts/download-models.sh models
cargo build --release
```

The script uses plain `curl` requests to the versioned model repository, checks
every file's SHA-256, and does not use `hf-hub`.

## CLI

```shell
target/release/indra-ocr scan.png --model-dir models
target/release/indra-ocr scan.jpg --model-dir models --format text
target/release/indra-ocr page.png --model-dir models --format json --timings
target/release/indra-ocr page.png --model-dir models --math
target/release/indra-ocr page.png --model-dir models --table-heads
target/release/indra-ocr *.png --model-dir models --out results/
```

`--model-dir` can be omitted when `INDRA_OCR_MODEL_DIR` is set or the weights
are in `./models`.

Output formats are `md`, `text`, and `json`. JSON includes line boxes,
confidences, layout blocks, Markdown, page rotation, skew, and stage timings.

## Library

```rust
use indra_ocr::pipeline::{Ocr, load_image};

fn main() -> anyhow::Result<()> {
    let image = load_image("page.png").map_err(anyhow::Error::msg)?;
    let mut ocr = Ocr::from_dir("models", 4)?;
    let page = ocr.ocr_image(&image, true)?;
    println!("{}", page.markdown.unwrap_or_else(|| page.text()));
    Ok(())
}
```

`Ocr::enable_math()` loads the optional math classifier and attention decoder.
CJK routing and page orientation are enabled automatically when `script.onnx`,
`rec_cjk.onnx`, and `orient.onnx` are present.

## Inference files

The downloaded folder contains:

- `det.onnx`, `rec.onnx`, `order.onnx`, `meta.json`
- `orient.onnx`, `script.onnx`, `rec_cjk.onnx`
- `mathcls.onnx`, `math_enc.onnx`, `math_dec_init.onnx`,
  `math_dec_step.onnx`, `math_meta.json`

The model version and filenames are pinned in `scripts/download-models.sh`.
