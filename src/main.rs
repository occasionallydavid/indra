use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indra_ocr::pipeline::{Ocr, default_model_dir, load_image};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Format {
    Md,
    Text,
    Json,
}

#[derive(Debug, Parser)]
#[command(
    name = "indra-ocr",
    version,
    about = "OCR images with Indra's existing ONNX models"
)]
struct Args {
    /// Image files. All formats supported by the image crate are accepted.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Write one output file per input instead of printing to stdout.
    #[arg(long)]
    out: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = Format::Md)]
    format: Format,

    /// Folder containing det.onnx, rec.onnx and meta.json.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// ONNX Runtime intra-op threads.
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Recognition lines per batch.
    #[arg(long, default_value_t = 64)]
    batch: usize,

    /// Do not run the page-orientation model.
    #[arg(long)]
    no_orient: bool,

    /// Read lines classified as math as inline LaTeX.
    #[arg(long)]
    math: bool,

    /// Build tables from the order model's table heads.
    #[arg(long)]
    table_heads: bool,

    /// Print stage timings to stderr.
    #[arg(long)]
    timings: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(failed) => {
            if failed > 0 {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            eprintln!("indra-ocr: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<usize> {
    let args = Args::parse();
    let model_dir = args.model_dir.unwrap_or_else(default_model_dir);
    let mut ocr = Ocr::from_dir(&model_dir, args.threads)?
        .with_batch(args.batch)
        .with_table_heads(args.table_heads);
    if args.math {
        ocr.enable_math()?;
    }
    if let Some(out) = &args.out {
        fs::create_dir_all(out)
            .with_context(|| format!("could not create output directory {}", out.display()))?;
    }

    let extension = match args.format {
        Format::Md => ".md",
        Format::Text => ".txt",
        Format::Json => ".jsonl",
    };
    let mut failed = 0;
    let mut names = Vec::<String>::new();
    for input in &args.inputs {
        let image = match load_image(input) {
            Ok(image) => image,
            Err(error) => {
                eprintln!("indra-ocr: {error}");
                failed += 1;
                continue;
            }
        };
        let page = match ocr.ocr_image(&image, !args.no_orient) {
            Ok(page) => page,
            Err(error) => {
                eprintln!("indra-ocr: {}: {error:#}", input.display());
                failed += 1;
                continue;
            }
        };
        if args.timings {
            eprintln!(
                "{}: {}, {:.0} ms {}",
                input.display(),
                page.source,
                page.timings.get("total_ms").copied().unwrap_or(0.0),
                page.note
            );
        }
        let body = match args.format {
            Format::Md => page.markdown.clone().unwrap_or_else(|| page.text()),
            Format::Text => page.text(),
            Format::Json => serde_json::to_string(&serde_json::json!({
                "file": input,
                "page": 0,
                "source": page.source,
                "rotation": page.rotation,
                "skew": page.skew,
                "note": page.note,
                "lines": page.lines,
                "markdown": page.markdown,
                "blocks": page.blocks,
                "timings": page.timings,
            }))?,
        };
        if let Some(out_dir) = &args.out {
            let mut name = input
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("output")
                .to_string();
            let base = name.clone();
            let mut suffix = 1;
            while names.contains(&name) {
                suffix += 1;
                name = format!("{base}_{suffix}");
            }
            names.push(name.clone());
            let output = out_dir.join(format!("{name}{extension}"));
            fs::write(&output, body + "\n")
                .with_context(|| format!("could not write {}", output.display()))?;
        } else {
            println!("{body}");
        }
    }
    if failed > 0 {
        eprintln!("indra-ocr: {failed} of {} inputs failed", args.inputs.len());
    }
    Ok(failed)
}

#[allow(dead_code)]
fn _is_file(path: &Path) -> bool {
    path.is_file()
}
