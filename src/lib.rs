//! Inference-only Rust port of Indra OCR.
//!
//! This crate runs the existing exported ONNX models. It deliberately contains
//! no model-authoring, loss, target-encoding or training code.

pub mod image_ops;
pub mod layout;
pub mod math;
pub mod model;
pub mod order;
pub mod pipeline;
pub mod preprocess;
pub mod vocab;

pub use layout::Block;
pub use pipeline::{InputError, Line, Ocr, Page, Timing};
pub use vocab::{Token, Vocab};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
