use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use image::GrayImage;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde::Deserialize;

use crate::order::Box4;
use crate::preprocess::{LineCrop, WIDTH_MULTIPLE, crop_line, pad_batch, width_batches};

const MATH_MIN: f32 = 0.5;
pub const MATH_MIN_CONF: f32 = 0.9;
const EARLY_STOP_MARGIN: f32 = 0.1;
const BOS: i64 = 1;
const EOS: i64 = 2;
const N_SPECIAL: i64 = 4;

const FILES: [&str; 5] = [
    "mathcls.onnx",
    "math_enc.onnx",
    "math_dec_init.onnx",
    "math_dec_step.onnx",
    "math_meta.json",
];

#[derive(Debug, Deserialize)]
struct MathMeta {
    config: MathConfig,
    tokens: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MathConfig {
    height: usize,
    max_tokens: usize,
    dec_layers: usize,
    dec_heads: usize,
    channels: usize,
}

pub struct MathReader {
    classifier: Session,
    encoder: Session,
    decoder_init: Session,
    decoder_step: Session,
    height: usize,
    max_tokens: usize,
    layers: usize,
    heads: usize,
    head_dim: usize,
    channels: usize,
    items: Vec<String>,
}

impl MathReader {
    pub fn load(
        model_dir: &Path,
        environment: &Environment,
        threads: usize,
    ) -> Result<Option<Self>> {
        if !FILES.iter().all(|file| model_dir.join(file).is_file()) {
            return Ok(None);
        }
        let meta: MathMeta = serde_json::from_str(
            &fs::read_to_string(model_dir.join("math_meta.json"))
                .context("could not read math_meta.json")?,
        )
        .context("invalid math_meta.json")?;
        let load = |name: &str| load_math_session(environment, &model_dir.join(name), threads);
        let classifier = load("mathcls.onnx")?;
        let encoder = load("math_enc.onnx")?;
        let decoder_init = load("math_dec_init.onnx")?;
        let decoder_step = load("math_dec_step.onnx")?;
        let mut items = vec![String::new(), String::new(), String::new(), String::new()];
        items.extend(meta.tokens);
        Ok(Some(Self {
            classifier,
            encoder,
            decoder_init,
            decoder_step,
            height: meta.config.height,
            max_tokens: meta.config.max_tokens,
            layers: meta.config.dec_layers,
            heads: meta.config.dec_heads,
            head_dim: meta.config.channels / meta.config.dec_heads,
            channels: meta.config.channels,
            items,
        }))
    }

    pub fn flags(&mut self, crops: &[LineCrop], batch: usize) -> Result<Vec<bool>> {
        let mut out = vec![false; crops.len()];
        let widths: Vec<usize> = crops.iter().map(|crop| crop.width).collect();
        for part in width_batches(&widths, batch, 1.3, 64, 64 * 640) {
            let lines: Vec<LineCrop> = part.iter().map(|&index| crops[index].clone()).collect();
            let batch = pad_batch(&lines, WIDTH_MULTIPLE);
            let x = Tensor::from_array((
                [batch.batch, 1, batch.height, batch.width],
                batch.data.into_boxed_slice(),
            ))?;
            let widths = Tensor::from_array(([batch.batch], batch.widths.into_boxed_slice()))?;
            let outputs = self
                .classifier
                .run(ort::inputs!["x" => x, "widths" => widths])?;
            let (_, probabilities) = outputs["p"].try_extract_tensor::<f32>()?;
            for (local, &index) in part.iter().enumerate() {
                out[index] = probabilities[local] > MATH_MIN;
            }
        }
        Ok(out)
    }

    pub fn read(
        &mut self,
        gray: &GrayImage,
        boxes: &[Box4],
        batch: usize,
    ) -> Result<Vec<(String, f32)>> {
        let crops: Vec<LineCrop> = boxes
            .iter()
            .map(|&bbox| crop_line(gray, bbox.as_array(), self.height, 4096))
            .collect();
        let mut out = vec![(String::new(), 0.0); crops.len()];
        let widths: Vec<usize> = crops.iter().map(|crop| crop.width).collect();
        for part in width_batches(&widths, batch, 4.0, 64, 16 * 64 * 640) {
            let lines: Vec<LineCrop> = part.iter().map(|&index| crops[index].clone()).collect();
            let batch = pad_batch(&lines, WIDTH_MULTIPLE);
            let x = Tensor::from_array((
                [batch.batch, 1, batch.height, batch.width],
                batch.data.into_boxed_slice(),
            ))?;
            let widths = Tensor::from_array(([batch.batch], batch.widths.into_boxed_slice()))?;
            let encoded = self
                .encoder
                .run(ort::inputs!["x" => x, "widths" => widths])?;
            let (_, memory) = encoded["memory"].try_extract_tensor::<f32>()?;
            let (_, mask) = encoded["mask"].try_extract_tensor::<f32>()?;
            let memory = Tensor::from_array((
                [
                    batch.batch,
                    memory.len() / (batch.batch * self.channels),
                    self.channels,
                ],
                memory.to_vec().into_boxed_slice(),
            ))?;
            let initialized = self.decoder_init.run(ort::inputs!["memory" => memory])?;
            let (cross_shape, cross_k) = initialized["cross_k"].try_extract_tensor::<f32>()?;
            let (_, cross_v) = initialized["cross_v"].try_extract_tensor::<f32>()?;
            let cross_shape: Vec<usize> = cross_shape.iter().map(|&d| d as usize).collect();
            let mut self_k = Vec::<f32>::new();
            let mut self_v = Vec::<f32>::new();
            let mut token = vec![BOS; batch.batch];
            let mut sequences = vec![Vec::<usize>::new(); batch.batch];
            let mut probabilities = vec![Vec::<f32>::new(); batch.batch];
            let mut done = vec![false; batch.batch];
            for position in 0..self.max_tokens.saturating_sub(1) {
                let self_width = if position == 0 { 0 } else { position };
                let token_tensor =
                    Tensor::from_array(([batch.batch], token.clone().into_boxed_slice()))?;
                let position_tensor = Tensor::from_array(((), vec![position as i64]))?;
                let self_k_tensor = Tensor::from_array((
                    [
                        self.layers,
                        batch.batch,
                        self.heads,
                        self_width,
                        self.head_dim,
                    ],
                    self_k.into_boxed_slice(),
                ))?;
                let self_v_tensor = Tensor::from_array((
                    [
                        self.layers,
                        batch.batch,
                        self.heads,
                        self_width,
                        self.head_dim,
                    ],
                    self_v.into_boxed_slice(),
                ))?;
                let cross_k_tensor =
                    Tensor::from_array((cross_shape.clone(), cross_k.to_vec().into_boxed_slice()))?;
                let cross_v_tensor =
                    Tensor::from_array((cross_shape.clone(), cross_v.to_vec().into_boxed_slice()))?;
                let mask_tensor = Tensor::from_array((
                    [batch.batch, mask.len() / batch.batch],
                    mask.to_vec().into_boxed_slice(),
                ))?;
                let outputs = self.decoder_step.run(ort::inputs![
                    "token" => token_tensor,
                    "pos" => position_tensor,
                    "self_k" => self_k_tensor,
                    "self_v" => self_v_tensor,
                    "cross_k" => cross_k_tensor,
                    "cross_v" => cross_v_tensor,
                    "mask" => mask_tensor,
                ])?;
                let (_, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
                let (_, new_k) = outputs["new_k"].try_extract_tensor::<f32>()?;
                let (_, new_v) = outputs["new_v"].try_extract_tensor::<f32>()?;
                let vocab = logits.len() / batch.batch;
                for item in 0..batch.batch {
                    let row = &logits[item * vocab..(item + 1) * vocab];
                    let mut best_index = 0;
                    for (index, &value) in row.iter().enumerate().skip(1) {
                        if value > row[best_index] {
                            best_index = index;
                        }
                    }
                    let best_logit = row[best_index];
                    let denominator: f32 =
                        row.iter().map(|&value| (value - best_logit).exp()).sum();
                    let probability = 1.0 / denominator;
                    token[item] = best_index as i64;
                    if done[item] {
                        continue;
                    }
                    probabilities[item].push(probability);
                    if best_index as i64 == EOS
                        || (probabilities[item].len() >= 8
                            && average(&probabilities[item]) < MATH_MIN_CONF - EARLY_STOP_MARGIN)
                    {
                        done[item] = true;
                    } else if best_index as i64 >= N_SPECIAL {
                        sequences[item].push(best_index);
                    }
                }
                self_k = new_k.to_vec();
                self_v = new_v.to_vec();
                if done.iter().all(|&value| value) {
                    break;
                }
            }
            for (local, &index) in part.iter().enumerate() {
                let tokens: Vec<&str> = sequences[local]
                    .iter()
                    .filter_map(|&id| self.items.get(id))
                    .map(String::as_str)
                    .collect();
                out[index] = (detokenize(&tokens), average(&probabilities[local]));
            }
        }
        Ok(out)
    }
}

pub fn detokenize(tokens: &[&str]) -> String {
    let mut out = Vec::<String>::new();
    let mut math = Vec::<&str>::new();
    let mut in_math = false;
    for &token in tokens {
        if token == "$" {
            if in_math {
                out.push(format!("${}$", math.join(" ")));
                math.clear();
            }
            in_math = !in_math;
        } else if in_math {
            math.push(token);
        } else {
            out.push(token.to_string());
        }
    }
    if in_math && !math.is_empty() {
        out.push(format!("${}$", math.join(" ")));
    }
    out.concat()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn average(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn load_math_session(environment: &Environment, path: &Path, threads: usize) -> Result<Session> {
    let builder =
        Session::builder(environment).map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let builder = builder
        .with_intra_threads(threads.max(1))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let builder = builder
        .with_inter_threads(1)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut builder = builder
        .with_memory_pattern(false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    builder
        .commit_from_file(path)
        .with_context(|| format!("could not load {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detokenizes_inline_math() {
        assert_eq!(
            detokenize(&["x", " ", "=", " ", "$", "\\frac", "1", "2", "$"]),
            "x = $\\frac 1 2$"
        );
    }

    #[test]
    #[ignore = "requires the downloaded ONNX models"]
    fn math_decoder_smoke_test() {
        let environment = ort::init()
            .with_name("indra-ocr-math-test")
            .build()
            .unwrap();
        let mut reader = MathReader::load(Path::new("models"), &environment, 2)
            .unwrap()
            .expect("math models");
        let image = image::open("assets/examples/multi_column.png")
            .unwrap()
            .into_luma8();
        let readings = reader
            .read(
                &image,
                &[Box4 {
                    x0: 35.3,
                    y0: 67.7,
                    x1: 437.7,
                    y1: 80.7,
                }],
                1,
            )
            .unwrap();
        assert_eq!(readings.len(), 1);
        assert!(readings[0].1.is_finite());
    }
}
