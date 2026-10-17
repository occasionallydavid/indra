use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use serde::Deserialize;

use crate::frame::GrayFrame;
use crate::model::{math_dec_init, math_dec_step, math_enc, mathcls};
use crate::order::Box4;
use crate::preprocess::{LineCrop, WIDTH_MULTIPLE, crop_line, pad_batch, width_batches};

const MATH_MIN: f32 = 0.5;
pub const MATH_MIN_CONF: f32 = 0.9;
const EARLY_STOP_MARGIN: f32 = 0.1;
const BOS: i64 = 1;
const EOS: i64 = 2;
const N_SPECIAL: i64 = 4;

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
    device: Device,
    classifier: mathcls::Model,
    encoder: math_enc::Model,
    decoder_init: math_dec_init::Model,
    decoder_step: math_dec_step::Model,
    height: usize,
    max_tokens: usize,
    layers: usize,
    heads: usize,
    head_dim: usize,
    items: Vec<String>,
}

impl MathReader {
    pub fn load(device: &Device) -> Result<Self> {
        let meta: MathMeta = serde_json::from_str(include_str!("../models/math_meta.json"))
            .context("invalid math_meta.json")?;
        let mut items = vec![String::new(), String::new(), String::new(), String::new()];
        items.extend(meta.tokens);
        Ok(Self {
            device: device.clone(),
            classifier: mathcls::Model::from_embedded(device),
            encoder: math_enc::Model::from_embedded(device),
            decoder_init: math_dec_init::Model::from_embedded(device),
            decoder_step: math_dec_step::Model::from_embedded(device),
            height: meta.config.height,
            max_tokens: meta.config.max_tokens,
            layers: meta.config.dec_layers,
            heads: meta.config.dec_heads,
            head_dim: meta.config.channels / meta.config.dec_heads,
            items,
        })
    }

    pub fn flags(&mut self, crops: &[LineCrop], batch: usize) -> Result<Vec<bool>> {
        let mut out = vec![false; crops.len()];
        let widths: Vec<usize> = crops.iter().map(|crop| crop.width).collect();
        for part in width_batches(&widths, batch, 1.3, 64, 64 * 640) {
            let lines: Vec<LineCrop> = part.iter().map(|&index| crops[index].clone()).collect();
            let batch = pad_batch(&lines, WIDTH_MULTIPLE);
            let x = Tensor::from_data(
                TensorData::new(batch.data, [batch.batch, 1, batch.height, batch.width]),
                &self.device,
            );
            let widths = Tensor::from_data(
                TensorData::new(batch.widths, [batch.batch])
                    .convert_dtype(burn::tensor::DType::I64),
                (&self.device, burn::tensor::DType::I64),
            );
            let (probabilities, _) = tensor_f32(self.classifier.forward(x, widths));
            for (local, &index) in part.iter().enumerate() {
                out[index] = probabilities[local] > MATH_MIN;
            }
        }
        Ok(out)
    }

    pub fn read(
        &mut self,
        gray: &GrayFrame,
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
            let x = Tensor::from_data(
                TensorData::new(batch.data, [batch.batch, 1, batch.height, batch.width]),
                &self.device,
            );
            let widths = Tensor::from_data(
                TensorData::new(batch.widths, [batch.batch])
                    .convert_dtype(burn::tensor::DType::I64),
                (&self.device, burn::tensor::DType::I64),
            );
            let (memory, mask) = self.encoder.forward(x, widths);
            let (cross_k, cross_v) = self.decoder_init.forward(memory);
            let mut self_k = Tensor::<5>::zeros(
                [self.layers, batch.batch, self.heads, 0, self.head_dim],
                &self.device,
            );
            let mut self_v = self_k.clone();
            let mut token = vec![BOS; batch.batch];
            let mut sequences = vec![Vec::<usize>::new(); batch.batch];
            let mut probabilities = vec![Vec::<f32>::new(); batch.batch];
            let mut done = vec![false; batch.batch];
            for position in 0..self.max_tokens.saturating_sub(1) {
                let token_tensor = Tensor::from_data(
                    TensorData::new(token.clone(), [batch.batch])
                        .convert_dtype(burn::tensor::DType::I64),
                    (&self.device, burn::tensor::DType::I64),
                );
                let (logits_tensor, new_k, new_v) = self.decoder_step.forward(
                    token_tensor,
                    position as i64,
                    self_k,
                    self_v,
                    cross_k.clone(),
                    cross_v.clone(),
                    mask.clone(),
                );
                let (logits, logits_dims) = tensor_f32(logits_tensor);
                let vocab = logits_dims[1];
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
                self_k = new_k;
                self_v = new_v;
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

fn tensor_f32<const D: usize>(tensor: Tensor<D>) -> (Vec<f32>, [usize; D]) {
    let dims = tensor.dims();
    (tensor.into_data().try_into_vec::<f32>().unwrap(), dims)
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
    fn math_decoder_smoke_test() {
        let device = Device::default();
        let mut reader = MathReader::load(&device).unwrap();
        let image = image::open("assets/examples/multi_column.png")
            .unwrap()
            .into_luma8();
        let (width, height) = (image.width() as usize, image.height() as usize);
        let image = GrayFrame::from_gray(width, height, image.into_raw());
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
