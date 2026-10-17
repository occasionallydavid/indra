use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use image::GrayImage;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use serde::Serialize;

use crate::image_ops::{
    resize_gray, rotate_f32_constant, rotate_gray_replicate, rotate_quarter_turns_clockwise,
};
use crate::layout::{Block as LayoutBlock, page_blocks, page_markdown};
use crate::math::{MATH_MIN_CONF, MathReader};
use crate::order::{
    Box4, Layout, N_FEATS, decode_blocks_order, line_features, median, order_xycut,
};
use crate::preprocess::{
    OUT_STRIDE, PAD_X, WIDTH_MULTIPLE, crop_line, pad_batch, page_to_input, width_batches,
};
use crate::vocab::{Token, Vocab};

const ADAPTIVE_MIN_H: f32 = 9.0;
const ADAPTIVE_TARGET_H: f32 = 12.0;
const ADAPTIVE_MAX_SIDE: usize = 2048;
const SPLIT_GAP: f32 = 3.0;
const CELL_GAP: f32 = 1.8;
const GUTTER_GAP: f32 = 1.6;
const GUTTER_EDGES: usize = 3;
const MIN_CONF: f32 = 0.5;
const ORIENT_SIDE: usize = 768;
const ORIENT_MIN: f32 = 0.95;
const DESKEW_MAX: f32 = 5.0;
const DESKEW_MIN: f32 = 0.3;

pub type Timing = BTreeMap<String, f64>;
type OrderingResult = (Vec<usize>, Option<Layout>, Option<Vec<f32>>);

#[derive(Clone, Debug, Serialize)]
pub struct Line {
    pub bbox: [f32; 4],
    pub text: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Page {
    pub lines: Vec<Line>,
    pub markdown: Option<String>,
    pub blocks: Vec<LayoutBlock>,
    pub source: String,
    pub note: String,
    pub rotation: i32,
    pub skew: f32,
    pub timings: Timing,
}

impl Page {
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct InputError(pub String);

#[derive(Clone, Debug)]
struct Recognition {
    text: String,
    confidence: f32,
    tokens: Vec<Token>,
}

impl Recognition {
    fn empty() -> Self {
        Self {
            text: String::new(),
            confidence: 0.0,
            tokens: Vec::new(),
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct Meta {
    det_long_side: usize,
    rec_height: usize,
    vocab: String,
    vocab_cjk: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct Detection {
    bbox: Box4,
    _score: f32,
}

pub struct Ocr {
    _environment: Environment,
    det: Session,
    rec: Session,
    script: Option<Session>,
    rec_cjk: Option<Session>,
    order: Option<Session>,
    orient: Option<Session>,
    vocab: Vocab,
    vocab_cjk: Option<Vocab>,
    math: Option<MathReader>,
    model_dir: PathBuf,
    threads: usize,
    det_long_side: usize,
    rec_height: usize,
    adaptive: bool,
    batch: usize,
    table_heads: bool,
}

impl Ocr {
    pub fn from_dir(model_dir: impl AsRef<Path>, threads: usize) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        if !model_dir.is_dir() {
            bail!("model directory does not exist: {}", model_dir.display());
        }
        let meta_path = model_dir.join("meta.json");
        let meta_text = fs::read_to_string(&meta_path)
            .with_context(|| format!("could not read {}", meta_path.display()))?;
        let meta: Meta = serde_json::from_str(&meta_text)
            .with_context(|| format!("invalid {}", meta_path.display()))?;
        let vocab = Vocab::from_json(&meta.vocab).context("invalid main vocabulary")?;
        let vocab_cjk = meta
            .vocab_cjk
            .as_deref()
            .map(Vocab::from_json)
            .transpose()
            .context("invalid CJK vocabulary")?;

        let rec_path = if model_dir.join("rec.int8.onnx").is_file() {
            model_dir.join("rec.int8.onnx")
        } else {
            model_dir.join("rec.onnx")
        };
        let environment = ort::init()
            .with_name("indra-ocr")
            .build()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let det = load_session(&environment, &model_dir.join("det.onnx"), threads)?;
        let rec = load_session(&environment, &rec_path, threads)?;
        let orient = load_optional_session(&environment, &model_dir.join("orient.onnx"), threads)?;
        let order = load_optional_session(&environment, &model_dir.join("order.onnx"), threads)?;
        let script_path = model_dir.join("script.onnx");
        let cjk_path = model_dir.join("rec_cjk.onnx");
        let (script, rec_cjk) = if script_path.is_file() && cjk_path.is_file() {
            (
                Some(load_session(&environment, &script_path, threads)?),
                Some(load_session(&environment, &cjk_path, threads)?),
            )
        } else {
            (None, None)
        };
        if rec_cjk.is_some() && vocab_cjk.is_none() {
            bail!("rec_cjk.onnx is present but meta.json has no vocab_cjk");
        }
        Ok(Self {
            _environment: environment,
            det,
            rec,
            script,
            rec_cjk,
            order,
            orient,
            vocab,
            vocab_cjk,
            math: None,
            model_dir: model_dir.to_path_buf(),
            threads,
            det_long_side: meta.det_long_side,
            rec_height: meta.rec_height,
            adaptive: true,
            batch: 64,
            table_heads: false,
        })
    }

    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    pub fn with_batch(mut self, batch: usize) -> Self {
        self.batch = batch.max(1);
        self
    }

    pub fn with_table_heads(mut self, table_heads: bool) -> Self {
        self.table_heads = table_heads;
        self
    }

    pub fn enable_math(&mut self) -> Result<()> {
        self.math = MathReader::load(&self.model_dir, &self._environment, self.threads)?;
        if self.math.is_none() {
            bail!(
                "--math needs the five math model files in {}",
                self.model_dir.display()
            );
        }
        Ok(())
    }

    pub fn ocr_image(&mut self, image: &GrayImage, orient: bool) -> Result<Page> {
        let mut timings = Timing::new();
        let start = Instant::now();
        let (upright, turns, skew) = self.upright(image, orient)?;
        timings.insert("orient_ms".into(), elapsed_ms(start.elapsed()));

        let start = Instant::now();
        let boxes = self.detect(&upright)?;
        timings.insert("det_ms".into(), elapsed_ms(start.elapsed()));

        let start = Instant::now();
        let recognition = self.recognize(&upright, &boxes)?;
        timings.insert("rec_ms".into(), elapsed_ms(start.elapsed()));

        let mut page = self.assemble(
            (upright.width() as usize, upright.height() as usize),
            &boxes,
            recognition,
            timings,
        )?;
        page.rotation = 90 * turns as i32;
        page.skew = (skew * 100.0).round() / 100.0;
        if skew != 0.0 {
            let wh = (upright.width() as f32, upright.height() as f32);
            for line in &mut page.lines {
                line.bbox = unrotate_box(line.bbox, skew, wh);
            }
            for block in &mut page.blocks {
                block.bbox = unrotate_box(block.bbox, skew, wh);
            }
        }
        Ok(page)
    }

    fn detect(&mut self, gray: &GrayImage) -> Result<Vec<Detection>> {
        let (mut boxes, mut scale) = self.detect_at(gray, self.det_long_side)?;
        if self.adaptive && boxes.len() >= 8 {
            let heights: Vec<f32> = boxes.iter().map(|b| b.bbox.height()).collect();
            let median_height = median(&heights);
            if median_height < ADAPTIVE_MIN_H {
                let long_side = ((self.det_long_side as f32 * ADAPTIVE_TARGET_H
                    / median_height.max(1.0))
                .min(ADAPTIVE_MAX_SIDE as f32) as usize)
                    / 32
                    * 32;
                if long_side > self.det_long_side {
                    (boxes, scale) = self.detect_at(gray, long_side)?;
                }
            }
        }
        let width = gray.width() as f32;
        let height = gray.height() as f32;
        for detection in &mut boxes {
            detection.bbox.x0 = (detection.bbox.x0 / scale).clamp(0.0, width);
            detection.bbox.x1 = (detection.bbox.x1 / scale).clamp(0.0, width);
            detection.bbox.y0 = (detection.bbox.y0 / scale).clamp(0.0, height);
            detection.bbox.y1 = (detection.bbox.y1 / scale).clamp(0.0, height);
        }
        Ok(boxes)
    }

    fn detect_at(&mut self, gray: &GrayImage, long_side: usize) -> Result<(Vec<Detection>, f32)> {
        let input = page_to_input(gray, long_side);
        let tensor = Tensor::from_array((
            [1_usize, 1, input.height, input.width],
            input.data.into_boxed_slice(),
        ))?;
        let outputs = self.det.run(ort::inputs!["x" => tensor])?;
        let (shape, output) = outputs[0].try_extract_tensor::<f32>()?;
        let out_h = shape[2] as usize;
        let out_w = shape[3] as usize;
        Ok((decode_detection(output, out_h, out_w), input.scale))
    }

    fn recognize(&mut self, gray: &GrayImage, boxes: &[Detection]) -> Result<Vec<Recognition>> {
        let bboxes: Vec<[f32; 4]> = boxes.iter().map(|b| b.bbox.as_array()).collect();
        let crops: Vec<_> = bboxes
            .iter()
            .map(|&bbox| crop_line(gray, bbox, self.rec_height, 4096))
            .collect();
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = vec![Recognition::empty(); crops.len()];
        let hanzi = if let Some(script) = self.script.as_mut() {
            Self::hanzi_lines(script, &crops, self.batch)?
        } else {
            vec![false; crops.len()]
        };
        let non_cjk: Vec<usize> = (0..crops.len()).filter(|&index| !hanzi[index]).collect();
        Self::recognize_group(
            &mut self.rec,
            &self.vocab,
            false,
            &crops,
            boxes,
            &non_cjk,
            self.batch,
            &mut out,
        )?;
        if let (Some(session), Some(vocab)) = (self.rec_cjk.as_mut(), self.vocab_cjk.as_ref()) {
            let cjk: Vec<usize> = (0..crops.len()).filter(|&index| hanzi[index]).collect();
            Self::recognize_group(
                session, vocab, true, &crops, boxes, &cjk, self.batch, &mut out,
            )?;
        }
        if let Some(math) = self.math.as_mut() {
            let flags = math.flags(&crops, self.batch)?;
            let indices: Vec<usize> = flags
                .iter()
                .enumerate()
                .filter_map(|(index, &is_math)| is_math.then_some(index))
                .collect();
            if !indices.is_empty() {
                let math_boxes: Vec<Box4> =
                    indices.iter().map(|&index| boxes[index].bbox).collect();
                let readings = math.read(gray, &math_boxes, self.batch)?;
                for (&index, (text, confidence)) in indices.iter().zip(readings) {
                    if text.contains('$') && confidence >= MATH_MIN_CONF {
                        let center_x = boxes[index].bbox.center_x();
                        out[index] = Recognition {
                            text: text.clone(),
                            confidence: 1.0,
                            tokens: vec![Token::Char { text, x: center_x }],
                        };
                    }
                }
            }
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn recognize_group(
        session: &mut Session,
        vocab: &Vocab,
        cjk: bool,
        crops: &[crate::preprocess::LineCrop],
        boxes: &[Detection],
        selected: &[usize],
        max_batch: usize,
        out: &mut [Recognition],
    ) -> Result<()> {
        let widths: Vec<usize> = selected.iter().map(|&index| crops[index].width).collect();
        for part in width_batches(&widths, max_batch, 1.3, 64, 64 * 640) {
            let indices: Vec<usize> = part.iter().map(|&index| selected[index]).collect();
            let lines: Vec<_> = indices.iter().map(|&index| crops[index].clone()).collect();
            let batch = pad_batch(&lines, WIDTH_MULTIPLE);
            let x = Tensor::from_array((
                [batch.batch, 1, batch.height, batch.width],
                batch.data.into_boxed_slice(),
            ))?;
            let widths_tensor =
                Tensor::from_array(([batch.batch], batch.widths.into_boxed_slice()))?;
            let outputs = session.run(ort::inputs!["x" => x, "widths" => widths_tensor])?;
            let (ids_shape, ids) = outputs["ids"].try_extract_tensor::<i64>()?;
            let (_, probabilities) = outputs["conf"].try_extract_tensor::<f32>()?;
            let (_, lengths) = outputs["lengths"].try_extract_tensor::<i64>()?;
            let timesteps = ids_shape[1] as usize;
            let stride = batch.width as f32 / timesteps.max(1) as f32;
            for (batch_index, &line_index) in indices.iter().enumerate() {
                let n = lengths[batch_index].max(0) as usize;
                let row = &ids[batch_index * timesteps..batch_index * timesteps + n];
                let confidence_row =
                    &probabilities[batch_index * timesteps..batch_index * timesteps + n];
                let mut keep = Vec::with_capacity(n);
                let mut previous = -1_i64;
                for (step, &id) in row.iter().enumerate() {
                    if id != 0 && id != previous {
                        keep.push(step);
                    }
                    previous = id;
                }
                let bbox = boxes[line_index].bbox;
                let line_height = bbox.height().max(1.0);
                let ix0 = (bbox.x0 - PAD_X * line_height).floor();
                let crop_width = (bbox.x1 + PAD_X * line_height).ceil() - ix0;
                let factor = crop_width / crops[line_index].width as f32;
                let xs: Vec<f32> = keep
                    .iter()
                    .map(|&step| ix0 + (step as f32 + 0.5) * stride * factor)
                    .collect();
                let ids_kept: Vec<usize> = keep.iter().map(|&step| row[step] as usize).collect();
                let confidence = if keep.is_empty() {
                    0.0
                } else {
                    keep.iter().map(|&step| confidence_row[step]).sum::<f32>() / keep.len() as f32
                };
                let tokens: Vec<Token> = if cjk {
                    ids_kept
                        .iter()
                        .zip(xs)
                        .filter_map(|(&id, x)| {
                            vocab
                                .chars()
                                .get(id.saturating_sub(1))
                                .map(|text| Token::Char {
                                    text: text.clone(),
                                    x,
                                })
                        })
                        .collect()
                } else {
                    ids_kept
                        .iter()
                        .zip(xs)
                        .map(|(&id, x)| Token::Id { id, x })
                        .collect()
                };
                let text = if cjk {
                    vocab.decode_tokens(&tokens)
                } else {
                    vocab.decode(&ids_kept)
                };
                out[line_index] = if confidence >= MIN_CONF {
                    Recognition {
                        text,
                        confidence,
                        tokens,
                    }
                } else {
                    Recognition {
                        text: String::new(),
                        confidence,
                        tokens: Vec::new(),
                    }
                };
            }
        }
        Ok(())
    }

    fn hanzi_lines(
        session: &mut Session,
        crops: &[crate::preprocess::LineCrop],
        max_batch: usize,
    ) -> Result<Vec<bool>> {
        let mut out = vec![false; crops.len()];
        let widths: Vec<usize> = crops.iter().map(|crop| crop.width).collect();
        for part in width_batches(&widths, max_batch, 1.3, 64, 64 * 640) {
            let lines: Vec<_> = part.iter().map(|&index| crops[index].clone()).collect();
            let batch = pad_batch(&lines, WIDTH_MULTIPLE);
            let x = Tensor::from_array((
                [batch.batch, 1, batch.height, batch.width],
                batch.data.into_boxed_slice(),
            ))?;
            let widths_tensor =
                Tensor::from_array(([batch.batch], batch.widths.into_boxed_slice()))?;
            let outputs = session.run(ort::inputs!["x" => x, "widths" => widths_tensor])?;
            let (_, probabilities) = outputs["p"].try_extract_tensor::<f32>()?;
            for (local, &index) in part.iter().enumerate() {
                out[index] = probabilities[local] > 0.5;
            }
        }
        Ok(out)
    }

    fn orientation(&mut self, gray: &GrayImage) -> Result<usize> {
        let Some(orient) = self.orient.as_mut() else {
            return Ok(0);
        };
        let (scaled, width, height) = orient_input(gray);
        let tensor = Tensor::from_array((
            [1_usize, 1, ORIENT_SIDE, ORIENT_SIDE],
            scaled.into_boxed_slice(),
        ))?;
        let _ = (width, height);
        let outputs = orient.run(ort::inputs!["x" => tensor])?;
        let (_, probabilities) = outputs["p"].try_extract_tensor::<f32>()?;
        let turn = crate::order::argmax(probabilities);
        Ok(if probabilities[turn] > ORIENT_MIN {
            turn
        } else {
            0
        })
    }

    fn upright(&mut self, gray: &GrayImage, orient: bool) -> Result<(GrayImage, usize, f32)> {
        let turns = if orient { self.orientation(gray)? } else { 0 };
        let mut upright = rotate_quarter_turns_clockwise(gray.clone(), turns);
        let mut skew = estimate_skew(&upright);
        if skew.abs() >= DESKEW_MIN {
            upright = rotate_gray_replicate(&upright, skew);
        } else {
            skew = 0.0;
        }
        Ok((upright, turns, skew))
    }

    fn reading_order(
        &mut self,
        boxes: &[Box4],
        texts: &[String],
        page_wh: (usize, usize),
    ) -> Result<OrderingResult> {
        if boxes.len() < 2 {
            return Ok((order_xycut(boxes), None, None));
        }
        let Some(order) = self.order.as_mut() else {
            return Ok((order_xycut(boxes), None, None));
        };
        let features = line_features(boxes, page_wh, texts);
        let feats =
            Tensor::from_array(([1_usize, boxes.len(), N_FEATS], features.into_boxed_slice()))?;
        let mask = Tensor::from_array((
            [1_usize, boxes.len()],
            vec![true; boxes.len()].into_boxed_slice(),
        ))?;
        let outputs = order.run(ort::inputs!["feats" => feats, "mask" => mask])?;
        let (_, pair_logits) = outputs["a"].try_extract_tensor::<f32>()?;
        let (_, cls) = outputs["cls"].try_extract_tensor::<f32>()?;
        let (_, same) = outputs["same"].try_extract_tensor::<f32>()?;
        let (_, table) = outputs["tab"].try_extract_tensor::<f32>()?;
        let pair_logits = pair_logits.to_vec();
        let layout = Layout {
            n: boxes.len(),
            cls: cls.to_vec(),
            same: same.to_vec(),
            table: self.table_heads.then(|| table.to_vec()),
        };
        let ordered = decode_blocks_order(&pair_logits, &layout, boxes);
        Ok((ordered, Some(layout), Some(pair_logits)))
    }

    fn assemble(
        &mut self,
        page_wh: (usize, usize),
        detections: &[Detection],
        recognition: Vec<Recognition>,
        mut timings: Timing,
    ) -> Result<Page> {
        let order_start = Instant::now();
        let boxes: Vec<Box4> = detections.iter().map(|detection| detection.bbox).collect();
        let cuts = self.gutter_cuts(&boxes, &recognition);
        let mut split_boxes = Vec::new();
        let mut split_recognition = Vec::new();
        for ((bbox, rec), at) in boxes.iter().zip(&recognition).zip(cuts) {
            for (new_box, text, tokens) in self.split_line(*bbox, &rec.tokens, SPLIT_GAP, &at) {
                split_boxes.push(new_box);
                split_recognition.push(Recognition {
                    text,
                    confidence: rec.confidence,
                    tokens,
                });
            }
        }
        let texts: Vec<String> = split_recognition
            .iter()
            .map(|recognition| recognition.text.clone())
            .collect();
        let (mut order, mut layout, mut pair_logits) =
            self.reading_order(&split_boxes, &texts, page_wh)?;

        if let Some(current_layout) = &layout {
            let table_class = 5;
            let is_table: Vec<bool> = (0..split_boxes.len())
                .map(|index| {
                    crate::order::argmax(current_layout.class_probabilities(index)) == table_class
                })
                .collect();
            if is_table.iter().any(|&value| value) {
                let mut new_boxes = Vec::new();
                let mut new_recognition = Vec::new();
                for (index, bbox) in split_boxes.iter().enumerate() {
                    let parts = if is_table[index] {
                        self.split_line(*bbox, &split_recognition[index].tokens, CELL_GAP, &[])
                    } else {
                        vec![(
                            *bbox,
                            split_recognition[index].text.clone(),
                            split_recognition[index].tokens.clone(),
                        )]
                    };
                    for (new_box, text, tokens) in parts {
                        new_boxes.push(new_box);
                        new_recognition.push(Recognition {
                            text,
                            confidence: split_recognition[index].confidence,
                            tokens,
                        });
                    }
                }
                if new_boxes.len() != split_boxes.len() {
                    split_boxes = new_boxes;
                    split_recognition = new_recognition;
                    let texts: Vec<String> = split_recognition
                        .iter()
                        .map(|recognition| recognition.text.clone())
                        .collect();
                    let (new_order, new_layout, new_pair_logits) =
                        self.reading_order(&split_boxes, &texts, page_wh)?;
                    order = new_order;
                    layout = new_layout;
                    pair_logits = new_pair_logits;
                }
            }
        }
        timings.insert("order_ms".into(), elapsed_ms(order_start.elapsed()));
        timings.insert(
            "total_ms".into(),
            timings
                .iter()
                .filter(|(key, _)| key.as_str() != "total_ms")
                .map(|(_, value)| *value)
                .sum(),
        );
        let lines = order
            .iter()
            .map(|&index| {
                let bbox = split_boxes[index].as_array().map(round1);
                Line {
                    bbox,
                    text: split_recognition[index].text.clone(),
                    confidence: (split_recognition[index].confidence * 10_000.0).round() / 10_000.0,
                }
            })
            .collect();
        let texts: Vec<String> = split_recognition
            .iter()
            .map(|recognition| recognition.text.clone())
            .collect();
        let (markdown, blocks) = if let Some(layout) = &layout {
            let blocks = page_blocks(&split_boxes, &texts, &order, layout);
            let cells: Vec<Vec<(Box4, String)>> = split_boxes
                .iter()
                .enumerate()
                .map(|(index, bbox)| {
                    self.split_line(*bbox, &split_recognition[index].tokens, CELL_GAP, &[])
                        .into_iter()
                        .map(|(cell_box, text, _)| (cell_box, text))
                        .collect()
                })
                .collect();
            (
                Some(page_markdown(
                    &split_boxes,
                    &texts,
                    &order,
                    layout,
                    false,
                    &cells,
                    layout.table.as_deref(),
                )),
                blocks,
            )
        } else if split_boxes.len() < 2 {
            (Some(texts.join("\n")), Vec::new())
        } else {
            (None, Vec::new())
        };
        let _ = pair_logits;
        Ok(Page {
            lines,
            markdown,
            blocks,
            source: "ocr".into(),
            note: "order=model".into(),
            rotation: 0,
            skew: 0.0,
            timings,
        })
    }

    fn gutter_cuts(&self, boxes: &[Box4], recognition: &[Recognition]) -> Vec<Vec<f32>> {
        let mut out = Vec::with_capacity(boxes.len());
        for (index, (bbox, recognition)) in boxes.iter().zip(recognition).enumerate() {
            let height = bbox.height().max(1.0);
            let xs: Vec<f32> = recognition
                .tokens
                .iter()
                .filter(|token| !self.is_space(token))
                .map(Token::x)
                .collect();
            let mut cuts = Vec::new();
            for pair in xs.windows(2) {
                let (a, c) = (pair[0], pair[1]);
                if c - a <= GUTTER_GAP * height {
                    continue;
                }
                let left = a + 0.35 * height;
                let right = c - 0.35 * height;
                let near = boxes
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .filter(|(_, other)| {
                        (other.x1 - left).abs() < 0.5 * height
                            || (other.x0 - right).abs() < 0.5 * height
                    })
                    .count();
                if near >= GUTTER_EDGES {
                    cuts.push((a + c) / 2.0);
                }
            }
            out.push(cuts);
        }
        out
    }

    fn is_space(&self, token: &Token) -> bool {
        match token {
            Token::Char { text, .. } => text == " ",
            Token::Id { id, .. } => self.vocab.space_id() == Some(*id),
        }
    }

    fn split_line(
        &self,
        bbox: Box4,
        tokens: &[Token],
        gap: f32,
        cuts: &[f32],
    ) -> Vec<(Box4, String, Vec<Token>)> {
        if tokens.is_empty() {
            return vec![(bbox, String::new(), Vec::new())];
        }
        let height = bbox.height().max(1.0);
        let mut cuts = cuts.to_vec();
        cuts.sort_by(f32::total_cmp);
        let mut segments: Vec<Vec<Token>> = vec![Vec::new()];
        let mut previous: Option<f32> = None;
        for token in tokens {
            let x = token.x();
            if !self.is_space(token) {
                if let Some(previous) = previous {
                    let should_cut = x - previous > gap * height
                        || cuts.iter().any(|&cut| previous < cut && cut < x);
                    if should_cut {
                        segments.push(Vec::new());
                    }
                }
                previous = Some(x);
            }
            segments.last_mut().unwrap().push(token.clone());
        }
        if segments.len() == 1 {
            return vec![(bbox, self.vocab.decode_tokens(tokens), tokens.to_vec())];
        }
        let mut out = Vec::new();
        for segment in segments {
            let text = self.vocab.decode_tokens(&segment).trim().to_string();
            let xs: Vec<f32> = segment
                .iter()
                .filter(|token| !self.is_space(token))
                .map(Token::x)
                .collect();
            if !text.is_empty() && !xs.is_empty() {
                let min = xs.iter().copied().fold(f32::INFINITY, f32::min);
                let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let new_box = Box4 {
                    x0: bbox.x0.max(min - 0.35 * height),
                    y0: bbox.y0,
                    x1: bbox.x1.min(max + 0.35 * height),
                    y1: bbox.y1,
                };
                out.push((new_box, text, segment));
            }
        }
        out
    }
}

fn load_session(environment: &Environment, path: &Path, threads: usize) -> Result<Session> {
    if !path.is_file() {
        bail!("missing ONNX model: {}", path.display());
    }
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

fn load_optional_session(
    environment: &Environment,
    path: &Path,
    threads: usize,
) -> Result<Option<Session>> {
    path.is_file()
        .then(|| load_session(environment, path, threads))
        .transpose()
}

pub fn load_image(path: impl AsRef<Path>) -> Result<GrayImage, InputError> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(InputError(format!("{}: no such file", path.display())));
    }
    image::open(path)
        .map(|image| image.into_luma8())
        .map_err(|error| {
            InputError(format!(
                "{}: not a readable image ({error})",
                path.display()
            ))
        })
}

pub fn default_model_dir() -> PathBuf {
    std::env::var_os("INDRA_OCR_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

fn decode_detection(output: &[f32], height: usize, width: usize) -> Vec<Detection> {
    let plane = height * width;
    let mut labels = vec![0_u32; plane];
    let mut boxes = Vec::new();
    let mut next_label = 1_u32;
    let mut stack = Vec::new();
    for start in 0..plane {
        if labels[start] != 0 || sigmoid(output[start]) <= 0.5 {
            continue;
        }
        let label = next_label;
        next_label += 1;
        labels[start] = label;
        stack.push(start);
        let mut area = 0_u32;
        let mut min_x = width;
        let mut max_x = 0;
        let mut min_y = height;
        let mut max_y = 0;
        let mut sum_top = 0.0_f64;
        let mut sum_bottom = 0.0_f64;
        let mut sum_score = 0.0_f64;
        while let Some(index) = stack.pop() {
            let y = index / width;
            let x = index % width;
            area += 1;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            let center_y = (y as f32 + 0.5) * OUT_STRIDE;
            sum_top += (center_y - output[plane + index]) as f64;
            sum_bottom += (center_y + output[2 * plane + index]) as f64;
            sum_score += sigmoid(output[index]) as f64;
            let neighbors = [
                (x > 0).then_some(index - 1),
                (x + 1 < width).then_some(index + 1),
                (y > 0).then_some(index - width),
                (y + 1 < height).then_some(index + width),
            ];
            for neighbor in neighbors.into_iter().flatten() {
                if labels[neighbor] == 0 && sigmoid(output[neighbor]) > 0.5 {
                    labels[neighbor] = label;
                    stack.push(neighbor);
                }
            }
        }
        if area < 2 {
            continue;
        }
        let count = area as f64;
        let y0 = (sum_top / count) as f32;
        let y1 = (sum_bottom / count) as f32;
        let line_height = (y1 - y0).max(1.0);
        let extension = (0.15 * line_height).min(0.5 * line_height);
        boxes.push(Detection {
            bbox: Box4 {
                x0: min_x as f32 * OUT_STRIDE - extension,
                y0,
                x1: (max_x + 1) as f32 * OUT_STRIDE + extension,
                y1,
            },
            _score: (sum_score / count) as f32,
        });
    }
    boxes
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn orient_input(gray: &GrayImage) -> (Vec<f32>, usize, usize) {
    let (h, w) = (gray.height() as usize, gray.width() as usize);
    let scale = ORIENT_SIDE as f32 / h.max(w) as f32;
    let nh = ((h as f32 * scale).round() as usize).max(1);
    let nw = ((w as f32 * scale).round() as usize).max(1);
    let resized = resize_gray(gray, nw as u32, nh as u32);
    let mut out = vec![0.0; ORIENT_SIDE * ORIENT_SIDE];
    for y in 0..nh {
        for x in 0..nw {
            out[y * ORIENT_SIDE + x] =
                1.0 - resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        }
    }
    (out, nw, nh)
}

pub fn estimate_skew(gray: &GrayImage) -> f32 {
    over_thumbnail(gray, 800, |small| {
        let width = small.width() as usize;
        let height = small.height() as usize;
        let mut ink = vec![0.0; width * height];
        let mut sum = 0.0_f64;
        for y in 0..height {
            for x in 0..width {
                let value = 1.0 - small.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
                ink[y * width + x] = value;
                sum += value as f64;
            }
        }
        if sum / (ink.len() as f64) < 0.003 {
            return 0.0;
        }
        let sharpness = |angle: f32| -> f32 {
            let rotated = rotate_f32_constant(&ink, width, height, angle);
            let mut profile = vec![0.0_f32; height];
            for y in 0..height {
                profile[y] = rotated[y * width..(y + 1) * width].iter().sum();
            }
            profile
                .windows(2)
                .map(|pair| {
                    let difference = pair[1] - pair[0];
                    difference * difference
                })
                .sum()
        };
        let mut best = -DESKEW_MAX;
        let mut best_score = f32::NEG_INFINITY;
        let mut angle = -DESKEW_MAX;
        while angle <= DESKEW_MAX + 1e-6 {
            let score = sharpness(angle);
            if score > best_score {
                best_score = score;
                best = angle;
            }
            angle += 0.5;
        }
        let mut fine_best = best;
        let mut fine_score = f32::NEG_INFINITY;
        let mut angle = best - 0.5;
        while angle <= best + 0.5 + 1e-6 {
            let score = sharpness(angle);
            if score > fine_score {
                fine_score = score;
                fine_best = angle;
            }
            angle += 0.1;
        }
        fine_best
    })
}

fn over_thumbnail<T>(gray: &GrayImage, side: usize, f: impl FnOnce(&GrayImage) -> T) -> T {
    let scale = side as f32 / gray.width().max(gray.height()) as f32;
    let width = ((gray.width() as f32 * scale).round() as u32).max(1);
    let height = ((gray.height() as f32 * scale).round() as u32).max(1);
    f(&resize_gray(gray, width, height))
}

fn unrotate_box(bbox: [f32; 4], degrees: f32, (width, height): (f32, f32)) -> [f32; 4] {
    let radians = (-degrees).to_radians();
    let (sin, cos) = radians.sin_cos();
    let cx = width / 2.0;
    let cy = height / 2.0;
    let corners = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[1]),
        (bbox[0], bbox[3]),
        (bbox[2], bbox[3]),
    ];
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for (x, y) in corners {
        let dx = x - cx;
        let dy = y - cy;
        let mapped_x = cos * dx - sin * dy + cx;
        let mapped_y = sin * dx + cos * dy + cy;
        min_x = min_x.min(mapped_x);
        min_y = min_y.min(mapped_y);
        max_x = max_x.max(mapped_x);
        max_y = max_y.max(mapped_y);
    }
    [
        round1(min_x.clamp(0.0, width)),
        round1(min_y.clamp(0.0, height)),
        round1(max_x.clamp(0.0, width)),
        round1(max_y.clamp(0.0, height)),
    ]
}

fn round1(value: f32) -> f32 {
    (value * 10.0).round() / 10.0
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_decodes_connected_rectangles() {
        let width = 8;
        let height = 4;
        let mut output = vec![0.0; width * height * 3];
        for y in 1..3 {
            for x in 1..5 {
                output[y * width + x] = 20.0;
                output[width * height + y * width + x] = 2.0;
                output[2 * width * height + y * width + x] = 2.0;
            }
        }
        let boxes = decode_detection(&output, height, width);
        assert_eq!(boxes.len(), 1);
        assert!(boxes[0].bbox.x0 < 2.0);
        assert!(boxes[0].bbox.x1 > 9.0);
    }
}
