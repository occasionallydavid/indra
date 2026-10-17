use image::{GrayImage, Luma};

use crate::image_ops::{resize_f32, resize_gray};

pub const INPUT_MULTIPLE: usize = 32;
pub const OUT_STRIDE: f32 = 2.0;
pub const CORE_X_SHRINK: f32 = 0.15;
pub const LINE_HEIGHT: usize = 32;
pub const PAD_X: f32 = 0.3;
pub const PAD_Y: f32 = 0.2;
pub const WIDTH_MULTIPLE: usize = 8;

#[derive(Clone, Debug)]
pub struct LineCrop {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

pub struct PageInput {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub scale: f32,
}

pub fn page_to_input(gray: &GrayImage, long_side: usize) -> PageInput {
    let (h, w) = (gray.height() as usize, gray.width() as usize);
    let scale = long_side as f32 / h.max(w) as f32;
    let nh = ((h as f32 * scale).round() as usize).max(1);
    let nw = ((w as f32 * scale).round() as usize).max(1);
    let resized = if (nh, nw) != (h, w) {
        resize_gray(gray, nw as u32, nh as u32)
    } else {
        gray.clone()
    };
    let ph = nh.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let pw = nw.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let mut out = vec![0.0; ph * pw];
    for y in 0..nh {
        for x in 0..nw {
            out[y * pw + x] = 1.0 - resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        }
    }
    PageInput {
        data: out,
        width: pw,
        height: ph,
        scale,
    }
}

pub fn crop_line(gray: &GrayImage, bbox: [f32; 4], height: usize, max_width: usize) -> LineCrop {
    let [x0, y0, x1, y1] = bbox;
    let h = (y1 - y0).max(1.0);
    let pad_x0 = x0 - PAD_X * h;
    let pad_x1 = x1 + PAD_X * h;
    let pad_y0 = y0 - PAD_Y * h;
    let pad_y1 = y1 + PAD_Y * h;
    let ix0 = pad_x0.floor() as i32;
    let iy0 = pad_y0.floor() as i32;
    let ix1 = pad_x1.ceil() as i32;
    let iy1 = pad_y1.ceil() as i32;
    let cw = (ix1 - ix0).max(1) as u32;
    let ch = (iy1 - iy0).max(1) as u32;
    let mut crop = GrayImage::from_pixel(cw, ch, Luma([255]));
    let (pw, ph) = (gray.width() as i32, gray.height() as i32);
    let sx0 = ix0.max(0);
    let sy0 = iy0.max(0);
    let sx1 = ix1.min(pw);
    let sy1 = iy1.min(ph);
    if sx1 > sx0 && sy1 > sy0 {
        for sy in sy0..sy1 {
            for sx in sx0..sx1 {
                crop.put_pixel(
                    (sx - ix0) as u32,
                    (sy - iy0) as u32,
                    *gray.get_pixel(sx as u32, sy as u32),
                );
            }
        }
    }
    let out_w =
        ((cw as f32 * height as f32 / ch as f32).round() as usize).clamp(WIDTH_MULTIPLE, max_width);
    let resized = resize_gray(&crop, out_w as u32, height as u32);
    let mut data = vec![0.0; out_w * height];
    for y in 0..height {
        for x in 0..out_w {
            data[y * out_w + x] = 1.0 - resized.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
        }
    }
    LineCrop {
        data,
        width: out_w,
        height,
    }
}

pub struct PaddedBatch {
    pub data: Vec<f32>,
    pub widths: Vec<i64>,
    pub batch: usize,
    pub height: usize,
    pub width: usize,
}

pub fn pad_batch(lines: &[LineCrop], multiple: usize) -> PaddedBatch {
    assert!(!lines.is_empty());
    let height = lines[0].height;
    let width = lines
        .iter()
        .map(|line| line.width)
        .max()
        .unwrap_or(1)
        .div_ceil(multiple)
        * multiple;
    let batch = lines.len();
    let mut data = vec![0.0; batch * height * width];
    let mut widths = Vec::with_capacity(batch);
    for (b, line) in lines.iter().enumerate() {
        widths.push(line.width as i64);
        let offset = b * height * width;
        for y in 0..height {
            data[offset + y * width..offset + y * width + line.width]
                .copy_from_slice(&line.data[y * line.width..(y + 1) * line.width]);
        }
    }
    PaddedBatch {
        data,
        widths,
        batch,
        height,
        width,
    }
}

pub fn width_batches(
    widths: &[usize],
    max_batch: usize,
    max_ratio: f32,
    min_width: usize,
    max_area: usize,
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|&index| widths[index]);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for index in order {
        let fits = batches.last().is_some_and(|batch| {
            batch.len() < max_batch
                && (widths[index].max(min_width) as f32)
                    <= max_ratio * widths[batch[0]].max(min_width) as f32
                && (batch.len() + 1) * widths[index].max(min_width) <= max_area
        });
        if fits {
            batches.last_mut().unwrap().push(index);
        } else {
            batches.push(vec![index]);
        }
    }
    batches
}

pub fn resize_float_for_mask(
    data: &[f32],
    width: usize,
    height: usize,
    nw: usize,
    nh: usize,
) -> Vec<f32> {
    resize_f32(data, width, height, nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_narrow_lines_together() {
        let batches = width_batches(&[100, 104, 120, 300, 310], 64, 1.3, 64, 40960);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 2);
    }
}
