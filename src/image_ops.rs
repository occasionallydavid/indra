use crate::frame::GrayFrame;

pub fn resize_gray(src: &GrayFrame, width: usize, height: usize) -> GrayFrame {
    let width = width.max(1);
    let height = height.max(1);
    if (src.width(), src.height()) == (width, height) {
        return src.clone();
    }
    let (sw, sh) = (src.width(), src.height());
    let downscale = width < sw || height < sh;
    let mut out = GrayFrame::new(width, height, 0);

    if downscale {
        // OpenCV's INTER_AREA equivalent: average the source area covered by
        // each destination pixel.
        let sx = sw as f64 / width as f64;
        let sy = sh as f64 / height as f64;
        for y in 0..height {
            let y0 = y as f64 * sy;
            let y1 = (y + 1) as f64 * sy;
            let iy0 = y0.floor() as usize;
            let iy1 = y1.ceil().min(sh as f64) as usize;
            for x in 0..width {
                let x0 = x as f64 * sx;
                let x1 = (x + 1) as f64 * sx;
                let ix0 = x0.floor() as usize;
                let ix1 = x1.ceil().min(sw as f64) as usize;
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                for sy_i in iy0..iy1 {
                    let wy = ((sy_i + 1) as f64).min(y1) - (sy_i as f64).max(y0);
                    for sx_i in ix0..ix1 {
                        let wx = ((sx_i + 1) as f64).min(x1) - (sx_i as f64).max(x0);
                        let weight = wx * wy;
                        sum += src.get(sx_i, sy_i) as f64 * weight;
                        weight_sum += weight;
                    }
                }
                out.set(x, y, (sum / weight_sum.max(f64::EPSILON)).round() as u8);
            }
        }
        return out;
    }

    // Bilinear interpolation with half-pixel centres, matching INTER_LINEAR.
    for y in 0..height {
        let fy = ((y as f64 + 0.5) * sh as f64 / height as f64 - 0.5).max(0.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - fy.floor();
        for x in 0..width {
            let fx = ((x as f64 + 0.5) * sw as f64 / width as f64 - 0.5).max(0.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - fx.floor();
            let p00 = src.get(x0, y0) as f64;
            let p10 = src.get(x1, y0) as f64;
            let p01 = src.get(x0, y1) as f64;
            let p11 = src.get(x1, y1) as f64;
            let value = p00 * (1.0 - wx) * (1.0 - wy)
                + p10 * wx * (1.0 - wy)
                + p01 * (1.0 - wx) * wy
                + p11 * wx * wy;
            out.set(x, y, value.round() as u8);
        }
    }
    out
}

pub fn resize_f32(src: &[f32], width: usize, height: usize, nw: usize, nh: usize) -> Vec<f32> {
    if width == nw && height == nh {
        return src.to_vec();
    }
    let downscale = nw < width || nh < height;
    let mut out = vec![0.0; nw * nh];
    if downscale {
        let sx = width as f64 / nw as f64;
        let sy = height as f64 / nh as f64;
        for y in 0..nh {
            let y0 = y as f64 * sy;
            let y1 = (y + 1) as f64 * sy;
            for x in 0..nw {
                let x0 = x as f64 * sx;
                let x1 = (x + 1) as f64 * sx;
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                for sy_i in y0.floor() as usize..(y1.ceil() as usize).min(height) {
                    let wy = ((sy_i + 1) as f64).min(y1) - (sy_i as f64).max(y0);
                    for sx_i in x0.floor() as usize..(x1.ceil() as usize).min(width) {
                        let wx = ((sx_i + 1) as f64).min(x1) - (sx_i as f64).max(x0);
                        let weight = wx * wy;
                        sum += src[sy_i * width + sx_i] as f64 * weight;
                        weight_sum += weight;
                    }
                }
                out[y * nw + x] = (sum / weight_sum.max(f64::EPSILON)) as f32;
            }
        }
        return out;
    }
    for y in 0..nh {
        let fy = ((y as f64 + 0.5) * height as f64 / nh as f64 - 0.5).max(0.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(height - 1);
        let wy = fy - fy.floor();
        for x in 0..nw {
            let fx = ((x as f64 + 0.5) * width as f64 / nw as f64 - 0.5).max(0.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let wx = fx - fx.floor();
            out[y * nw + x] = (src[y0 * width + x0] as f64 * (1.0 - wx) * (1.0 - wy)
                + src[y0 * width + x1] as f64 * wx * (1.0 - wy)
                + src[y1 * width + x0] as f64 * (1.0 - wx) * wy
                + src[y1 * width + x1] as f64 * wx * wy) as f32;
        }
    }
    out
}

pub fn rotate_gray_replicate(src: &GrayFrame, degrees: f32) -> GrayFrame {
    let (w, h) = (src.width(), src.height());
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    let radians = (degrees as f64).to_radians();
    let (sin, cos) = radians.sin_cos();
    let mut out = GrayFrame::new(w, h, 0);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 + 0.5 - cx;
            let dy = y as f64 + 0.5 - cy;
            let sx = cos * dx + sin * dy + cx - 0.5;
            let sy = -sin * dx + cos * dy + cy - 0.5;
            out.set(x, y, sample_bilinear_replicate(src, sx, sy));
        }
    }
    out
}

pub fn rotate_f32_constant(src: &[f32], width: usize, height: usize, degrees: f32) -> Vec<f32> {
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let radians = (degrees as f64).to_radians();
    let (sin, cos) = radians.sin_cos();
    let mut out = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 + 0.5 - cx;
            let dy = y as f64 + 0.5 - cy;
            let sx = cos * dx + sin * dy + cx - 0.5;
            let sy = -sin * dx + cos * dy + cy - 0.5;
            out[y * width + x] = sample_f32_constant(src, width, height, sx, sy);
        }
    }
    out
}

fn sample_bilinear_replicate(src: &GrayFrame, x: f64, y: f64) -> u8 {
    let (w, h) = (src.width() as i64, src.height() as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = (x0 + 1).clamp(0, w - 1);
    let y1 = (y0 + 1).clamp(0, h - 1);
    let x0 = x0.clamp(0, w - 1) as usize;
    let y0 = y0.clamp(0, h - 1) as usize;
    let x1 = x1 as usize;
    let y1 = y1 as usize;
    let wx = x - x.floor();
    let wy = y - y.floor();
    let value = src.get(x0, y0) as f64 * (1.0 - wx) * (1.0 - wy)
        + src.get(x1, y0) as f64 * wx * (1.0 - wy)
        + src.get(x0, y1) as f64 * (1.0 - wx) * wy
        + src.get(x1, y1) as f64 * wx * wy;
    value.round().clamp(0.0, 255.0) as u8
}

fn sample_f32_constant(src: &[f32], width: usize, height: usize, x: f64, y: f64) -> f32 {
    if x < 0.0 || y < 0.0 || x >= width as f64 || y >= height as f64 {
        return 0.0;
    }
    let x0 = (x.floor() as usize).min(width - 1);
    let y0 = (y.floor() as usize).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let wx = x - x.floor();
    let wy = y - y.floor();
    (src[y0 * width + x0] as f64 * (1.0 - wx) * (1.0 - wy)
        + src[y0 * width + x1] as f64 * wx * (1.0 - wy)
        + src[y1 * width + x0] as f64 * (1.0 - wx) * wy
        + src[y1 * width + x1] as f64 * wx * wy) as f32
}

pub fn rotate_quarter_turns_clockwise(image: GrayFrame, turns: usize) -> GrayFrame {
    let (w, h) = (image.width(), image.height());
    match turns % 4 {
        1 => {
            let mut out = GrayFrame::new(h, w, 0);
            for y in 0..h {
                for x in 0..w {
                    out.set(h - 1 - y, x, image.get(x, y));
                }
            }
            out
        }
        2 => {
            let mut out = GrayFrame::new(w, h, 0);
            for y in 0..h {
                for x in 0..w {
                    out.set(w - 1 - x, h - 1 - y, image.get(x, y));
                }
            }
            out
        }
        3 => {
            let mut out = GrayFrame::new(h, w, 0);
            for y in 0..h {
                for x in 0..w {
                    out.set(y, w - 1 - x, image.get(x, y));
                }
            }
            out
        }
        _ => image,
    }
}
