pub const BLOCK_CLASSES: [&str; 10] = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
];

pub const N_FEATS: usize = 16;
pub const N_GEOM: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box4 {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Box4 {
    pub fn width(self) -> f32 {
        self.x1 - self.x0
    }

    pub fn height(self) -> f32 {
        self.y1 - self.y0
    }

    pub fn center_x(self) -> f32 {
        (self.x0 + self.x1) / 2.0
    }

    pub fn center_y(self) -> f32 {
        (self.y0 + self.y1) / 2.0
    }

    pub fn as_array(self) -> [f32; 4] {
        [self.x0, self.y0, self.x1, self.y1]
    }
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub n: usize,
    pub cls: Vec<f32>,
    pub same: Vec<f32>,
    pub table: Option<Vec<f32>>,
}

impl Layout {
    pub fn class_probabilities(&self, line: usize) -> &[f32] {
        &self.cls[line * BLOCK_CLASSES.len()..(line + 1) * BLOCK_CLASSES.len()]
    }

    pub fn class_of_block(&self, lines: &[usize]) -> usize {
        let mut sums = [0.0; BLOCK_CLASSES.len()];
        for &line in lines {
            for (sum, value) in sums.iter_mut().zip(self.class_probabilities(line)) {
                *sum += *value;
            }
        }
        argmax(&sums)
    }

    pub fn same_at(&self, a: usize, b: usize) -> f32 {
        self.same[a * self.n + b]
    }
}

pub fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate().skip(1) {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

pub fn line_features(boxes: &[Box4], page_wh: (usize, usize), texts: &[String]) -> Vec<f32> {
    let (page_w, page_h) = (page_wh.0 as f32, page_wh.1 as f32);
    let widths: Vec<f32> = boxes.iter().map(|b| b.width().max(1e-3)).collect();
    let heights: Vec<f32> = boxes.iter().map(|b| b.height().max(1e-3)).collect();
    let med_h = median(&heights);
    let mut out = vec![0.0; boxes.len() * N_FEATS];
    for (i, b) in boxes.iter().enumerate() {
        let values = [
            b.x0 / page_w,
            b.y0 / page_h,
            b.x1 / page_w,
            b.y1 / page_h,
            (b.x0 + b.x1) / (2.0 * page_w),
            (b.y0 + b.y1) / (2.0 * page_h),
            widths[i] / page_w,
            (heights[i] / med_h).ln(),
            (widths[i] / heights[i]).ln() / 4.0,
            med_h / page_h * 20.0,
        ];
        out[i * N_FEATS..i * N_FEATS + N_GEOM].copy_from_slice(&values);
        let text = texts.get(i).map(|text| text.trim()).unwrap_or("");
        if text.is_empty() {
            continue;
        }
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let digits = chars.iter().filter(|ch| ch.is_numeric()).count();
        let mut bullet = "•-–▪◦*·→".contains(chars[0]);
        if !bullet && len > 3 {
            let prefix: String = chars.iter().take(3).collect();
            bullet = prefix
                .trim_end_matches(['.', ')'])
                .chars()
                .all(|ch| ch.is_numeric())
                && prefix.trim_end_matches(['.', ')']).chars().next().is_some();
        }
        let text_values = [
            (len as f32 + 1.0).ln() / 5.0,
            digits as f32 / len as f32,
            f32::from(bullet),
            f32::from(chars.last().is_some_and(|&ch| ch == '-')),
            f32::from(chars.last().is_some_and(|&ch| ".!?:;".contains(ch))),
            f32::from(chars[0].is_uppercase()),
        ];
        out[i * N_FEATS + N_GEOM..(i + 1) * N_FEATS].copy_from_slice(&text_values);
    }
    out
}

fn logaddexp(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    max + ((-(a - b).abs()).exp() + 1.0).ln()
}

pub fn decode_order(matrix: &[f32], n: usize, sweeps: usize) -> Vec<usize> {
    if n <= 1 {
        return (0..n).collect();
    }
    let probability: Vec<f32> = matrix
        .iter()
        .map(|&value| -logaddexp(0.0, -value))
        .collect();
    let mut row_sums = vec![0.0_f32; n];
    let mut col_sums = vec![0.0_f32; n];
    for i in 0..n {
        for j in 0..n {
            row_sums[i] += probability[i * n + j];
            col_sums[j] += probability[i * n + j];
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| (-row_sums[a] + col_sums[a]).total_cmp(&(-row_sums[b] + col_sums[b])));

    for _ in 0..sweeps {
        let mut improved = false;
        let current_order = order.clone();
        for line in current_order {
            let rest: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&value| value != line)
                .collect();
            let mut gains = vec![0.0_f32; rest.len() + 1];
            let mut cumulative = 0.0;
            for (k, &other) in rest.iter().enumerate() {
                cumulative += probability[other * n + line];
                gains[k + 1] = cumulative;
            }
            cumulative = 0.0;
            for k in (0..rest.len()).rev() {
                cumulative += probability[line * n + rest[k]];
                gains[k] += cumulative;
            }
            let best_position = argmax(&gains);
            let current_position = order.iter().position(|&value| value == line).unwrap();
            if gains[best_position] > gains[current_position] + 1e-6 {
                order = rest;
                order.insert(best_position, line);
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    order
}

pub fn order_top_down(boxes: &[Box4]) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let centers: Vec<f32> = boxes.iter().map(|b| b.center_y()).collect();
    let height = median(&boxes.iter().map(|b| b.height()).collect::<Vec<_>>());
    let mut out: Vec<usize> = (0..boxes.len()).collect();
    out.sort_by(|&a, &b| {
        let row_a = (centers[a] / (height * 0.5).max(1e-3)).round();
        let row_b = (centers[b] / (height * 0.5).max(1e-3)).round();
        row_a
            .total_cmp(&row_b)
            .then_with(|| boxes[a].x0.total_cmp(&boxes[b].x0))
    });
    out
}

fn best_cut(boxes: &[Box4], axis_y: bool) -> (f32, Option<usize>, Vec<usize>) {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        let lo_a = if axis_y { boxes[a].y0 } else { boxes[a].x0 };
        let lo_b = if axis_y { boxes[b].y0 } else { boxes[b].x0 };
        lo_a.total_cmp(&lo_b)
    });
    let hi = |index: usize| {
        if axis_y {
            boxes[index].y1
        } else {
            boxes[index].x1
        }
    };
    let lo = |index: usize| {
        if axis_y {
            boxes[index].y0
        } else {
            boxes[index].x0
        }
    };
    let mut end = hi(order[0]);
    let mut best_gap = 0.0_f32;
    let mut best_k = None;
    for (k, &item) in order.iter().enumerate().skip(1) {
        let gap = lo(item) - end;
        if gap > best_gap {
            best_gap = gap;
            best_k = Some(k);
        }
        end = end.max(hi(item));
    }
    (best_gap, best_k, order)
}

pub fn order_xycut(boxes: &[Box4]) -> Vec<usize> {
    fn recurse(boxes: &[Box4], ids: Vec<usize>) -> Vec<usize> {
        if ids.len() <= 1 {
            return ids;
        }
        let local: Vec<Box4> = ids.iter().map(|&id| boxes[id]).collect();
        let (g_y, k_y, o_y) = best_cut(&local, true);
        let (g_x, k_x, o_x) = best_cut(&local, false);
        let Some(k) = k_y.or(k_x) else {
            let local_order = order_top_down(&local);
            return local_order.into_iter().map(|index| ids[index]).collect();
        };
        let order = if g_y >= g_x { o_y } else { o_x };
        let first: Vec<usize> = order[..k].iter().map(|&index| ids[index]).collect();
        let second: Vec<usize> = order[k..].iter().map(|&index| ids[index]).collect();
        let mut out = recurse(boxes, first);
        out.extend(recurse(boxes, second));
        out
    }
    recurse(boxes, (0..boxes.len()).collect())
}

pub fn cluster(logits: &[f32], n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }
    let mut sums = logits.to_vec();
    for i in 0..n {
        sums[i * n + i] = 0.0;
    }
    let mut size = vec![1.0_f32; n];
    let mut alive = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut means = sums.clone();
    for i in 0..n {
        means[i * n + i] = f32::NEG_INFINITY;
    }
    let mut best: Vec<usize> = (0..n)
        .map(|i| {
            (0..n)
                .max_by(|&a, &b| means[i * n + a].total_cmp(&means[i * n + b]))
                .unwrap()
        })
        .collect();
    let mut best_value: Vec<f32> = (0..n).map(|i| means[i * n + best[i]]).collect();

    for _ in 0..n.saturating_sub(1) {
        let a = argmax(&best_value);
        let b = best[a];
        if best_value[a] <= 0.0 {
            break;
        }
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        for j in 0..n {
            sums[a * n + j] += sums[b * n + j];
            sums[j * n + a] += sums[j * n + b];
        }
        sums[a * n + a] = 0.0;
        size[a] += size[b];
        alive[b] = false;
        let tail = members[b].clone();
        members[a].extend(tail);

        let mut row = vec![f32::NEG_INFINITY; n];
        for j in 0..n {
            if alive[j] && j != a {
                row[j] = sums[a * n + j] / (size[a] * size[j]);
            }
        }
        for j in 0..n {
            means[a * n + j] = row[j];
            means[j * n + a] = row[j];
            means[b * n + j] = f32::NEG_INFINITY;
            means[j * n + b] = f32::NEG_INFINITY;
        }
        best_value[b] = f32::NEG_INFINITY;
        best[a] = argmax(&row);
        best_value[a] = row[best[a]];

        for j in 0..n {
            if alive[j] && j != a && (best[j] == a || best[j] == b) {
                best[j] = argmax(&means[j * n..(j + 1) * n]);
                best_value[j] = means[j * n + best[j]];
            }
        }
        for j in 0..n {
            if alive[j] && j != a && row[j] > best_value[j] {
                best[j] = a;
                best_value[j] = row[j];
            }
        }
    }

    let mut out: Vec<Vec<usize>> = (0..n)
        .filter(|&i| alive[i])
        .map(|i| {
            members[i].sort_unstable();
            std::mem::take(&mut members[i])
        })
        .collect();
    out.sort_by_key(|block| block.first().copied().unwrap_or(usize::MAX));
    out
}

pub fn decode_blocks_order(matrix: &[f32], layout: &Layout, boxes: &[Box4]) -> Vec<usize> {
    let n = boxes.len();
    if n <= 1 {
        return (0..n).collect();
    }
    let blocks = cluster(&layout.same, n);
    let grid_classes = [5_usize, 8, 9];
    let mut inner: Vec<Vec<usize>> = Vec::with_capacity(blocks.len());
    for block in &blocks {
        let local_boxes: Vec<Box4> = block.iter().map(|&index| boxes[index]).collect();
        let class = layout.class_of_block(block);
        let local_order = if grid_classes.contains(&class) {
            order_top_down(&local_boxes)
        } else {
            order_xycut(&local_boxes)
        };
        inner.push(local_order.into_iter().map(|index| block[index]).collect());
    }

    let m = blocks.len();
    let mut means = vec![0.0_f32; m * m];
    for a in 0..m {
        for b in a + 1..m {
            let mut sum = 0.0_f64;
            for &i in &blocks[a] {
                for &j in &blocks[b] {
                    sum += matrix[i * n + j] as f64;
                }
            }
            means[a * m + b] = (sum / (blocks[a].len() * blocks[b].len()) as f64) as f32;
        }
    }
    let mut block_logits = vec![0.0_f32; m * m];
    for a in 0..m {
        for b in a + 1..m {
            let value = means[a * m + b] - means[b * m + a];
            block_logits[a * m + b] = value;
            block_logits[b * m + a] = -value;
        }
    }
    let block_order = decode_order(&block_logits, m, 4);
    block_order
        .into_iter()
        .flat_map(|index| inner[index].iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xycut_reads_columns_before_rows() {
        let boxes = [
            Box4 {
                x0: 0.0,
                y0: 0.0,
                x1: 10.0,
                y1: 8.0,
            },
            Box4 {
                x0: 0.0,
                y0: 20.0,
                x1: 10.0,
                y1: 28.0,
            },
            Box4 {
                x0: 100.0,
                y0: 0.0,
                x1: 110.0,
                y1: 8.0,
            },
            Box4 {
                x0: 100.0,
                y0: 20.0,
                x1: 110.0,
                y1: 28.0,
            },
        ];
        assert_eq!(order_xycut(&boxes), vec![0, 1, 2, 3]);
    }

    #[test]
    fn order_decoder_obeys_pairwise_logits() {
        let matrix = [0.0, 2.0, 2.0, -2.0, 0.0, 2.0, -2.0, -2.0, 0.0];
        assert_eq!(decode_order(&matrix, 3, 4), vec![0, 1, 2]);
    }
}
