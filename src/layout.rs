use crate::order::{Box4, Layout, argmax, median};

pub const HEADER: usize = 2;
pub const FOOTER: usize = 3;
pub const SECTION: usize = 1;
pub const LIST: usize = 4;
pub const TABLE: usize = 5;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Block {
    pub label: String,
    pub bbox: [f32; 4],
    pub text: String,
}

fn is_cjk(ch: char) -> bool {
    let value = ch as u32;
    (0x3000..=0x30ff).contains(&value)
        || (0x3400..=0x9fff).contains(&value)
        || (0xf900..=0xfaff).contains(&value)
        || (0xff00..=0xffef).contains(&value)
        || (0x20000..=0x2fa1f).contains(&value)
}

pub fn join_lines(texts: &[String]) -> String {
    let mut out = String::new();
    for text in texts {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        let previous = out.chars().last();
        let next = text.chars().next();
        if out.chars().count() > 1
            && out.ends_with('-')
            && out.chars().rev().nth(1).is_some_and(char::is_alphabetic)
            && next.is_some_and(char::is_lowercase)
        {
            out.pop();
            out.push_str(text);
        } else if previous.is_some_and(is_cjk) && next.is_some_and(is_cjk) {
            out.push_str(text);
        } else {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(text);
        }
    }
    out
}

pub fn split_blocks(order: &[usize], layout: &Layout) -> Vec<Vec<usize>> {
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for &line in order {
        let joins = blocks
            .last()
            .is_some_and(|block| layout.same_at(*block.last().unwrap(), line) > 0.0);
        if joins {
            blocks.last_mut().unwrap().push(line);
        } else {
            blocks.push(vec![line]);
        }
    }
    blocks
}

fn rows(boxes: &[Box4]) -> Vec<Vec<usize>> {
    let centers: Vec<f32> = boxes.iter().map(|b| b.center_y()).collect();
    let height = median(&boxes.iter().map(|b| b.height()).collect::<Vec<_>>());
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| centers[a].total_cmp(&centers[b]));
    let mut out: Vec<Vec<usize>> = Vec::new();
    for index in order {
        if out.last().is_some_and(|row| {
            let mean = row.iter().map(|&i| centers[i]).sum::<f32>() / row.len() as f32;
            (centers[index] - mean).abs() < 0.5 * height
        }) {
            out.last_mut().unwrap().push(index);
        } else {
            out.push(vec![index]);
        }
    }
    for row in &mut out {
        row.sort_by(|&a, &b| boxes[a].x0.total_cmp(&boxes[b].x0));
    }
    out
}

fn merge_wrapped(rows: Vec<Vec<usize>>, boxes: &[Box4]) -> Vec<Vec<usize>> {
    if rows.len() < 3 {
        return rows;
    }
    let height = median(&boxes.iter().map(|b| b.height()).collect::<Vec<_>>());
    let gaps: Vec<f32> = rows
        .windows(2)
        .map(|pair| {
            let current_bottom = pair[0]
                .iter()
                .map(|&index| boxes[index].y1)
                .fold(f32::NEG_INFINITY, f32::max);
            let next_top = pair[1]
                .iter()
                .map(|&index| boxes[index].y0)
                .fold(f32::INFINITY, f32::min);
            next_top - current_bottom
        })
        .collect();
    let median_gap = median(&gaps);
    let mut out = vec![rows[0].clone()];
    let mut previous = &rows[0];
    for (gap, row) in gaps.iter().zip(rows.iter().skip(1)) {
        if *gap < 0.5 * median_gap && *gap < 0.6 * height && row.len() < previous.len() {
            out.last_mut().unwrap().extend(row.iter().copied());
        } else {
            out.push(row.clone());
        }
        previous = row;
    }
    out
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn table_block(table: &[f32], n: usize, block: &[usize]) -> Vec<f32> {
    let size = block.len();
    let mut out = vec![0.0; 3 * size * size];
    for k in 0..3 {
        for (a, &i) in block.iter().enumerate() {
            for (b, &j) in block.iter().enumerate() {
                out[k * size * size + a * size + b] = table[k * n * n + i * n + j];
            }
        }
    }
    out
}

fn heads_plausible(boxes: &[Box4], table: &[f32]) -> bool {
    let n = boxes.len();
    let cell_logits = &table[2 * n * n..3 * n * n];
    let cells = crate::order::cluster(cell_logits, n);
    let mut bad = 0;
    for cell in &cells {
        if cell.len() < 2 {
            continue;
        }
        let mut overlaps = true;
        for &a in cell {
            for &b in cell {
                if boxes[a].x1.min(boxes[b].x1) - boxes[a].x0.max(boxes[b].x0) < 0.0 {
                    overlaps = false;
                }
            }
        }
        if !overlaps {
            bad += 1;
        }
    }
    bad as f32 <= 0.1 * cells.len() as f32
}

fn cluster_table_logits(
    table: &[f32],
    n: usize,
    k: usize,
    cells: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let m = cells.len();
    let mut logits = vec![0.0_f32; m * m];
    for a in 0..m {
        for b in a + 1..m {
            let mut sum = 0.0_f64;
            for &i in &cells[a] {
                for &j in &cells[b] {
                    sum += table[k * n * n + i * n + j] as f64;
                }
            }
            let value = (sum / (cells[a].len() * cells[b].len()) as f64) as f32;
            logits[a * m + b] = value;
            logits[b * m + a] = value;
        }
    }
    crate::order::cluster(&logits, m)
}

fn table_html_heads(boxes: &[Box4], texts: &[String], table: &[f32]) -> String {
    let n = boxes.len();
    let cells = crate::order::cluster(&table[2 * n * n..3 * n * n], n);
    let cell_boxes: Vec<Box4> = cells
        .iter()
        .map(|cell| Box4 {
            x0: cell
                .iter()
                .map(|&i| boxes[i].x0)
                .fold(f32::INFINITY, f32::min),
            y0: cell
                .iter()
                .map(|&i| boxes[i].y0)
                .fold(f32::INFINITY, f32::min),
            x1: cell
                .iter()
                .map(|&i| boxes[i].x1)
                .fold(f32::NEG_INFINITY, f32::max),
            y1: cell
                .iter()
                .map(|&i| boxes[i].y1)
                .fold(f32::NEG_INFINITY, f32::max),
        })
        .collect();
    let cell_texts: Vec<String> = cells
        .iter()
        .map(|cell| {
            let mut ordered = cell.clone();
            ordered.sort_by(|&a, &b| {
                boxes[a]
                    .y0
                    .total_cmp(&boxes[b].y0)
                    .then_with(|| boxes[a].x0.total_cmp(&boxes[b].x0))
            });
            ordered
                .iter()
                .map(|&i| texts[i].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let m = cells.len();
    let mut rows = cluster_table_logits(table, n, 0, &cells);
    rows.sort_by(|a, b| {
        let ya = a
            .iter()
            .map(|&i| cell_boxes[i].y0)
            .fold(f32::INFINITY, f32::min);
        let yb = b
            .iter()
            .map(|&i| cell_boxes[i].y0)
            .fold(f32::INFINITY, f32::min);
        ya.total_cmp(&yb)
    });
    let mut columns = cluster_table_logits(table, n, 1, &cells);
    columns.sort_by(|a, b| {
        let xa = a
            .iter()
            .map(|&i| cell_boxes[i].x0)
            .fold(f32::INFINITY, f32::min);
        let xb = b
            .iter()
            .map(|&i| cell_boxes[i].x0)
            .fold(f32::INFINITY, f32::min);
        xa.total_cmp(&xb)
    });
    let mut row_of = vec![0_usize; m];
    let mut column_of = vec![0_usize; m];
    for (row, group) in rows.iter().enumerate() {
        for &cell in group {
            row_of[cell] = row;
        }
    }
    for (column, group) in columns.iter().enumerate() {
        for &cell in group {
            column_of[cell] = column;
        }
    }
    let centers: Vec<f32> = columns
        .iter()
        .map(|group| {
            median(
                &group
                    .iter()
                    .map(|&cell| cell_boxes[cell].center_x())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let height = median(&boxes.iter().map(|b| b.height()).collect::<Vec<_>>());
    let mut grid: std::collections::BTreeMap<(usize, usize), (usize, String)> =
        std::collections::BTreeMap::new();
    for cell in 0..m {
        let first_column = column_of[cell];
        let mut span = 1;
        while first_column + span < columns.len()
            && centers[first_column + span] >= cell_boxes[cell].x0 - height
            && centers[first_column + span] <= cell_boxes[cell].x1 + height
        {
            span += 1;
        }
        let key = (row_of[cell], first_column);
        if let Some((old_span, old_text)) = grid.get_mut(&key) {
            *old_span = (*old_span).max(span);
            old_text.push(' ');
            old_text.push_str(&cell_texts[cell]);
        } else {
            grid.insert(key, (span, cell_texts[cell].clone()));
        }
    }

    let mut out = vec!["<table>".to_string()];
    for row in 0..rows.len() {
        let tag = if row == 0 && rows.len() > 1 {
            "th"
        } else {
            "td"
        };
        let mut tds = Vec::new();
        let mut column = 0;
        while column < columns.len() {
            if let Some((span, text)) = grid.get(&(row, column)) {
                let attr = if *span > 1 {
                    format!(r#" colspan="{}""#, span)
                } else {
                    String::new()
                };
                tds.push(format!("<{tag}{attr}>{}</{tag}>", escape_html(text)));
                column += *span;
            } else {
                tds.push(format!("<{tag}></{tag}>"));
                column += 1;
            }
        }
        out.push(format!("<tr>{}</tr>", tds.concat()));
    }
    out.push("</table>".into());
    out.join("\n")
}

pub fn table_html(boxes: &[Box4], texts: &[String]) -> String {
    let mut rows = rows(boxes);
    let span_x = boxes.iter().map(|b| b.x1).fold(f32::NEG_INFINITY, f32::max)
        - boxes.iter().map(|b| b.x0).fold(f32::INFINITY, f32::min);
    if rows.len() > 2
        && rows[0].len() == 1
        && rows.iter().map(Vec::len).max().unwrap_or(0) > 1
        && boxes[rows[0][0]].width() > 0.6 * span_x
    {
        let title_index = rows[0][0];
        let keep: Vec<usize> = (0..texts.len()).filter(|&i| i != title_index).collect();
        let kept_boxes: Vec<Box4> = keep.iter().map(|&i| boxes[i]).collect();
        let kept_texts: Vec<String> = keep.iter().map(|&i| texts[i].clone()).collect();
        return format!(
            "{}\n\n{}",
            texts[title_index],
            table_html(&kept_boxes, &kept_texts)
        );
    }
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return String::new();
    }
    let full: Vec<Vec<usize>> = rows
        .iter()
        .filter(|row| row.len() == width)
        .cloned()
        .collect();
    rows = merge_wrapped(rows, boxes);
    let mut bands = vec![(0.0_f32, 0.0_f32); width];
    for column in 0..width {
        let lefts: Vec<f32> = full.iter().map(|row| boxes[row[column]].x0).collect();
        let rights: Vec<f32> = full.iter().map(|row| boxes[row[column]].x1).collect();
        bands[column] = (median(&lefts), median(&rights));
    }
    let mut edges = Vec::with_capacity(width + 1);
    edges.push(f32::NEG_INFINITY);
    for column in 0..width.saturating_sub(1) {
        edges.push((bands[column].1 + bands[column + 1].0) / 2.0);
    }
    edges.push(f32::INFINITY);

    let mut out = vec!["<table>".to_string()];
    for (row_index, row) in rows.iter().enumerate() {
        let tag = if row_index == 0 && rows.len() > 1 {
            "th"
        } else {
            "td"
        };
        let mut ordered = row.clone();
        ordered.sort_by(|&a, &b| {
            boxes[a]
                .y0
                .total_cmp(&boxes[b].y0)
                .then_with(|| boxes[a].x0.total_cmp(&boxes[b].x0))
        });
        let mut cells: Vec<Option<(usize, String)>> = vec![None; width];
        for index in ordered {
            let b = boxes[index];
            let first = edges
                .partition_point(|&edge| edge <= b.x0)
                .saturating_sub(1);
            let last = edges
                .partition_point(|&edge| edge <= b.x1)
                .saturating_sub(1);
            let begin = first.min(width - 1);
            let end = last.min(width - 1);
            let mut span: Vec<usize> = (begin..=end)
                .filter(|&column| {
                    b.x1.min(bands[column].1) - b.x0.max(bands[column].0)
                        > 0.5 * (bands[column].1 - bands[column].0)
                })
                .collect();
            if span.is_empty() {
                span.push(begin);
            }
            let c0 = span[0];
            if let Some((count, text)) = &mut cells[c0] {
                *count = (*count).max(span.len());
                text.push(' ');
                text.push_str(&texts[index]);
            } else {
                cells[c0] = Some((span.len(), texts[index].clone()));
            }
        }
        let mut tds = Vec::new();
        let mut column = 0;
        while column < width {
            if let Some((span, text)) = &cells[column] {
                let attr = if *span > 1 {
                    format!(r#" colspan="{}""#, span)
                } else {
                    String::new()
                };
                tds.push(format!("<{tag}{attr}>{}</{tag}>", escape_html(text)));
                column += *span;
            } else {
                tds.push(format!("<{tag}></{tag}>"));
                column += 1;
            }
        }
        out.push(format!("<tr>{}</tr>", tds.concat()));
    }
    out.push("</table>".into());
    out.join("\n")
}

fn is_list_item(text: &str) -> bool {
    let text = text.trim_start();
    let Some(first) = text.chars().next() else {
        return false;
    };
    if "•-–▪◦*·→".contains(first) {
        return true;
    }
    let mut chars = text.chars().peekable();
    if chars.peek() == Some(&'(') {
        chars.next();
    }
    let mut count = 0;
    while chars.peek().is_some_and(|ch| ch.is_numeric()) && count < 3 {
        chars.next();
        count += 1;
    }
    if count > 0 && chars.peek().is_some_and(|ch| *ch == '.' || *ch == ')') {
        chars.next();
        return chars.peek().is_none_or(|ch| ch.is_whitespace());
    }
    let text_chars: Vec<char> = text.chars().collect();
    if text_chars.len() >= 2
        && text_chars[0].is_alphabetic()
        && matches!(text_chars[1], '.' | ')')
        && text_chars.get(2).is_none_or(|ch| ch.is_whitespace())
    {
        return true;
    }
    false
}

fn is_whole_math(text: &str) -> bool {
    let text = text.trim();
    text.len() >= 3
        && text.starts_with('$')
        && text.ends_with('$')
        && !text[1..text.len() - 1].contains('$')
}

pub fn page_blocks(
    boxes: &[Box4],
    texts: &[String],
    order: &[usize],
    layout: &Layout,
) -> Vec<Block> {
    split_blocks(order, layout)
        .into_iter()
        .map(|block| {
            let class = layout.class_of_block(&block);
            let x0 = block
                .iter()
                .map(|&i| boxes[i].x0)
                .fold(f32::INFINITY, f32::min);
            let y0 = block
                .iter()
                .map(|&i| boxes[i].y0)
                .fold(f32::INFINITY, f32::min);
            let x1 = block
                .iter()
                .map(|&i| boxes[i].x1)
                .fold(f32::NEG_INFINITY, f32::max);
            let y1 = block
                .iter()
                .map(|&i| boxes[i].y1)
                .fold(f32::NEG_INFINITY, f32::max);
            Block {
                label: crate::order::BLOCK_CLASSES[class].to_string(),
                bbox: [round1(x0), round1(y0), round1(x1), round1(y1)],
                text: block
                    .iter()
                    .map(|&i| texts[i].as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            }
        })
        .collect()
}

fn round1(value: f32) -> f32 {
    (value * 10.0).round() / 10.0
}

pub fn page_markdown(
    boxes: &[Box4],
    texts: &[String],
    order: &[usize],
    layout: &Layout,
    keep_furniture: bool,
    cells: &[Vec<(Box4, String)>],
    table_heads: Option<&[f32]>,
) -> String {
    let mut parts = Vec::new();
    for block in split_blocks(order, layout) {
        let class = layout.class_of_block(&block);
        if (class == HEADER || class == FOOTER) && !keep_furniture {
            continue;
        }
        let block_texts: Vec<String> = block.iter().map(|&i| texts[i].clone()).collect();
        if block_texts.iter().all(|text| is_whole_math(text)) {
            parts.push(
                block_texts
                    .iter()
                    .map(|text| {
                        let trimmed = text.trim();
                        format!("$${}$$", &trimmed[1..trimmed.len() - 1])
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            continue;
        }
        if class == SECTION {
            parts.push(format!("## {}", join_lines(&block_texts)));
        } else if class == TABLE && block.len() > 1 {
            if let Some(table_heads) = table_heads {
                let block_heads = table_block(table_heads, boxes.len(), &block);
                let block_boxes: Vec<Box4> = block.iter().map(|&line| boxes[line]).collect();
                if heads_plausible(&block_boxes, &block_heads) {
                    parts.push(table_html_heads(&block_boxes, &block_texts, &block_heads));
                    continue;
                }
            }
            let mut cell_boxes = Vec::new();
            let mut cell_texts = Vec::new();
            for &line in &block {
                if let Some(parts) = cells.get(line).filter(|parts| !parts.is_empty()) {
                    for (bbox, text) in parts {
                        cell_boxes.push(*bbox);
                        cell_texts.push(text.clone());
                    }
                } else {
                    cell_boxes.push(boxes[line]);
                    cell_texts.push(texts[line].clone());
                }
            }
            parts.push(table_html(&cell_boxes, &cell_texts));
        } else if class == LIST {
            let mut items: Vec<Vec<String>> = Vec::new();
            for text in block_texts {
                if items.is_empty() || is_list_item(&text) {
                    items.push(vec![text]);
                } else {
                    items.last_mut().unwrap().push(text);
                }
            }
            parts.push(
                items
                    .iter()
                    .map(|item| join_lines(item))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        } else {
            parts.push(join_lines(&block_texts));
        }
    }
    parts.retain(|part| !part.is_empty());
    parts.join("\n\n")
}

pub fn best_class(class_sums: &[f32]) -> usize {
    argmax(class_sums)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_hyphenated_lines() {
        assert_eq!(
            join_lines(&["tumori-".into(), "genesis".into()]),
            "tumorigenesis"
        );
    }

    #[test]
    fn joins_cjk_without_space() {
        assert_eq!(join_lines(&["你好".into(), "世界".into()]), "你好世界");
    }

    #[test]
    fn builds_simple_table() {
        let boxes = [
            Box4 {
                x0: 0.0,
                y0: 0.0,
                x1: 10.0,
                y1: 8.0,
            },
            Box4 {
                x0: 20.0,
                y0: 0.0,
                x1: 30.0,
                y1: 8.0,
            },
            Box4 {
                x0: 0.0,
                y0: 20.0,
                x1: 10.0,
                y1: 28.0,
            },
            Box4 {
                x0: 20.0,
                y0: 20.0,
                x1: 30.0,
                y1: 28.0,
            },
        ];
        let html = table_html(&boxes, &["A".into(), "B".into(), "1".into(), "2".into()]);
        assert!(html.contains("<th>A</th>"));
        assert!(html.contains("<td>2</td>"));
    }
}
