use wasm_bindgen::prelude::*;

use crate::frame::GrayFrame;
use crate::pipeline::Ocr;

#[wasm_bindgen]
pub struct WebOcr {
    inner: Ocr,
}

#[wasm_bindgen]
impl WebOcr {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WebOcr, JsValue> {
        console_error_panic_hook::set_once();
        let inner = Ocr::new().map_err(js_error)?;
        Ok(Self { inner })
    }

    pub fn set_batch(&mut self, batch: usize) {
        self.inner.set_batch(batch);
    }

    /// Run OCR on RGBA pixels decoded by the browser. Returns the page as JSON.
    pub fn ocr_rgba(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        run_orientation: bool,
    ) -> Result<String, JsValue> {
        if width == 0 || height == 0 {
            return Err(JsValue::from_str("image dimensions must be non-zero"));
        }
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| JsValue::from_str("image dimensions overflow"))?;
        if rgba.len() != expected {
            return Err(JsValue::from_str(&format!(
                "RGBA buffer has {} bytes; expected {expected}",
                rgba.len()
            )));
        }
        let frame = GrayFrame::from_rgba(width, height, rgba);
        let page = self
            .inner
            .ocr_image(&frame, run_orientation)
            .map_err(js_error)?;
        serde_json::to_string(&serde_json::json!({
            "source": page.source,
            "rotation": page.rotation,
            "skew": page.skew,
            "note": page.note,
            "lines": page.lines,
            "markdown": page.markdown,
            "blocks": page.blocks,
            "timings": page.timings,
        }))
        .map_err(js_error)
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{error:#}"))
}
