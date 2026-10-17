#[derive(Clone, Debug)]
pub struct GrayFrame {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl GrayFrame {
    pub fn new(width: usize, height: usize, fill: u8) -> Self {
        Self {
            width,
            height,
            data: vec![fill; width * height],
        }
    }

    pub fn from_gray(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height, "grayscale buffer size mismatch");
        Self {
            width,
            height,
            data,
        }
    }

    pub fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Self {
        assert_eq!(rgba.len(), width * height * 4, "RGBA buffer size mismatch");
        let mut data = Vec::with_capacity(width * height);
        for pixel in rgba.as_chunks::<4>().0 {
            let alpha = pixel[3] as f32 / 255.0;
            let luma =
                0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
            data.push((luma * alpha + 255.0 * (1.0 - alpha)).round() as u8);
        }
        Self {
            width,
            height,
            data,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, value: u8) {
        self.data[y * self.width + x] = value;
    }
}
