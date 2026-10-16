//! Motor nativo de Terminal User Interface (TUI) em Rust para THZ-LANG.
//! Fornece renderização direta de caixas, barras de progresso e layouts via C-ABI / LLVM IR.

pub struct ThzNativeTui {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<char>,
}

impl ThzNativeTui {
    pub fn new(width: usize, height: usize) -> Self {
        let w = width.max(10);
        let h = height.max(3);
        Self {
            width: w,
            height: h,
            buffer: vec![' '; w * h],
        }
    }

    pub fn clear(&mut self) {
        self.buffer.fill(' ');
    }

    pub fn set_char(&mut self, x: usize, y: usize, c: char) {
        if x < self.width && y < self.height {
            self.buffer[y * self.width + x] = c;
        }
    }

    pub fn draw_text(&mut self, x: usize, y: usize, text: &str) {
        if y >= self.height {
            return;
        }
        for (i, c) in text.chars().enumerate() {
            if x + i < self.width {
                self.set_char(x + i, y, c);
            }
        }
    }

    pub fn draw_box(&mut self, x: usize, y: usize, w: usize, h: usize, title: Option<&str>) {
        if x + w > self.width || y + h > self.height || w < 2 || h < 2 {
            return;
        }
        let x2 = x + w - 1;
        let y2 = y + h - 1;

        // Cantos Unicode
        self.set_char(x, y, '┌');
        self.set_char(x2, y, '┐');
        self.set_char(x, y2, '└');
        self.set_char(x2, y2, '┘');

        // Bordas horizontais
        for cx in (x + 1)..x2 {
            self.set_char(cx, y, '─');
            self.set_char(cx, y2, '─');
        }

        // Bordas verticais
        for cy in (y + 1)..y2 {
            self.set_char(x, cy, '│');
            self.set_char(x2, cy, '│');
        }

        // Título centralizado
        if let Some(t) = title {
            if !t.is_empty() {
                let formatted = format!(" {} ", t.trim());
                let start = x + (w.saturating_sub(formatted.chars().count()) / 2).max(1);
                self.draw_text(start, y, &formatted);
            }
        }
    }

    pub fn draw_progress(&mut self, x: usize, y: usize, w: usize, pct: f64) {
        if x + w > self.width || y >= self.height || w < 4 {
            return;
        }
        let clamped = pct.clamp(0.0, 1.0);
        let inner_w = w - 2;
        let filled = (inner_w as f64 * clamped).round() as usize;

        self.set_char(x, y, '[');
        for i in 0..inner_w {
            let c = if i < filled { '█' } else { '░' };
            self.set_char(x + 1 + i, y, c);
        }
        self.set_char(x + 1 + inner_w, y, ']');
    }

    pub fn render_to_string(&self) -> String {
        let mut out = String::with_capacity((self.width + 1) * self.height);
        for y in 0..self.height {
            let row_start = y * self.width;
            let row_end = row_start + self.width;
            for &c in &self.buffer[row_start..row_end] {
                out.push(c);
            }
            if y + 1 < self.height {
                out.push('\n');
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_tui_box_and_progress() {
        let mut tui = ThzNativeTui::new(40, 8);
        tui.draw_box(1, 1, 30, 5, Some("STATUS"));
        tui.draw_progress(3, 3, 20, 0.5);

        let rendered = tui.render_to_string();
        assert!(rendered.contains('┌'));
        assert!(rendered.contains('┘'));
        assert!(rendered.contains("STATUS"));
        assert!(rendered.contains('█'));
        assert!(rendered.contains('░'));
    }
}
