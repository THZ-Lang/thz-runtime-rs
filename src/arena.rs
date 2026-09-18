//! Alocador de Memória Contígua em Arena (ISO/IEC TR 24772 / DoD)
//!
//! Permite alocações em lote ultra-rápidas e descarte total em tempo constante O(1).

use std::alloc::{alloc, dealloc, Layout};

pub struct ThzArena {
    buffer: *mut u8,
    capacidade: usize,
    offset: usize,
    layout: Layout,
}

impl ThzArena {
    pub fn new(bytes: usize) -> Option<Box<Self>> {
        let align = std::mem::align_of::<usize>();
        let layout = Layout::from_size_align(bytes, align).ok()?;
        let buffer = unsafe { alloc(layout) };
        if buffer.is_null() {
            return None;
        }
        Some(Box::new(Self {
            buffer,
            capacidade: bytes,
            offset: 0,
            layout,
        }))
    }

    pub fn alocar(&mut self, bytes: usize, align: usize) -> *mut u8 {
        let current_ptr = unsafe { self.buffer.add(self.offset) as usize };
        let aligned_ptr = (current_ptr + align - 1) & !(align - 1);
        let padding = aligned_ptr - current_ptr;

        if self.offset + padding + bytes > self.capacidade {
            return std::ptr::null_mut();
        }

        self.offset += padding;
        let result = unsafe { self.buffer.add(self.offset) };
        self.offset += bytes;
        result
    }

    pub fn resetar(&mut self) {
        self.offset = 0;
    }
}

impl Drop for ThzArena {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            unsafe {
                dealloc(self.buffer, self.layout);
            }
            self.buffer = std::ptr::null_mut();
        }
    }
}
