//! WebAssembly (WASM) Bridge para THZ-LANG Engine
//!
//! Permite execução do motor de regras de negócio, aritmética exata,
//! embeddings semânticos e validações fiscais diretamente no navegador ou Edge Workers.

use crate::ml::ThzEmbeddingEngine;
use crate::simd_math;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Retorna a versão oficial do motor WASM
#[no_mangle]
pub extern "C" fn thz_wasm_versao() -> *mut c_char {
    CString::new("0.4.0-WASM").unwrap().into_raw()
}

/// Calcula similaridade de cosseno diretamente no ambiente WASM
#[no_mangle]
pub extern "C" fn thz_wasm_similaridade_cosseno(a_ptr: *const f32, b_ptr: *const f32, len: usize) -> f32 {
    if a_ptr.is_null() || b_ptr.is_null() || len == 0 {
        return 0.0;
    }
    let a = unsafe { std::slice::from_raw_parts(a_ptr, len) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, len) };
    simd_math::similaridade_cosseno(a, b)
}

/// Gera embeddings determinísticos on-device para uso em Web / SPA
#[no_mangle]
pub extern "C" fn thz_wasm_gerar_embedding(texto: *const c_char, dim: usize, out_ptr: *mut f32) -> i32 {
    if texto.is_null() || out_ptr.is_null() || dim == 0 {
        return 0;
    }
    let s = match unsafe { CStr::from_ptr(texto).to_str() } {
        Ok(v) => v,
        Err(_) => return 0,
    };
    let emb = ThzEmbeddingEngine::gerar_embedding(s, dim);
    unsafe {
        std::ptr::copy_nonoverlapping(emb.as_ptr(), out_ptr, dim);
    }
    1
}
