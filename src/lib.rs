//! THZ-LANG High-Performance Native Runtime (Rust Engine)
//!
//! Dual-OS ABI C export para LLVM Clang, JNI e FFI.

pub mod arena;
pub mod simd_math;
pub mod crypto;
pub mod ml;
pub mod wasm;
pub mod llm;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use arena::ThzArena;

#[no_mangle]
pub extern "C" fn thz_arena_alloc(bytes: u64) -> *mut ThzArena {
    match ThzArena::new(bytes as usize) {
        Some(arena) => Box::into_raw(arena),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn thz_arena_free_all(arena_ptr: *mut ThzArena) {
    if !arena_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(arena_ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn thz_tamanho_str(s: *const c_char) -> i32 {
    if s.is_null() {
        return 0;
    }
    unsafe { CStr::from_ptr(s).to_bytes().len() as i32 }
}

#[no_mangle]
pub extern "C" fn thz_char_at(s: *const c_char, idx: i32) -> i32 {
    if s.is_null() || idx < 0 {
        return 0;
    }
    let bytes = unsafe { CStr::from_ptr(s).to_bytes() };
    if (idx as usize) < bytes.len() {
        bytes[idx as usize] as i32
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn thz_substring(s: *const c_char, inicio: i32, len: i32) -> *mut c_char {
    if s.is_null() || inicio < 0 || len <= 0 {
        return CString::new("").unwrap().into_raw();
    }
    let bytes = unsafe { CStr::from_ptr(s).to_bytes() };
    let start = inicio as usize;
    if start >= bytes.len() {
        return CString::new("").unwrap().into_raw();
    }
    let end = (start + len as usize).min(bytes.len());
    let sub = &bytes[start..end];
    match CString::new(sub) {
        Ok(cs) => cs.into_raw(),
        Err(_) => CString::new("").unwrap().into_raw(),
    }
}

#[no_mangle]
pub extern "C" fn thz_ler_arquivo(caminho: *const c_char) -> *mut c_char {
    if caminho.is_null() {
        return CString::new("").unwrap().into_raw();
    }
    let path = match unsafe { CStr::from_ptr(caminho).to_str() } {
        Ok(p) => p,
        Err(_) => return CString::new("").unwrap().into_raw(),
    };
    match std::fs::read_to_string(path) {
        Ok(content) => CString::new(content).unwrap_or_default().into_raw(),
        Err(_) => CString::new("").unwrap().into_raw(),
    }
}

#[no_mangle]
pub extern "C" fn thz_escrever_arquivo(caminho: *const c_char, conteudo: *const c_char) -> i32 {
    if caminho.is_null() || conteudo.is_null() {
        return 0;
    }
    let path = match unsafe { CStr::from_ptr(caminho).to_str() } {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let content = match unsafe { CStr::from_ptr(conteudo).to_str() } {
        Ok(c) => c,
        Err(_) => return 0,
    };
    if std::fs::write(path, content).is_ok() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn thz_exiba_str(msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    if let Ok(s) = unsafe { CStr::from_ptr(msg).to_str() } {
        println!("{}", s);
    }
}

#[no_mangle]
pub extern "C" fn thz_exiba_i64(val: i64) {
    println!("{}", val);
}

#[no_mangle]
pub extern "C" fn thz_exiba_i32(val: i32) {
    println!("{}", val);
}

#[no_mangle]
pub extern "C" fn thz_exiba_bool(val: bool) {
    if val {
        println!("VERDADEIRO");
    } else {
        println!("FALSO");
    }
}

#[no_mangle]
pub extern "C" fn thz_exiba_i128(val: i128, scale: i32) {
    if scale <= 0 {
        println!("{}", val);
        return;
    }
    let sinal = if val < 0 { "-" } else { "" };
    let abs_val = val.abs();
    let mut fator: i128 = 1;
    for _ in 0..scale {
        fator *= 10;
    }
    let inteira = abs_val / fator;
    let mut fracionaria = (abs_val % fator).to_string();
    while fracionaria.len() < (scale as usize) {
        fracionaria = format!("0{}", fracionaria);
    }
    println!("{}{}.{}", sinal, inteira, fracionaria);
}

// -----------------------------------------------------------------------------
// FFI: Álgebra Vetorial & SIMD
// -----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn thz_vetor_similaridade_cosseno(a_ptr: *const f32, b_ptr: *const f32, len: usize) -> f32 {
    if a_ptr.is_null() || b_ptr.is_null() || len == 0 {
        return 0.0;
    }
    let a = unsafe { std::slice::from_raw_parts(a_ptr, len) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, len) };
    simd_math::similaridade_cosseno(a, b)
}

#[no_mangle]
pub extern "C" fn thz_vetor_distancia_euclidiana(a_ptr: *const f32, b_ptr: *const f32, len: usize) -> f32 {
    if a_ptr.is_null() || b_ptr.is_null() || len == 0 {
        return 0.0;
    }
    let a = unsafe { std::slice::from_raw_parts(a_ptr, len) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, len) };
    simd_math::distancia_euclidiana(a, b)
}

#[no_mangle]
pub extern "C" fn thz_vetor_produto_escalar(a_ptr: *const f32, b_ptr: *const f32, len: usize) -> f32 {
    if a_ptr.is_null() || b_ptr.is_null() || len == 0 {
        return 0.0;
    }
    let a = unsafe { std::slice::from_raw_parts(a_ptr, len) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, len) };
    simd_math::produto_escalar(a, b)
}

// -----------------------------------------------------------------------------
// FFI: Criptografia (Argon2id)
// -----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn thz_crypto_argon2id(senha: *const c_char) -> *mut c_char {
    if senha.is_null() {
        return CString::new("").unwrap().into_raw();
    }
    let s = match unsafe { CStr::from_ptr(senha).to_str() } {
        Ok(v) => v,
        Err(_) => return CString::new("").unwrap().into_raw(),
    };
    match crypto::hash_argon2id(s) {
        Ok(h) => CString::new(h).unwrap_or_default().into_raw(),
        Err(_) => CString::new("").unwrap().into_raw(),
    }
}

// -----------------------------------------------------------------------------
// FFI: IA & Machine Learning On-Device
// -----------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn thz_ia_embedding_texto(texto: *const c_char, dimensoes: usize, saida_ptr: *mut f32) -> i32 {
    if texto.is_null() || saida_ptr.is_null() || dimensoes == 0 {
        return 0;
    }
    let s = match unsafe { CStr::from_ptr(texto).to_str() } {
        Ok(v) => v,
        Err(_) => return 0,
    };
    let emb = ml::ThzEmbeddingEngine::gerar_embedding(s, dimensoes);
    unsafe {
        std::ptr::copy_nonoverlapping(emb.as_ptr(), saida_ptr, dimensoes);
    }
    1
}

#[no_mangle]
pub extern "C" fn thz_ml_predizer_sigmoide(features_ptr: *const f32, pesos_ptr: *const f32, len: usize, bias: f32) -> f32 {
    if features_ptr.is_null() || pesos_ptr.is_null() || len == 0 {
        return 0.5;
    }
    let features = unsafe { std::slice::from_raw_parts(features_ptr, len) };
    let pesos = unsafe { std::slice::from_raw_parts(pesos_ptr, len) };
    let classificador = ml::ThzClassificadorSimples::new(pesos.to_vec(), bias);
    classificador.predizer_probabilidade(features)
}

#[no_mangle]
pub extern "C" fn thz_liberar_str(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}
