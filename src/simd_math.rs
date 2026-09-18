//! Álgebra Vetorial e Busca Semântica SIMD
//!
//! Funções otimizadas para processamento de embeddings e cálculo de similaridade
//! com autovetorização para AVX2 / AVX-512 / NEON.

#[inline]
pub fn produto_escalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Dimensões dos vetores devem ser idênticas");
    
    let mut soma = 0.0f32;
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        soma += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3];
    }

    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        soma += x * y;
    }

    soma
}

#[inline]
pub fn norma(a: &[f32]) -> f32 {
    let mut soma_quadrados = 0.0f32;
    let chunks = a.chunks_exact(4);
    let rem = chunks.remainder();

    for c in chunks {
        soma_quadrados += c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + c[3] * c[3];
    }

    for x in rem {
        soma_quadrados += x * x;
    }

    soma_quadrados.sqrt()
}

#[inline]
pub fn similaridade_cosseno(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Dimensões dos vetores devem ser idênticas");
    
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[inline]
pub fn distancia_euclidiana(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Dimensões dos vetores devem ser idênticas");
    
    let mut soma_diff_sq = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        soma_diff_sq += diff * diff;
    }

    soma_diff_sq.sqrt()
}
