//! Machine Learning Clássico e Embeddings Nativos em Rust (Zero Python)
//!
//! Algoritmos determinísticos de alta velocidade para predição tabular,
//! classificação e vetorização sem dependências externas pesadas.


/// Vetorizador de Texto e Geração de Embeddings Semânticos em Memória
pub struct ThzEmbeddingEngine;

impl ThzEmbeddingEngine {
    /// Gera embedding determinístico de dimensão fixa (ex: 128, 256, 384) para um texto.
    pub fn gerar_embedding(texto: &str, dimensoes: usize) -> Vec<f32> {
        let mut vetor = vec![0.0f32; dimensoes];
        if texto.is_empty() || dimensoes == 0 {
            return vetor;
        }

        let palavras: Vec<&str> = texto
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        if palavras.is_empty() {
            return vetor;
        }

        // Hash Feature Embedding com dispersão uniforme e n-gramas de caracteres
        for palavra in &palavras {
            let palavra_min = palavra.to_lowercase();
            let p_bytes = palavra_min.as_bytes();
            
            // Hash primário da palavra (FNV-1a 64-bit)
            let mut hash: u64 = 0xcbf29ce484222325;
            for &b in p_bytes {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }

            let idx = (hash as usize) % dimensoes;
            let sinal = if (hash >> 32) & 1 == 0 { 1.0f32 } else { -1.0f32 };
            vetor[idx] += sinal;

            // Sub-palavras (tri-gramas) para capturar morfologia e semântica em português
            if p_bytes.len() >= 3 {
                for window in p_bytes.windows(3) {
                    let mut sub_hash: u64 = 0xcbf29ce484222325;
                    for &b in window {
                        sub_hash ^= b as u64;
                        sub_hash = sub_hash.wrapping_mul(0x100000001b3);
                    }
                    let sub_idx = (sub_hash as usize) % dimensoes;
                    let sub_sinal = if (sub_hash >> 32) & 1 == 0 { 0.5f32 } else { -0.5f32 };
                    vetor[sub_idx] += sub_sinal;
                }
            }
        }

        // Normalização L2 (Norma Unitária = 1.0)
        let norma_sq: f32 = vetor.iter().map(|&x| x * x).sum();
        let norma = norma_sq.sqrt();
        if norma > 0.0 {
            for v in &mut vetor {
                *v /= norma;
            }
        }

        vetor
    }
}

/// Motor de Árvore de Decisão / Classificação Tabular
pub struct ThzClassificadorSimples {
    pub pesos: Vec<f32>,
    pub bias: f32,
}

impl ThzClassificadorSimples {
    pub fn new(pesos: Vec<f32>, bias: f32) -> Self {
        Self { pesos, bias }
    }

    /// Prediz uma probabilidade sigmoide a partir de um vetor de features
    pub fn predizer_probabilidade(&self, features: &[f32]) -> f32 {
        let mut score = self.bias;
        for (i, &f) in features.iter().enumerate() {
            if i < self.pesos.len() {
                score += f * self.pesos[i];
            }
        }
        // Sigmoide: 1 / (1 + e^-z)
        1.0 / (1.0 + (-score).exp())
    }
}
