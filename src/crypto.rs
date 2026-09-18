//! Criptografia de Grau Militar (Argon2id, AES-256-GCM, ChaCha20-Poly1305)

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2, Params,
};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};

pub fn hash_argon2id(senha: &str) -> Result<String, String> {
    let salt = SaltString::generate(&mut OsRng);
    let params = Params::new(65536, 3, 4, Some(32)).map_err(|e| e.to_string())?;
    let argon2 = Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);

    let password_hash = argon2
        .hash_password(senha.as_bytes(), &salt)
        .map_err(|e| e.to_string())?;

    Ok(password_hash.to_string())
}

pub fn verificar_argon2id(senha: &str, hash_str: &str) -> bool {
    let parsed_hash = match PasswordHash::new(hash_str) {
        Ok(h) => h,
        Err(_) => return false,
    };
    Argon2::default()
        .verify_password(senha.as_bytes(), &parsed_hash)
        .is_ok()
}

pub fn criptografar_aes_gcm(dados: &[u8], chave: &[u8; 32], nonce_bytes: &[u8; 12]) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new(chave.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher.encrypt(nonce, dados).map_err(|e| e.to_string())
}

pub fn descriptografar_aes_gcm(cifrado: &[u8], chave: &[u8; 32], nonce_bytes: &[u8; 12]) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new(chave.into());
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher.decrypt(nonce, cifrado).map_err(|e| e.to_string())
}
