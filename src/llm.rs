//! THZ-LANG LLM Engine — Inferência On-Device via llama.cpp
//!
//! Fornece carregamento de modelos GGUF, geração de texto, embeddings
//! e suporte a fine-tuning LoRA para o THZ-Agent.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

// ---------------------------------------------------------------------------
// Configuração do modelo
// ---------------------------------------------------------------------------

/// Parâmetros de geração de texto
#[derive(Debug, Clone)]
pub struct GerarParams {
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub seed: i32,
}

impl Default for GerarParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            seed: -1,
        }
    }
}

/// Configuração para carregamento de modelo
#[derive(Debug, Clone)]
pub struct ModeloConfig {
    pub caminho: String,
    pub n_gpu_layers: i32,
    pub n_ctx: i32,
    pub n_batch: i32,
    pub vocab_only: bool,
}

impl Default for ModeloConfig {
    fn default() -> Self {
        Self {
            caminho: String::new(),
            n_gpu_layers: 0,
            n_ctx: 2048,
            n_batch: 512,
            vocab_only: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Contexto LLM (handle opaque para FFI)
// ---------------------------------------------------------------------------

/// Contexto de inferência LLM. Armazena o modelo carregado e metadados.
/// O ponteiro para esta struct é o "handle" exposto via FFI.
pub struct ThzLlmContext {
    pub config: ModeloConfig,
    pub params: GerarParams,
    pub modelo_carregado: bool,
    pub n_tokens_gerados: u64,
    pub memoria_ativa: bool,
}

impl ThzLlmContext {
    pub fn novo(config: ModeloConfig) -> Self {
        Self {
            config,
            params: GerarParams::default(),
            modelo_carregado: false,
            n_tokens_gerados: 0,
            memoria_ativa: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Backend trait (abstração para Local e API)
// ---------------------------------------------------------------------------

/// Trait para backends de LLM. Permite trocar entre local (llama.cpp) e API.
pub trait LlmBackend: Send {
    fn nome(&self) -> &str;
    fn gerar(&mut self, prompt: &str, params: &GerarParams) -> Result<String, String>;
    fn embedding(&mut self, texto: &str) -> Result<Vec<f32>, String>;
    fn token_count(&self, texto: &str) -> usize;
    fn modelo_info(&self) -> ModeloInfo;
    fn fechar(&mut self);
}

#[derive(Debug, Clone)]
pub struct ModeloInfo {
    pub nome: String,
    pub tipo: String,       // "local" ou "api"
    pub provider: String,   // "llama.cpp", "openai", "anthropic", etc.
    pub dims_embedding: usize,
}

// ---------------------------------------------------------------------------
// Backend Local (llama.cpp via FFI)
// ---------------------------------------------------------------------------

/// Backend local usando llama.cpp. Quando compilado com o submodule vendor/,
/// carrega modelos GGUF diretamente na memória.
pub struct LocalBackend {
    pub config: ModeloConfig,
    pub modelo_valido: bool,
}

impl LocalBackend {
    pub fn novo(config: ModeloConfig) -> Result<Self, String> {
        if !Path::new(&config.caminho).exists() {
            return Err(format!(
                "Arquivo de modelo não encontrado: {}",
                config.caminho
            ));
        }

        // Validação básica de extensão
        let ext = Path::new(&config.caminho)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        if ext != "gguf" {
            return Err(format!(
                "Formato não suportado: .{}. Use arquivos .gguf",
                ext
            ));
        }

        Ok(Self {
            config,
            modelo_valido: true,
        })
    }

    /// Gera texto usando llama.cpp (stub — substituir por llama_cpp bindings reais)
    fn gerar_interno(&mut self, prompt: &str, params: &GerarParams) -> Result<String, String> {
        if !self.modelo_valido {
            return Err("Modelo não carregado".into());
        }

        // TODO: Integrar com llama-cpp-2 crate ou cc::Build de llama.cpp
        // Por agora, retorna resposta placeholder para validar o pipeline
        let resposta = format!(
            "[THZ-LLM Local] Prompt recebido ({} chars), max_tokens={}, temp={}. \
             Resposta simulada — integre llama.cpp para inferência real.",
            prompt.len(),
            params.max_tokens,
            params.temperature
        );

        Ok(resposta)
    }

    /// Gera embedding usando o modelo (se suportado)
    fn embedding_interno(&mut self, _texto: &str) -> Result<Vec<f32>, String> {
        // TODO: Implementar via llama_cpp embedding API
        // Por agora, retorna embedding zero de 384 dimensões
        Ok(vec![0.0; 384])
    }
}

impl LlmBackend for LocalBackend {
    fn nome(&self) -> &str {
        "llama.cpp (local)"
    }

    fn gerar(&mut self, prompt: &str, params: &GerarParams) -> Result<String, String> {
        self.gerar_interno(prompt, params)
    }

    fn embedding(&mut self, texto: &str) -> Result<Vec<f32>, String> {
        self.embedding_interno(texto)
    }

    fn token_count(&self, texto: &str) -> usize {
        // Estimativa simples: ~1 token por 4 caracteres (BPE médio)
        texto.len() / 4
    }

    fn modelo_info(&self) -> ModeloInfo {
        let nome = Path::new(&self.config.caminho)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("desconhecido")
            .to_string();

        ModeloInfo {
            nome,
            tipo: "local".into(),
            provider: "llama.cpp".into(),
            dims_embedding: 384,
        }
    }

    fn fechar(&mut self) {
        self.modelo_valido = false;
    }
}

// ---------------------------------------------------------------------------
// Backend API — implementado em Java (ThzAgent usa java.net.http.HttpClient)
// O Rust foca em inferência local via llama.cpp.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// FFI Functions — expostas para Java via JNI/Panama
// ---------------------------------------------------------------------------

/// Cria um novo contexto LLM com um modelo local
#[no_mangle]
pub extern "C" fn thz_llm_carregar(
    caminho: *const c_char,
    n_gpu_layers: i32,
    n_ctx: i32,
) -> *mut ThzLlmContext {
    if caminho.is_null() {
        return std::ptr::null_mut();
    }

    let path = match unsafe { CStr::from_ptr(caminho).to_str() } {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };

    let config = ModeloConfig {
        caminho: path.to_string(),
        n_gpu_layers,
        n_ctx,
        ..Default::default()
    };

    let mut ctx = ThzLlmContext::novo(config.clone());

    // Tenta criar o backend local
    match LocalBackend::novo(config) {
        Ok(_backend) => {
            ctx.modelo_carregado = true;
        }
        Err(e) => {
            eprintln!("[THZ-LLM] Aviso: {}", e);
            ctx.modelo_carregado = false;
        }
    }

    Box::into_raw(Box::new(ctx))
}

/// Cria um contexto LLM conectado a uma API remota
/// (API backend é gerenciado em Java — esta função apenas cria o handle)
#[no_mangle]
pub extern "C" fn thz_llm_carregar_api(
    url: *const c_char,
    _api_key: *const c_char,
    modelo: *const c_char,
) -> *mut ThzLlmContext {
    if url.is_null() || modelo.is_null() {
        return std::ptr::null_mut();
    }

    let url_str = match unsafe { CStr::from_ptr(url).to_str() } {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };
    let model_str = match unsafe { CStr::from_ptr(modelo).to_str() } {
        Ok(p) => p,
        Err(_) => return std::ptr::null_mut(),
    };

    let config = ModeloConfig {
        caminho: format!("api:{}", url_str),
        ..Default::default()
    };

    let mut ctx = ThzLlmContext::novo(config);
    ctx.modelo_carregado = true;

    eprintln!(
        "[THZ-LLM] API handle criado: {} modelo={}",
        url_str, model_str
    );

    Box::into_raw(Box::new(ctx))
}

/// Gera texto a partir de um prompt
#[no_mangle]
pub extern "C" fn thz_llm_gerar(
    ctx: *mut ThzLlmContext,
    prompt: *const c_char,
    max_tokens: i32,
    temperature: f32,
    top_k: i32,
    top_p: f32,
) -> *mut c_char {
    if ctx.is_null() || prompt.is_null() {
        return CString::new("").unwrap().into_raw();
    }

    let ctx_ref = unsafe { &mut *ctx };
    let p = match unsafe { CStr::from_ptr(prompt).to_str() } {
        Ok(s) => s,
        Err(_) => return CString::new("").unwrap().into_raw(),
    };

    let params = GerarParams {
        max_tokens,
        temperature,
        top_k,
        top_p,
        ..Default::default()
    };

    // Cria backend local e gera
    let resultado = if ctx_ref.config.caminho.starts_with("api:") {
        // Backend API — por enquanto retorna stub
        Ok(format!(
            "[THZ-LLM API] Resposta simulada para prompt de {} chars",
            p.len()
        ))
    } else {
        match LocalBackend::novo(ctx_ref.config.clone()) {
            Ok(mut backend) => backend.gerar(p, &params),
            Err(e) => Err(e),
        }
    };

    match resultado {
        Ok(texto) => {
            ctx_ref.n_tokens_gerados += (texto.len() / 4) as u64;
            CString::new(texto).unwrap_or_default().into_raw()
        }
        Err(_) => CString::new("").unwrap().into_raw(),
    }
}

/// Gera embedding para um texto
#[no_mangle]
pub extern "C" fn thz_llm_embedding(
    ctx: *mut ThzLlmContext,
    texto: *const c_char,
    saida_ptr: *mut f32,
    dims: usize,
) -> i32 {
    if ctx.is_null() || texto.is_null() || saida_ptr.is_null() || dims == 0 {
        return 0;
    }

    let ctx_ref = unsafe { &mut *ctx };
    let t = match unsafe { CStr::from_ptr(texto).to_str() } {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let emb = match LocalBackend::novo(ctx_ref.config.clone()) {
        Ok(mut backend) => backend.embedding(t).unwrap_or_else(|_| vec![0.0; dims]),
        Err(_) => vec![0.0; dims],
    };

    let copy_len = dims.min(emb.len());
    unsafe {
        std::ptr::copy_nonoverlapping(emb.as_ptr(), saida_ptr, copy_len);
    }

    1
}

/// Define temperatura de geração
#[no_mangle]
pub extern "C" fn thz_llm_set_temperatura(ctx: *mut ThzLlmContext, temp: f32) {
    if !ctx.is_null() {
        unsafe {
            (*ctx).params.temperature = temp;
        }
    }
}

/// Define top_k de geração
#[no_mangle]
pub extern "C" fn thz_llm_set_top_k(ctx: *mut ThzLlmContext, top_k: i32) {
    if !ctx.is_null() {
        unsafe {
            (*ctx).params.top_k = top_k;
        }
    }
}

/// Define top_p de geração
#[no_mangle]
pub extern "C" fn thz_llm_set_top_p(ctx: *mut ThzLlmContext, top_p: f32) {
    if !ctx.is_null() {
        unsafe {
            (*ctx).params.top_p = top_p;
        }
    }
}

/// Retorna info do modelo carregado (nome, tipo, provider)
#[no_mangle]
pub extern "C" fn thz_llm_info_modelo(ctx: *const ThzLlmContext) -> *mut c_char {
    if ctx.is_null() {
        return CString::new("Nenhum modelo carregado").unwrap().into_raw();
    }

    let ctx_ref = unsafe { &*ctx };
    let info = if ctx_ref.config.caminho.starts_with("api:") {
        format!("API: {}", ctx_ref.config.caminho)
    } else {
        let nome = Path::new(&ctx_ref.config.caminho)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("desconhecido");
        format!("Local: {} (gpu_layers={})", nome, ctx_ref.config.n_gpu_layers)
    };

    CString::new(info).unwrap_or_default().into_raw()
}

/// Retorna total de tokens gerados nesta sessão
#[no_mangle]
pub extern "C" fn thz_llm_tokens_gerados(ctx: *const ThzLlmContext) -> u64 {
    if ctx.is_null() {
        return 0;
    }
    unsafe { (*ctx).n_tokens_gerados }
}

/// Libera memória do contexto LLM
#[no_mangle]
pub extern "C" fn thz_llm_liberar(ctx: *mut ThzLlmContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}
