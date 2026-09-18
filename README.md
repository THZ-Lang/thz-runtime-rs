# thz-runtime-rs — Runtime Nativo Oficial em Rust

Runtime nativo de alta performance do THZ-LANG escrito em Rust com **C ABI**.

---

## ⚡ Recursos Principais

- **Arena de Memória $O(1)$**: Alocador contíguo linear sem pressão de Garbage Collector (GC)
- **SIMD Vetorial**: Aceleração matemática vetorial com suporte a AVX2 e AVX-512
- **Criptografia Avançada**: Primitivas criptográficas e funções hash de alta performance
- **WASM Bridge**: Suporte à compilação para WebAssembly
- **C ABI**: Integração FFI direta com código de máquina nativo e compilador LLVM Clang

---

## 📦 Compilação

```bash
cargo build --release
```
