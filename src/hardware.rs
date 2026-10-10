//! THZ-LANG Hardware Transport Subsystem (C-ABI Driverless Transport)
//!
//! Fornece comunicação direta sem spooler do Windows com impressoras térmicas
//! via USB (Win32 usbprint.sys) e COM/Bluetooth SPP (serialport com timeout síncrono).

use std::ffi::CStr;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::raw::c_char;
use std::time::Duration;

#[no_mangle]
pub extern "C" fn thz_hardware_usb_enviar(
    caminho: *const c_char,
    payload: *const u8,
    len: usize,
) -> i32 {
    if caminho.is_null() || payload.is_null() || len == 0 {
        return 0;
    }

    let path_str = match unsafe { CStr::from_ptr(caminho).to_str() } {
        Ok(p) => p,
        Err(_) => return 0,
    };

    let slice = unsafe { std::slice::from_raw_parts(payload, len) };

    match OpenOptions::new().write(true).open(path_str) {
        Ok(mut file) => match file.write_all(slice).and_then(|_| file.flush()) {
            Ok(_) => 1,
            Err(_) => 0,
        },
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn thz_hardware_com_enviar(
    porta: *const c_char,
    baud: u32,
    payload: *const u8,
    len: usize,
) -> i32 {
    if porta.is_null() || payload.is_null() || len == 0 {
        return 0;
    }

    let port_str = match unsafe { CStr::from_ptr(porta).to_str() } {
        Ok(p) => p,
        Err(_) => return 0,
    };

    let slice = unsafe { std::slice::from_raw_parts(payload, len) };
    let baud_rate = if baud == 0 { 9600 } else { baud };

    match serialport::new(port_str, baud_rate)
        .timeout(Duration::from_secs(10))
        .open()
    {
        Ok(mut port) => match port.write_all(slice).and_then(|_| port.flush()) {
            Ok(_) => 1,
            Err(_) => 0,
        },
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn thz_hardware_imprimir_cupom(
    tipo: *const c_char,
    destino: *const c_char,
    payload: *const u8,
    len: usize,
) -> i32 {
    if tipo.is_null() || destino.is_null() || payload.is_null() || len == 0 {
        return 0;
    }

    let tipo_str = match unsafe { CStr::from_ptr(tipo).to_str() } {
        Ok(t) => t.to_ascii_uppercase(),
        Err(_) => return 0,
    };

    if tipo_str.contains("USB") {
        thz_hardware_usb_enviar(destino, payload, len)
    } else {
        thz_hardware_com_enviar(destino, 9600, payload, len)
    }
}

#[no_mangle]
pub extern "C" fn thz_hardware_descobrir_dispositivos(buffer: *mut c_char, max_len: usize) -> i32 {
    if buffer.is_null() || max_len == 0 {
        return 0;
    }

    let ports = match serialport::available_ports() {
        Ok(p) => p,
        Err(_) => Vec::new(),
    };

    let mut port_names = Vec::new();
    for p in ports {
        port_names.push(p.port_name);
    }

    let json = match serde_json::to_string(&port_names) {
        Ok(j) => j,
        Err(_) => "[]".to_string(),
    };

    let bytes = json.as_bytes();
    if bytes.len() + 1 > max_len {
        return 0;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer as *mut u8, bytes.len());
        std::ptr::write((buffer as *mut u8).add(bytes.len()), 0);
    }

    bytes.len() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validacao_parametros_nulos() {
        assert_eq!(
            thz_hardware_usb_enviar(std::ptr::null(), std::ptr::null(), 0),
            0
        );
        assert_eq!(
            thz_hardware_com_enviar(std::ptr::null(), 9600, std::ptr::null(), 0),
            0
        );
        assert_eq!(
            thz_hardware_imprimir_cupom(std::ptr::null(), std::ptr::null(), std::ptr::null(), 0),
            0
        );
    }
}
