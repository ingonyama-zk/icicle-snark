mod utils;
mod file_wrapper;
mod groth16;
mod icicle;
mod zkey;

use file_wrapper::FileWrapper;
use icicle_bn254::curve::{CurveCfg, G2CurveCfg, ScalarField};
use icicle_core::curve::{Affine, Projective};
use groth16::{
    prove::{prove as groth16_prove, parallel_prove as groth16_parallel_prove, Proof},
    verify::VerificationKey
};

use std::ffi::{c_char, c_ulonglong, CStr};
use utils::string_to_ffi_buf;

#[cfg(feature = "debug")]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        println!($($arg)*);
    };
}

#[cfg(not(feature = "debug"))]
macro_rules! debug_println {
    ($($arg:tt)*) => {};
}
#[cfg(feature = "android")]
use log::LevelFilter;
#[cfg(feature = "android")]
use android_logger::Config;
#[cfg(feature = "android")]
use std::sync::Once;
#[cfg(feature = "android")]
static INIT: Once = Once::new();

#[cfg(feature = "android")]
fn init_android_logger() {
    INIT.call_once(|| {
        android_logger::init_once(
            Config::default()
                .with_max_level(LevelFilter::Trace)
                .with_tag("IMP1")
                .with_filter(android_logger::FilterBuilder::new()
                    .parse("debug,IMP1=trace")
                    .build()),
        );
        log::info!("Android logger initialized successfully");
    });
}


pub type F = ScalarField;
pub type C1 = CurveCfg;
pub type C2 = G2CurveCfg;
pub type G1 = Affine<C1>;
pub type G2 = Affine<C2>;
pub type ProjectiveG1 = Projective<C1>;
pub type ProjectiveG2 = Projective<C2>;

#[derive(Debug, Clone, Copy)]
enum ProtocolId {
    Groth16 = 1,
}

impl ProtocolId {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(ProtocolId::Groth16),
            _ => None,
        }
    }
}

#[no_mangle]
pub extern "C" fn free_parallel_results(results: *const ProverResult, count: usize) {
    unsafe {
        if !results.is_null() {
            // Convert back to Vec and let it drop
            let _ = Vec::from_raw_parts(results as *mut ProverResult, count, count);
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum DeviceType {
    CpuMetal,
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ProverResult {
    pub value: i32,
}

impl ProverResult {
    pub const SUCCESS: ProverResult = ProverResult { value: 0 };
    pub const FAILURE: ProverResult = ProverResult { value: 1 };
}



#[no_mangle]
pub extern "C" fn prove(
    witness_path: *const c_char,
    zkey_path: *const c_char,
    proof_path: *const c_char,
    public_path: *const c_char,
    error_msg: *mut c_char,
    error_msg_maxsize: c_ulonglong,
    device_type: DeviceType,
) -> ProverResult {
    #[cfg(feature = "android")]
    init_android_logger();

    unsafe {
        let witness_path = CStr::from_ptr(witness_path).to_str().unwrap();
        let zkey_path = CStr::from_ptr(zkey_path).to_str().unwrap();
        let proof_path = CStr::from_ptr(proof_path).to_str().unwrap();
        let public_path = CStr::from_ptr(public_path).to_str().unwrap();

        let zkey = zkey::ZKey::load(zkey_path);
        if zkey.is_err() {
            let zkey_error = zkey.err().unwrap();
            string_to_ffi_buf(zkey_error.to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
            return ProverResult::FAILURE;
        }
        let zkey = zkey.unwrap();

        match zkey.protocol_id {
            ProtocolId::Groth16 => {
                let prove_result = groth16_prove(witness_path, &zkey, device_type);
                if prove_result.is_err() {
                    string_to_ffi_buf(prove_result.err().unwrap().to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
                    return ProverResult::FAILURE;
                }
                let (proof_data, public_signals) = prove_result.unwrap();
                let proof_written = FileWrapper::save_json_file(proof_path, &proof_data);
                if proof_written.is_err() {
                    string_to_ffi_buf(proof_written.err().unwrap().to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
                    return ProverResult::FAILURE;
                }
                let public_written = FileWrapper::save_json_file(public_path, &public_signals);
                if public_written.is_err() {
                    string_to_ffi_buf(public_written.err().unwrap().to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
                    return ProverResult::FAILURE;
                }
                ProverResult::SUCCESS
            },
        }
    }
}

#[no_mangle]
pub extern "C" fn parallel_prove(
    witness_paths: *const *const c_char,
    zkey_path: *const c_char,
    proof_paths: *const *const c_char,
    public_paths: *const *const c_char,
    num_proofs: c_ulonglong,
    error_msg: *mut c_char,
    error_msg_maxsize: c_ulonglong,
    device_type: DeviceType,
    max_batch_size: c_ulonglong,
) -> *const ProverResult {
    debug_println!("[RUST] parallel_prove called with num_proofs: {}", num_proofs);
    unsafe {
        // Convert C arrays to Rust vectors
        let mut witness_paths_vec = Vec::new();
        let mut proof_paths_vec = Vec::new();
        let mut public_paths_vec = Vec::new();

        // Get the single zkey path
        let zkey_path_str = CStr::from_ptr(zkey_path).to_str().unwrap();
        debug_println!("[RUST] zkey_path: {}", zkey_path_str);

        for i in 0..num_proofs {
            let witness_path = CStr::from_ptr(*witness_paths.offset(i as isize)).to_str().unwrap();
            let proof_path = CStr::from_ptr(*proof_paths.offset(i as isize)).to_str().unwrap();
            let public_path = CStr::from_ptr(*public_paths.offset(i as isize)).to_str().unwrap();

            debug_println!("[RUST] Proof {}: witness={}, proof={}, public={}", 
                     i + 1, witness_path, proof_path, public_path);

            witness_paths_vec.push(witness_path.to_string());
            proof_paths_vec.push(proof_path.to_string());
            public_paths_vec.push(public_path.to_string());
        }
        
        debug_println!("[RUST] Converted {} witness paths, {} proof paths, {} public paths", 
                 witness_paths_vec.len(), proof_paths_vec.len(), public_paths_vec.len());

        let parallel_result = groth16_parallel_prove(
            &witness_paths_vec,
            zkey_path_str,
            &proof_paths_vec,
            &public_paths_vec,
            device_type,
            if max_batch_size > 0 { Some(max_batch_size as usize) } else { None },
        );

        match parallel_result {
            Ok(results) => {
                debug_println!("[RUST] parallel_prove succeeded with {} results", results.len());
                // Convert results directly to ProverResult array
                let mut c_results = Vec::with_capacity(results.len());
                for (i, result) in results.iter().enumerate() {
                    debug_println!("[RUST] Result {}: {:?}", i + 1, result);
                    c_results.push(*result);
                }
                let boxed_results = c_results.into_boxed_slice();
                debug_println!("[RUST] Returning {} results to C", boxed_results.len());
                debug_println!("[RUST] Size of ProverResult: {} bytes", std::mem::size_of::<ProverResult>());
                Box::into_raw(boxed_results) as *const ProverResult
            }
            Err(e) => {
                debug_println!("[RUST] parallel_prove failed with error: {}", e);
                string_to_ffi_buf(e.to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
                std::ptr::null_mut()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum VerifierResult {
    Success,
    Failure,
}

// TODO: this needs to be updated for more protocols
#[no_mangle]
pub extern "C" fn verify(
    proof_path: *const c_char,
    public_path: *const c_char,
    vk_path: *const c_char,
) -> VerifierResult {
    #[cfg(feature = "android")]
    init_android_logger();

    unsafe {
        let proof_path = CStr::from_ptr(proof_path).to_str().unwrap();
        let proof_str = std::fs::read_to_string(proof_path).unwrap();
        let proof: Proof = serde_json::from_str(&proof_str).unwrap();
        
        let public_path = CStr::from_ptr(public_path).to_str().unwrap();
        let public_str = std::fs::read_to_string(public_path).unwrap();
        let public: Vec<String> = serde_json::from_str(&public_str).unwrap();
        
        let vk_path = CStr::from_ptr(vk_path).to_str().unwrap();
        let vk_str = std::fs::read_to_string(vk_path).unwrap();
        let vk: VerificationKey = serde_json::from_str(&vk_str).unwrap();

        let pairing_result = groth16::verify::verify(&proof, &public, &vk);

        if !pairing_result {
            return VerifierResult::Failure;
        }

        VerifierResult::Success
    }
}
