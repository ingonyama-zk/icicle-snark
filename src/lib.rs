mod utils;
mod file_wrapper;
mod groth16;
mod icicle;
mod zkey;

use file_wrapper::FileWrapper;
use icicle_bn254::curve::{CurveCfg, G2CurveCfg, ScalarField};
use icicle_core::curve::{Affine, Projective};
use groth16::{
    prove::{prove as groth16_prove, Proof},
    verify::VerificationKey
};
// use serde_json;

use std::ffi::{c_char, c_ulonglong, CStr};
use utils::string_to_ffi_buf;

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

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum DeviceType {
    Cpu,
    Metal,
    CpuMetal,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum ProverResult {
    Success,
    Failure,
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
    unsafe {
        let witness_path = CStr::from_ptr(witness_path).to_str().unwrap();
        let zkey_path = CStr::from_ptr(zkey_path).to_str().unwrap();
        let proof_path = CStr::from_ptr(proof_path).to_str().unwrap();
        let public_path = CStr::from_ptr(public_path).to_str().unwrap();

        let zkey = zkey::ZKey::load(zkey_path);
        if zkey.is_err() {
            let zkey_error = zkey.err().unwrap();
            string_to_ffi_buf(zkey_error.to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
            return ProverResult::Failure;
        }
        let zkey = zkey.unwrap();

        match zkey.protocol_id {
            ProtocolId::Groth16 => {
                let prove_result = groth16_prove(witness_path, &zkey, device_type);
                if prove_result.is_err() {
                    string_to_ffi_buf(prove_result.err().unwrap().to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
                    return ProverResult::Failure;
                }
                let (proof_data, public_signals) = prove_result.unwrap();
                let proof_written = FileWrapper::save_json_file(proof_path, &proof_data);
                if proof_written.is_err() {
                    string_to_ffi_buf(proof_written.err().unwrap().to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
                    return ProverResult::Failure;
                }
                let public_written = FileWrapper::save_json_file(public_path, &public_signals);
                if public_written.is_err() {
                    string_to_ffi_buf(public_written.err().unwrap().to_string().as_str(), error_msg, error_msg_maxsize).unwrap();
                    return ProverResult::Failure;
                }
                ProverResult::Success
            },
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

        println!("Verification successful");

        VerifierResult::Success
    }
}
