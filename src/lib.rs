mod cache;
mod conversions;
mod file_wrapper;
mod icicle_helper;
mod proof_helper;
mod zkey;

use cache::VerificationKey;
pub use cache::{CacheManager, ZKeyCache};
use file_wrapper::FileWrapper;
use icicle_bn254::curve::{CurveCfg, G2CurveCfg, ScalarField};
use icicle_core::curve::{Affine, Projective};
use proof_helper::{groth16_prove_helper, groth16_verify_helper, Proof};
use std::time::Instant;
use serde_json;

pub type F = ScalarField;
pub type C1 = CurveCfg;
pub type C2 = G2CurveCfg;
pub type G1 = Affine<C1>;
pub type G2 = Affine<C2>;
pub type ProjectiveG1 = Projective<C1>;
pub type ProjectiveG2 = Projective<C2>;

fn try_load_and_set_backend_device(device_type: &str) {
    if device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    let device = icicle_runtime::Device::new(device_type, 0 /* =device_id*/);
    icicle_runtime::set_device(&device).unwrap();
}

pub fn groth16_prove(
    witness: &str,
    zkey: &str,
    proof: &str,
    public: &str,
    device: &str,
    cache_manager: &mut CacheManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    try_load_and_set_backend_device(device);

    let cache_key = format!("{}_{}", zkey, device);

    if !cache_manager.contains(&cache_key) {
        let computed_cache = cache_manager.compute(zkey)?;
        cache_manager.insert_cache(&cache_key, computed_cache);
    }

    let zkey_cache = cache_manager.get_cache(&cache_key);

    let (proof_data, public_signals) = groth16_prove_helper(witness, zkey_cache)?;

    FileWrapper::save_json_file(proof, &proof_data)?;
    FileWrapper::save_json_file(public, &public_signals)?;

    println!("proof took: {:?}", start.elapsed());

    Ok(())
}

pub fn groth16_verify(
    proof: &str,
    public: &str,
    vk: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let proof_str = std::fs::read_to_string(proof)?;
    let proof: Proof = serde_json::from_str(&proof_str)?;

    let public_str = std::fs::read_to_string(public)?;
    let public: Vec<String> = serde_json::from_str(&public_str)?;

    let vk_str = std::fs::read_to_string(vk)?;
    let vk: VerificationKey = serde_json::from_str(&vk_str)?;

    let pairing_result = groth16_verify_helper(&proof, &public, &vk)?;

    assert!(pairing_result, "Verification failed");

    Ok(())
}
