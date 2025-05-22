mod cache;
mod conversions;
mod file_wrapper;
mod icicle_helper;
mod proof_helper;
mod zkey;

pub use cache::{CacheManager, ZKeyCache};
use circom_witnesscalc::calc_witness;
use file_wrapper::FileWrapper;
use icicle_bn254::curve::{CurveCfg, G2CurveCfg, ScalarField};
use icicle_core::curve::{Affine, Projective};
use proof_helper::groth16_prove_helper;
use std::time::Instant;
use std::fs::File;
use std::io::Write;

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

pub fn groth16_witness(
    input: &str,
    graph: &str,
    witness: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_data = std::fs::read_to_string(&input)
    .expect("Failed to read input file");

    let graph_data = std::fs::read(&graph).expect("Failed to read graph file");

    let start = Instant::now();

    let wtns_bytes = calc_witness(&input_data, &graph_data).unwrap();

    let duration = start.elapsed();
    println!("Witness generated in: {:?}", duration);

    {
        let mut f = File::create(&witness).unwrap();
        f.write_all(&wtns_bytes).unwrap();
    }

    println!("witness saved to {}", &witness);
    Ok(())
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
