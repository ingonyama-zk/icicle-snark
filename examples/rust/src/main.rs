use icicle_snark::{prove, verify};
use std::time::{Duration, Instant};


fn main() {
    let base_path = "../../benchmark/rsa/";
    let mut results = Vec::new();

    let witness = format!("{}witness.wtns", base_path);
    let zkey = format!("{}circuit_final.zkey", base_path);
    let proof = format!("{}proof.json", base_path);
    let public = format!("{}public.json", base_path);
    let vk = format!("{}verification_key.json", base_path);
    let device = "CUDA";


    let start = Instant::now();
    
    prove(&witness, &zkey, device).unwrap();
    println!("prove took: {:?}", start.elapsed());

    // let start = Instant::now();
    // groth16_verify(&proof, &public, &vk).unwrap();
    // println!("verification took: {:?}", start.elapsed());
}