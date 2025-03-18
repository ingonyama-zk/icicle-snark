use icicle_snark::{groth16_prove, CacheManager};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

const NUMBER_OF_WARMUP: usize = 2;
const NUMBER_OF_ITERATIONS: usize = 5;

fn main() {
    let mut cache_manager = CacheManager::default();
    let base_path = "../../benchmark/rsa/";
    let mut results = Vec::new();

    let witness = format!("{}witness.wtns", base_path);
    let zkey = format!("{}circuit_final.zkey", base_path);
    let proof = format!("{}proof.json", base_path);
    let public = format!("{}public.json", base_path);
    let device = "CUDA";

    let mut durations = Vec::new();

    for i in 0..NUMBER_OF_ITERATIONS {
        let start = Instant::now();

        groth16_prove(&witness, &zkey, &proof, &public, device, &mut cache_manager).unwrap();

        let duration = start.elapsed();
        verify(base_path);

        if i >= NUMBER_OF_WARMUP {
            durations.push(duration);
        }
    }

    if !durations.is_empty() {
        let avg_time = durations.iter().sum::<Duration>() / durations.len() as u32;
        println!(
            "Average running time for {} on {}: {:?}",
            base_path, device, avg_time
        );
        results.push(format!("{}: {:?} for {}", device, avg_time, base_path));
    } else {
        println!("No valid measurements for {} on {}", base_path, device);
    }
}

fn verify(base_path: &str) {
    let output = Command::new("snarkjs")
        .arg("g16v")
        .current_dir(Path::new(base_path))
        .output()
        .expect("Failed to execute snarkjs");

    if !output.stdout.is_empty() {
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    }

    if !output.stderr.is_empty() {
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    }
}
