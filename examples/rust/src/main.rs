use icicle_snark::{groth16_prove, CacheManager};
use std::time::{Duration, Instant};
const NUMBER_OF_WARMUP: usize = 0;
const NUMBER_OF_ITERATIONS: usize = 1;

fn main() {
    let mut cache_manager = CacheManager::default();
    let base_path = "../../benchmark/bionet/";
    let mut results = Vec::new();

    let witness = format!("{}witness.wtns", base_path);
    let zkey = format!("{}circuit_final.zkey", base_path);
    let device = "CPU";

    let mut durations = Vec::new();

    for i in 0..NUMBER_OF_ITERATIONS {
        let start = Instant::now();
        
        groth16_prove(&witness, &zkey, device, &mut cache_manager).unwrap();
        
        let duration = start.elapsed();

        if i >= NUMBER_OF_WARMUP {
            durations.push(duration);
        }
    }
    
    if !durations.is_empty() {
        let avg_time = durations.iter().sum::<Duration>() / durations.len() as u32;
        println!("Average running time for {} on {}: {:?}", base_path, device, avg_time);
        results.push(format!("{}: {:?} for {}", device, avg_time, base_path));
    } else {
        println!("No valid measurements for {} on {}", base_path, device);
    }
}