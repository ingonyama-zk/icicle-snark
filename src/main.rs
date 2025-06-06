use icicle_snark::{groth16_prove, groth16_verify, CacheManager};
use std::io::{self, BufRead, Write};

enum ProofSystem {
    Groth16,
}

enum Command {
    Prove {
        system: ProofSystem,
        witness: String,
        zkey: String,
        proof: String,
        public: String,
        device: String,
    },
    Verify {
        system: ProofSystem,
        proof: String,
        public: String,
        vk: String,
    }
}

impl Command {
    fn print_help() {
        println!(
            "Usage: prove [OPTIONS]\n\n\
            Options:\n\
            --system <system>   Set the proof system (default: Groth16)\n\
            --witness <path>    Path to the witness file\n\
            --zkey <path>       Path to the zkey file\n\
            --proof <path>      Path to the proof output file\n\
            --public <path>     Path to the public output file\n\
            --device <device>   Set the computation device (default: CUDA)\n\
            --help              Show this message and exit"
        );
    }
    fn parse_command(command: &str) -> Option<Self> {
        let mut parts = command.split_whitespace();
        let command_type = parts.next()?;
        let mut proof_system = ProofSystem::Groth16;

        match command_type {
            "prove" => {
                let mut witness = "witness.wtns".to_string();
                let mut zkey = "circuit_final.zkey".to_string();
                let mut proof = "proof.json".to_string();
                let mut public = "public.json".to_string();
                let mut device = "CUDA".to_string();

                while let Some(arg) = parts.next() {
                    match arg {
                        "--system" => {
                            if let Some(val) = parts.next() {
                                proof_system = match val.to_lowercase().as_str() {
                                    "groth16" => ProofSystem::Groth16,
                                    _ => {
                                        eprintln!("Unknown proof system: {}", val);
                                        return None;
                                    }
                                };
                            }
                        }
                        "--witness" => witness = parts.next()?.to_string(),
                        "--zkey" => zkey = parts.next()?.to_string(),
                        "--proof" => proof = parts.next()?.to_string(),
                        "--public" => public = parts.next()?.to_string(),
                        "--device" => device = parts.next()?.to_string(),
                        _ => Command::print_help(),
                    }
                }

                Some(Command::Prove {
                    system: proof_system,
                    witness,
                    zkey,
                    proof,
                    public,
                    device,
                })
            }
            "verify" => {
                let mut proof = "proof.json".to_string();
                let mut public = "public.json".to_string();
                let mut vk = "verification_key.json".to_string();
                let mut system = ProofSystem::Groth16;

                while let Some(arg) = parts.next() {
                    match arg {
                        "--system" => {
                            if let Some(val) = parts.next() {
                                system = match val.to_lowercase().as_str() {
                                    "groth16" => ProofSystem::Groth16,
                                    _ => {
                                        eprintln!("Unknown proof system: {}", val);
                                        return None;
                                    }
                                };
                            }
                        }
                        "--proof" => proof = parts.next()?.to_string(),
                        "--public" => public = parts.next()?.to_string(),
                        "--vk" => vk = parts.next()?.to_string(),
                        _ => Command::print_help(),
                    }
                }

                Some(Command::Verify {
                    system,
                    proof,
                    public,
                    vk,
                })
            }
            _ => None,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();
    let mut cache_manager = CacheManager::default();

    loop {
        print!("> ");
        io::stdout().flush()?;
        input.clear();
        handle.read_line(&mut input)?;

        let command = input.trim();

        if command.is_empty() {
            println!("COMMAND_EMPTY");
            println!("COMMAND_COMPLETED");
            continue;
        }

        if command.eq_ignore_ascii_case("exit") {
            println!("COMMAND_EXIT");
            println!("COMMAND_COMPLETED");
            break;
        }

        match Command::parse_command(command) {
            Some(Command::Prove {
                system,
                witness,
                zkey,
                proof,
                public,
                device,
            }) => {
                match system {
                    ProofSystem::Groth16 => groth16_prove(
                        &witness,
                        &zkey,
                        &proof,
                        &public,
                        &device,
                        &mut cache_manager,
                    )
                    .unwrap(),
                }
                println!("COMMAND_COMPLETED");
            }
            Some(Command::Verify { system, proof, public, vk }) => {
                match system {
                    ProofSystem::Groth16 => groth16_verify(
                        &proof,
                        &public,
                        &vk,
                    )
                    .unwrap(),
                }
                println!("COMMAND_COMPLETED");
            }
            None => Command::print_help(),
        }
    }

    println!("Exiting CLI worker...");
    Ok(())
}
