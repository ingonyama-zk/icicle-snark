pub fn parallel_prove_old(
    witness_paths: &[String],
    zkey_path: &str,
    proof_paths: &[String],
    public_paths: &[String],
    device_type: DeviceType,
) -> Result<Vec<ProverResult>, Box<dyn std::error::Error>> {
    if witness_paths.len() != proof_paths.len() || 
       proof_paths.len() != public_paths.len() {
        return Err("All input arrays must have the same length".into());
    }

    println!("Parallel proof generation started");
    println!("Witness paths: {:?}", witness_paths);
    println!("Proof paths: {:?}", proof_paths);
    println!("Public paths: {:?}", public_paths);
    println!("Zkey path: {:?}", zkey_path);
    println!("Device type: {:?}", device_type);

    // Load zkey once for all proofs
    let zkey = match ZKey::load(zkey_path) {
        Ok(zkey) => zkey,
        Err(_) => return Err("Failed to load zkey file".into()),
    };

    let results: Vec<ProverResult> = witness_paths
        .par_iter()
        .zip(proof_paths.par_iter())
        .zip(public_paths.par_iter())
        .map(|((witness_path, proof_path), public_path)| {

            // Load witness
            let (mut wtns_file, sections_wtns) = match FileWrapper::read_bin_file(witness_path, "wtns", 2) {
                Ok(result) => result,
                Err(_) => return ProverResult::Failure,
            };

            let wtns = match wtns_file.read_wtns_header(&sections_wtns[..]) {
                Ok(wtns) => wtns,
                Err(_) => return ProverResult::Failure,
            };

            let ZKeyHeader::Groth16(header) = &zkey.header;
            
            if !F::eq(&header.r, &wtns.q) {
                return ProverResult::Failure;
            }
            
            if wtns.n_witness != header.n_vars {
                return ProverResult::Failure;
            }
            
            let buff_witness = match wtns_file.read_section(&sections_wtns[..], 2) {
                Ok(buff) => buff,
                Err(_) => return ProverResult::Failure,
            };
            let scalars = from_u8(buff_witness);
            
            // Set device and initialize domain for this thread
            set_device(device_type);
            icicle_initialize_domain(header.domain_size as u64);

            // Generate proof
            let prove_result = match device_type {
                DeviceType::Cpu => {
                    prove_cpu(scalars, &zkey, &header)
                }
                DeviceType::CpuMetal => {
                    prove_metal_cpu(scalars, &zkey, &header)
                }
                DeviceType::Metal => {
                    prove_metal(scalars, &zkey, &header)
                }
            };

            let (pi_a, pi_b1, pi_b, pi_c, pi_h) = prove_result;

            #[cfg(not(feature = "no-randomness"))]
            let (pi_a, pi_b, pi_c) = {
                let rs = ScalarCfg::generate_random(2);
                let r = rs[0];
                let s = rs[1];

                let pi_a = pi_a + header.vk_alpha_1 + header.vk_delta_1 * r;
                let pi_b = pi_b + header.vk_beta_2 + header.vk_delta_2 * s;
                let pi_b1 = pi_b1 + header.vk_beta_1 + header.vk_delta_1 * s;
                let pi_c = pi_c + pi_h + pi_a * s + pi_b1 * r - header.vk_delta_1 * r * s;

                (pi_a, pi_b, pi_c)
            };
            #[cfg(feature = "no-randomness")]
            let (pi_a, pi_b, pi_c) = {
                let pi_a = pi_a + zkey.vk_alpha_1 + zkey.vk_delta_1;
                let pi_b = pi_b + zkey.vk_beta_2 + zkey.vk_delta_2;
                let pi_b1 = pi_b1 + zkey.vk_beta_1 + zkey.vk_delta_1;
                let pi_c = pi_c + pi_h + pi_a + pi_b1 - zkey.vk_delta_1;

                (pi_a, pi_b, pi_c)
            };

            // Extract public signals
            let mut public_signals = Vec::with_capacity(header.n_public);
            let field_size = ScalarField::zero().to_bytes_le().len();

            for i in 1..=header.n_public {
                let start = i * field_size;
                let end = start + field_size;
                let b = &buff_witness[start..end];
                let scalar_bytes: BigUint = BigUint::from_bytes_le(b);
                public_signals.push(scalar_bytes.to_str_radix(10));
            }

            // Create proof
            let proof = Proof {
                pi_a: serialize_g1_affine(pi_a.into()),
                pi_b: serialize_g2_affine(pi_b.into()),
                pi_c: serialize_g1_affine(pi_c.into()),
                protocol: "groth16".to_string(),
                curve: "bn128".to_string(),
            };

            // Save proof and public signals
            if FileWrapper::save_json_file(proof_path, &proof).is_err() {
                return ProverResult::Failure;
            }
            if FileWrapper::save_json_file(public_path, &public_signals).is_err() {
                return ProverResult::Failure;
            }

            ProverResult::Success
        })
        .collect();

    Ok(results)
}
