use crate::{
    utils::{from_u8, serialize_g1_affine, serialize_g2_affine}, 
    file_wrapper::FileWrapper, 
    icicle::{icicle_msm, icicle_ntt, set_device, icicle_initialize_domain}, ProjectiveG1, ProjectiveG2,
    F, G1, G2
};
use icicle_bn254::curve::ScalarField;
use icicle_core::{
    msm::{MSMConfig}, 
    traits::{FieldImpl, MontgomeryConvertible}, vec_ops::{mul_scalars, sub_scalars, VecOpsConfig}, ntt::{NTTConfig, release_domain}
};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice}, stream::IcicleStream, Device
};
use num_bigint::BigUint;
use serde::{Serialize, Deserialize};
use serde_json::Value;

use rayon::prelude::*;

use super::zkey::{ZKeyHeader as Groth16ZKeyHeader};
use crate::{
    zkey::{ZKey, ZKeyHeader, W},
    DeviceType, ProverResult
};

use std::time::Instant; // SP: measure runtime

#[cfg(not(feature = "no-randomness"))]
use icicle_bn254::curve::ScalarCfg;
#[cfg(not(feature = "no-randomness"))]
use icicle_core::traits::GenerateRandom;

#[derive(Serialize, Deserialize, Debug)]
pub struct Proof {
    pub pi_a: Vec<String>,
    pub pi_b: Vec<Vec<String>>,
    pub pi_c: Vec<String>,
    pub protocol: String,
    pub curve: String,
}

fn construct_r1cs(witness: &[ScalarField], zkey: &ZKey, header: &Groth16ZKeyHeader, stream: &IcicleStream) -> DeviceVec<ScalarField> {
    let mut cfg = VecOpsConfig::default();
    cfg.is_async = true;
    cfg.stream_handle = stream.handle;

    // ------------------------------------------------------------
    let calc_wire_vals = || {
        let buff_coeffs = zkey.file.read_section(&zkey.sections[..], 4).unwrap();
        let s_coef = 4 * 3 + header.n8r;
        let n_coef = (buff_coeffs.len() - 4) / s_coef;

        let mut first_slice = Vec::with_capacity(n_coef);
        let mut second_slice = Vec::with_capacity(n_coef);
        let mut c_values = Vec::with_capacity(n_coef);
        let mut m_values = Vec::with_capacity(n_coef);

        unsafe {
            first_slice.set_len(n_coef);
            second_slice.set_len(n_coef);
            c_values.set_len(n_coef);
            m_values.set_len(n_coef);
        }
        
        let n8 = 32;
        second_slice
            .par_iter_mut()
            .zip(c_values.par_iter_mut())
            .zip(m_values.par_iter_mut())
            .zip(first_slice.par_iter_mut())
            .enumerate()
            .for_each(|(i, (((witness_val, c_val), m_val), coef_val))| {
                let start = 4 + i * s_coef;
                let buff_coef = &buff_coeffs[start..start + s_coef];

                let s =
                    u32::from_le_bytes([buff_coef[8], buff_coef[9], buff_coef[10], buff_coef[11]])
                        as usize;
                let c = u32::from_le_bytes([buff_coef[4], buff_coef[5], buff_coef[6], buff_coef[7]])
                    as usize;
                let m = buff_coef[0];
                let coef = ScalarField::from_bytes_le(&buff_coef[12..12 + n8]);

                *witness_val = witness[s];
                *c_val = c;
                *m_val = m as usize;
                *coef_val = coef;
            });

        let mut d_first_slice = DeviceVec::device_malloc_async(first_slice.len(), &stream).unwrap();
        d_first_slice
            .copy_from_host_async(HostSlice::from_slice(&first_slice), &stream)
            .unwrap();

        ScalarField::from_mont(&mut d_first_slice, &stream);
        
        let mut d_second_slice = DeviceVec::device_malloc_async(n_coef, &stream).unwrap();
        d_second_slice
            .copy_from_host_async(HostSlice::from_slice(&second_slice), &stream)
            .unwrap();
        ScalarField::from_mont(&mut d_second_slice, &stream);
        
        let mut res = Vec::with_capacity(n_coef);
        unsafe {
            res.set_len(n_coef);
        }
        let res_slice = HostSlice::from_mut_slice(&mut res);
        mul_scalars(&d_first_slice[..], &d_second_slice, res_slice, &cfg).unwrap();
        
        stream.synchronize().unwrap();

        (res, c_values, m_values)
    };
    
    let (res, c_values, m_values) = calc_wire_vals();

    let nof_coef = header.domain_size;
    let zero_scalar = ScalarField::zero();
    let mut out_buff_b_a = vec![ScalarField::zero(); nof_coef * 2];

    for i in 0..res.len() {
        let c = c_values[i];
        let m = m_values[i];
        let idx = c + m * nof_coef;
        let value = &mut out_buff_b_a[idx];

        if zero_scalar.eq(value) {
            *value = res[i];
        } else if !res[i].eq(&zero_scalar) {
            *value = *value + res[i];
        }
    }

    let mut d_vec = DeviceVec::device_malloc_async(nof_coef * 3, &stream).unwrap();

    d_vec[0..nof_coef]
        .copy_from_host_async(HostSlice::from_slice(&out_buff_b_a[nof_coef..]), &stream)
        .unwrap();
    d_vec[nof_coef..nof_coef * 2]
        .copy_from_host_async(HostSlice::from_slice(&out_buff_b_a[..nof_coef]), &stream)
        .unwrap();

    let d_vec_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            d_vec.as_mut_ptr(),
            d_vec.len(),
        ))
    };

    mul_scalars(&d_vec[0..nof_coef], &d_vec[nof_coef..nof_coef * 2], &mut d_vec_copy[2 * nof_coef..], &cfg).unwrap();

    d_vec
}

fn compute_h(d_vec: &mut DeviceVec<ScalarField>, coset_gen: Option<ScalarField>, nof_coef: usize, keys: Option<&Vec<F>>, stream: &IcicleStream) -> DeviceVec<ScalarField> {
    let mut ntt_cfg = NTTConfig::default();
    ntt_cfg.stream_handle = stream.handle;
    ntt_cfg.is_async = true;
    ntt_cfg.batch_size = 3;
    icicle_ntt(d_vec, true, &ntt_cfg);

    let d_vec_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            d_vec.as_mut_ptr(),
            d_vec.len(),
        ))
    };

    let mut cfg: VecOpsConfig = VecOpsConfig::default();
    cfg.stream_handle = stream.handle;
    if let Some(keys) = keys {
        let mut d_keys = DeviceVec::device_malloc_async(keys.len(), &stream).unwrap();
        d_keys
            .copy_from_host_async(HostSlice::from_slice(&keys), &stream)
            .unwrap();


        // ntt_cfg.coset_gen = coset_gen;

        mul_scalars(
            &d_vec[..nof_coef],
            &d_keys[..],
            &mut d_vec_copy[..nof_coef],
            &cfg,
        )
        .unwrap();
        mul_scalars(
            &d_vec[nof_coef..nof_coef * 2],
            &d_keys[..],
            &mut d_vec_copy[nof_coef..2 * nof_coef],
            &cfg,
        )
        .unwrap();
        mul_scalars(
            &d_vec[nof_coef * 2..],
            &d_keys[..],
            &mut d_vec_copy[2 * nof_coef..],
            &cfg,
        )
        .unwrap();
    } else if let Some(coset_gen) = coset_gen {
        ntt_cfg.coset_gen = coset_gen;
    } else {
        panic!("[compute_h]: Neither coset_gen nor keys provided");
    }
    icicle_ntt(d_vec, false, &ntt_cfg);

    // L * R - O
    mul_scalars(&d_vec[0..nof_coef], &d_vec[nof_coef..nof_coef * 2], &mut d_vec_copy[0..nof_coef], &cfg).unwrap();
    let mut d_h = DeviceVec::device_malloc(nof_coef).unwrap();
    sub_scalars(&d_vec[0..nof_coef], &d_vec[2 * nof_coef..], &mut d_h, &cfg).unwrap();
    let _ = release_domain::<ScalarField>();
    stream.synchronize().unwrap();
    d_h
}

fn commit_g1(d_scalars: &(impl HostOrDeviceSlice<ScalarField> + ?Sized), zkey: &ZKey, section_idx: usize, label: &str, commit_config: &MSMConfig, stream: &IcicleStream) -> ProjectiveG1 {
    let points_raw = from_u8(&zkey.file.read_section(&zkey.sections, section_idx).unwrap());
    let points = HostSlice::from_slice(&points_raw);
    let mut d_points = DeviceVec::device_malloc_async(points.len(), &stream).unwrap();
    d_points.copy_from_host_async(points, &stream).unwrap();
    G1::from_mont(&mut d_points, &stream);
    stream.synchronize().unwrap();

    icicle_msm(d_scalars, &d_points, commit_config, label)
    // stream.synchronize().unwrap();
}

fn commit_g2(d_scalars: &(impl HostOrDeviceSlice<ScalarField> + ?Sized), zkey: &ZKey, section_idx: usize, label: &str, commit_config: &MSMConfig, stream: &IcicleStream) -> ProjectiveG2 {
    let points_raw = from_u8(&zkey.file.read_section(&zkey.sections, section_idx).unwrap());
    let points = HostSlice::from_slice(&points_raw);
    let mut d_points = DeviceVec::device_malloc_async(points.len(), &stream).unwrap();
    d_points.copy_from_host_async(points, &stream).unwrap();
    G2::from_mont(&mut d_points, &stream);
    stream.synchronize().unwrap();
    icicle_msm(d_scalars, &d_points, commit_config, label)
    // stream.synchronize().unwrap();
}

fn commitments(scalars: &[F], zkey: &ZKey, n_public: usize, stream: &IcicleStream) -> (ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1) {
    let host_scalars = HostSlice::from_slice(scalars);
    let mut msm_config = MSMConfig::default();
    msm_config.is_async = false;
    msm_config.c = 14;
    
    let a = commit_g1(&host_scalars[..], zkey, 5, "a", &msm_config, stream);
    let b1 = commit_g1(&host_scalars[..], zkey, 6, "b1", &msm_config, stream);
    let b = commit_g2(&host_scalars[..], zkey, 7, "b", &msm_config, stream);
    let c = commit_g1(&host_scalars[n_public+1..], zkey, 8, "c", &msm_config, stream);

    (a, b1, b, c)
}

fn prove_cpu(scalars: &[F], zkey: &ZKey, header: &Groth16ZKeyHeader) -> (ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1) {
    let coset_gen = F::from_hex(W[header.power + 1]);
    let stream = IcicleStream::create().unwrap();
    let (pi_a, pi_b1, pi_b, pi_c) = commitments(scalars, zkey, header.n_public, &stream);
 
    let mut d_vec = construct_r1cs(scalars, zkey, header, &stream);
    let d_h = compute_h(&mut d_vec, Some(coset_gen), header.domain_size, None, &stream);
    let mut msm_config = MSMConfig::default();
    msm_config.is_async = false;
    msm_config.c = 14;
    let pi_h = commit_g1(&d_h, zkey, 9, "h", &msm_config, &stream);
    stream.synchronize().unwrap();

    (pi_a, pi_b1, pi_b, pi_c, pi_h)
}

fn prove_metal_cpu(scalars: &[F], zkey: &ZKey, header: &Groth16ZKeyHeader) -> (ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1) {
    std::thread::scope(|s| {
        let cpu_thread = s.spawn(|| {
            let device = Device::new("CPU", 0);
            icicle_runtime::set_device(&device).unwrap();
            let cpu_stream = IcicleStream::create().unwrap();
            commitments(scalars, zkey, header.n_public, &cpu_stream)
        });

        let domain_size = header.domain_size;
        let keys = super::compute_keys(F::one(), F::from_hex(W[header.power + 1]), domain_size).unwrap();
        let mut stream = IcicleStream::create().unwrap();
        let mut d_vec = construct_r1cs(scalars, zkey, header, &stream);
        // Arbitrary coset is not supported in METAL yet
        let d_h = compute_h(&mut d_vec, None, domain_size, Some(&keys), &stream); 
        

        let mut msm_config = MSMConfig::default();
        msm_config.stream_handle = stream.handle;
        msm_config.is_async = false;
        let pi_h = commit_g1(&d_h, zkey, 9, "h", &msm_config, &stream);

        let (pi_a, pi_b1, pi_b, pi_c) = cpu_thread.join().unwrap();
        stream.synchronize().unwrap();
        stream.destroy().unwrap();
        (pi_a, pi_b1, pi_b, pi_c, pi_h)
    })
}

fn prove_metal(scalars: &[F], zkey: &ZKey, header: &Groth16ZKeyHeader) -> (ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1) {
    // TODO: Implement full metal proof
    prove_metal_cpu(scalars, zkey, header)
}

pub fn prove(
    witness: &str,
    zkey: &ZKey,
    device_type: DeviceType,
) -> Result<(Value, Value), Box<dyn std::error::Error>> {
    
    let start = Instant::now(); // SP: measure runtime
    let (mut wtns_file, sections_wtns) = FileWrapper::read_bin_file(witness, "wtns", 2).unwrap();
    let wtns = wtns_file.read_wtns_header(&sections_wtns[..]).unwrap();
    
    let ZKeyHeader::Groth16(header) = &zkey.header;
    
    if !F::eq(&header.r, &wtns.q) {
        panic!("Curve of the witness does not match the curve of the proving key");
    }
    
    if wtns.n_witness != header.n_vars {
        panic!(
            "Invalid witness length. Circuit: {}, witness: {}",
            header.n_vars, wtns.n_witness
        );
    }
    
    let buff_witness = wtns_file.read_section(&sections_wtns[..], 2).unwrap();
    let scalars = from_u8(buff_witness);
    
    set_device(device_type);
    icicle_initialize_domain(header.domain_size as u64);

    let (pi_a, pi_b1, pi_b, pi_c, pi_h) = match device_type {
        DeviceType::Cpu => {
            prove_cpu(scalars, zkey, header)
        }
        DeviceType::CpuMetal => {
            prove_metal_cpu(scalars, zkey, header)
        }
        DeviceType::Metal => {
            prove_metal(scalars, zkey, header)
        }
    };

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

    let mut public_signals = Vec::with_capacity(header.n_public);
    let field_size = ScalarField::zero().to_bytes_le().len();

    for i in 1..=header.n_public {
        let start = i * field_size;
        let end = start + field_size;
        let b = &buff_witness[start..end];
        let scalar_bytes: BigUint = BigUint::from_bytes_le(b);
        public_signals.push(scalar_bytes.to_str_radix(10));
    }

    let proof = Proof {
        pi_a: serialize_g1_affine(pi_a.into()),
        pi_b: serialize_g2_affine(pi_b.into()),
        pi_c: serialize_g1_affine(pi_c.into()),
        protocol: "groth16".to_string(),
        curve: "bn128".to_string(),
    };
    println!("proof took: {:?}", start.elapsed()); // SP: measure runtime
    Ok((serde_json::json!(proof), serde_json::json!(public_signals)))
}

pub fn parallel_prove(
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


