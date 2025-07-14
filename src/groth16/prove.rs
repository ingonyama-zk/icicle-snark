use crate::{
    utils::{from_u8, serialize_g1_affine, serialize_g2_affine}, 
    file_wrapper::FileWrapper, 
    icicle::{icicle_msm, icicle_ntt, set_device, icicle_initialize_domain}, ProjectiveG1, ProjectiveG2,
    F, G1, G2
};
use icicle_bn254::curve::ScalarField;
use icicle_core::{
    msm::{MSMConfig, msm}, 
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

use std::time::Instant;

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

fn compute_h_batched(
    d_vecs: &mut [DeviceVec<ScalarField>], 
    coset_gen: Option<ScalarField>, 
    nof_coef: usize, 
    keys: Option<&Vec<F>>, 
    stream: &IcicleStream,
    batch_size: usize
) -> Vec<DeviceVec<ScalarField>> {
    
    // Verify all d_vecs have the same size
    let expected_size = d_vecs[0].len();
    for (i, d_vec) in d_vecs.iter().enumerate() {
        if d_vec.len() != expected_size {
            panic!("d_vecs[{}] has size {} but expected {}", i, d_vec.len(), expected_size);
        }
    }
    
    debug_println!("compute_h_batched: batch_size={}, individual_size={}, total_size={}, domain_size={}", 
             batch_size, expected_size, expected_size * batch_size, nof_coef);
    
    // Concatenate all d_vecs into a single batch for NTT
    let total_size = expected_size * batch_size;
    let mut batched_d_vec = DeviceVec::device_malloc_async(total_size, stream).unwrap();
    
    // Copy all d_vecs into the batched array
    for (i, d_vec) in d_vecs.iter().enumerate() {
        let start_idx = i * d_vec.len();
        let end_idx = start_idx + d_vec.len();
        batched_d_vec[start_idx..end_idx].copy(&d_vec[..]).unwrap();
    }
    
    // Configure NTT for batch processing
    let mut ntt_cfg = NTTConfig::default();
    ntt_cfg.stream_handle = stream.handle;
    ntt_cfg.is_async = true;
    ntt_cfg.batch_size = (3 * batch_size) as i32; // 3 internal sections per vector * number of vectors
    ntt_cfg.columns_batch = false; // Process as separate NTTs, not matrix columns
    
    // Forward NTT (inverse = true)
    icicle_ntt(&mut batched_d_vec, true, &ntt_cfg);

    let batched_d_vec_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            batched_d_vec.as_mut_ptr(),
            batched_d_vec.len(),
        ))
    };

    let mut cfg: VecOpsConfig = VecOpsConfig::default();
    cfg.stream_handle = stream.handle;
    
    if let Some(keys) = keys {
        let mut d_keys = DeviceVec::device_malloc_async(keys.len(), stream).unwrap();
        d_keys.copy_from_host_async(HostSlice::from_slice(&keys), stream).unwrap();

        // Apply keys to each batch element
        for batch_idx in 0..batch_size {
            let start_idx = batch_idx * d_vecs[0].len();
            let end_idx = start_idx + d_vecs[0].len();
            
            mul_scalars(
                &batched_d_vec[start_idx..start_idx + nof_coef],
                &d_keys[..],
                &mut batched_d_vec_copy[start_idx..start_idx + nof_coef],
                &cfg,
            ).unwrap();
            mul_scalars(
                &batched_d_vec[start_idx + nof_coef..start_idx + nof_coef * 2],
                &d_keys[..],
                &mut batched_d_vec_copy[start_idx + nof_coef..start_idx + 2 * nof_coef],
                &cfg,
            ).unwrap();
            mul_scalars(
                &batched_d_vec[start_idx + nof_coef * 2..end_idx],
                &d_keys[..],
                &mut batched_d_vec_copy[start_idx + nof_coef * 2..end_idx],
                &cfg,
            ).unwrap();
        }
    } else if let Some(coset_gen) = coset_gen {
        ntt_cfg.coset_gen = coset_gen;
    } else {
        panic!("[compute_h_batched]: Neither coset_gen nor keys provided");
    }
    
    // Inverse NTT (inverse = false)
    icicle_ntt(&mut batched_d_vec, false, &ntt_cfg);

    // Process each batch element: L * R - O
    let mut d_h_results = Vec::with_capacity(batch_size);
    for batch_idx in 0..batch_size {
        let start_idx = batch_idx * d_vecs[0].len();
        
        mul_scalars(
            &batched_d_vec[start_idx..start_idx + nof_coef], 
            &batched_d_vec[start_idx + nof_coef..start_idx + nof_coef * 2], 
            &mut batched_d_vec_copy[start_idx..start_idx + nof_coef], 
            &cfg
        ).unwrap();
        
        let mut d_h = DeviceVec::device_malloc(nof_coef).unwrap();
        sub_scalars(
            &batched_d_vec[start_idx..start_idx + nof_coef], 
            &batched_d_vec[start_idx + nof_coef * 2..start_idx + nof_coef * 3], 
            &mut d_h, 
            &cfg
        ).unwrap();
        d_h_results.push(d_h);
    }
    
    stream.synchronize().unwrap();
    d_h_results
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

fn commitments_batched(
    scalars_array: &[Vec<F>], 
    zkey: &ZKey, 
    n_public: usize, 
    stream: &IcicleStream,
    batch_size: usize
) -> Vec<(ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1)> {
    
    debug_println!("commitments_batched: batch_size={}, n_public={}", batch_size, n_public);
    debug_println!("commitments_batched: scalars_array lengths: {:?}", scalars_array.iter().map(|s| s.len()).collect::<Vec<_>>());
    
    // Configure MSM for batch processing
    let mut msm_config = MSMConfig::default();
    msm_config.is_async = false;
    msm_config.c = 14;
    msm_config.batch_size = batch_size as i32;
    msm_config.are_points_shared_in_batch = true; // All proofs use the same zkey
    
    // Concatenate all scalars for batch MSM
    let total_scalars: Vec<F> = scalars_array.iter().flat_map(|s| s.iter()).cloned().collect();
    let host_scalars = HostSlice::from_slice(&total_scalars);
    
    debug_println!("commitments_batched: total_scalars length: {}", total_scalars.len());
    
    // Batch commit for pi_a
    debug_println!("commitments_batched: Starting pi_a MSM");
    let mut pi_a_results = vec![ProjectiveG1::zero(); batch_size];
    let pi_a_slice = HostSlice::from_mut_slice(&mut pi_a_results);
    commit_g1_batched(&host_scalars[..], zkey, 5, "a", &msm_config, stream, pi_a_slice);
    debug_println!("commitments_batched: Completed pi_a MSM");
    
    // Batch commit for pi_b1
    debug_println!("commitments_batched: Starting pi_b1 MSM");
    let mut pi_b1_results = vec![ProjectiveG1::zero(); batch_size];
    let pi_b1_slice = HostSlice::from_mut_slice(&mut pi_b1_results);
    commit_g1_batched(&host_scalars[..], zkey, 6, "b1", &msm_config, stream, pi_b1_slice);
    debug_println!("commitments_batched: Completed pi_b1 MSM");
    
    // Batch commit for pi_b (G2)
    debug_println!("commitments_batched: Starting pi_b MSM");
    let mut pi_b_results = vec![ProjectiveG2::zero(); batch_size];
    let pi_b_slice = HostSlice::from_mut_slice(&mut pi_b_results);
    commit_g2_batched(&host_scalars[..], zkey, 7, "b", &msm_config, stream, pi_b_slice);
    debug_println!("commitments_batched: Completed pi_b MSM");
    
    // Batch commit for pi_c (only private inputs)
    let private_scalars: Vec<F> = scalars_array.iter()
        .flat_map(|s| s[n_public+1..].iter())
        .cloned()
        .collect();
    let host_private_scalars = HostSlice::from_slice(&private_scalars);
    debug_println!("commitments_batched: private_scalars length: {}", private_scalars.len());
    debug_println!("commitments_batched: Starting pi_c MSM");
    let mut pi_c_results = vec![ProjectiveG1::zero(); batch_size];
    let pi_c_slice = HostSlice::from_mut_slice(&mut pi_c_results);
    commit_g1_batched(&host_private_scalars[..], zkey, 8, "c", &msm_config, stream, pi_c_slice);
    debug_println!("commitments_batched: Completed pi_c MSM");
    
    // Combine results
    (0..batch_size).map(|i| (
        pi_a_results[i],
        pi_b1_results[i], 
        pi_b_results[i],
        pi_c_results[i]
    )).collect()
}

fn commit_g1(d_scalars: &(impl HostOrDeviceSlice<ScalarField> + ?Sized), zkey: &ZKey, section_idx: usize, label: &str, commit_config: &MSMConfig, stream: &IcicleStream) -> ProjectiveG1 {
    let points_raw = from_u8(&zkey.file.read_section(&zkey.sections, section_idx).unwrap());
    let points = HostSlice::from_slice(&points_raw);
    let mut d_points = DeviceVec::device_malloc_async(points.len(), &stream).unwrap();
    d_points.copy_from_host_async(points, &stream).unwrap();
    G1::from_mont(&mut d_points, &stream);
    stream.synchronize().unwrap();

    icicle_msm(d_scalars, &d_points, commit_config, label)
}

fn commit_g2(d_scalars: &(impl HostOrDeviceSlice<ScalarField> + ?Sized), zkey: &ZKey, section_idx: usize, label: &str, commit_config: &MSMConfig, stream: &IcicleStream) -> ProjectiveG2 {
    let points_raw = from_u8(&zkey.file.read_section(&zkey.sections, section_idx).unwrap());
    let points = HostSlice::from_slice(&points_raw);
    let mut d_points = DeviceVec::device_malloc_async(points.len(), &stream).unwrap();
    d_points.copy_from_host_async(points, &stream).unwrap();
    G2::from_mont(&mut d_points, &stream);
    stream.synchronize().unwrap();
    icicle_msm(d_scalars, &d_points, commit_config, label)
}

fn commit_g1_batched(
    d_scalars: &(impl HostOrDeviceSlice<ScalarField> + ?Sized), 
    zkey: &ZKey, 
    section_idx: usize, 
    _label: &str, 
    commit_config: &MSMConfig, 
    stream: &IcicleStream,
    results: &mut (impl HostOrDeviceSlice<ProjectiveG1> + ?Sized)
) {
    debug_println!("commit_g1_batched: section_idx={}, d_scalars len={}, points len={}, batch_size={}", 
             section_idx, d_scalars.len(), commit_config.batch_size, commit_config.batch_size);
    
    let points_raw = from_u8(&zkey.file.read_section(&zkey.sections, section_idx).unwrap());
    let points = HostSlice::from_slice(&points_raw);
    let mut d_points = DeviceVec::device_malloc_async(points.len(), stream).unwrap();
    d_points.copy_from_host_async(points, stream).unwrap();
    G1::from_mont(&mut d_points, stream);
    stream.synchronize().unwrap();

    // Use Icicle's native batch MSM
    icicle_core::msm::msm(d_scalars, &d_points, commit_config, results).unwrap();
    debug_println!("commit_g1_batched: MSM completed for section {}", section_idx);
}

fn commit_g2_batched(
    d_scalars: &(impl HostOrDeviceSlice<ScalarField> + ?Sized), 
    zkey: &ZKey, 
    section_idx: usize, 
    _label: &str, 
    commit_config: &MSMConfig, 
    stream: &IcicleStream,
    results: &mut (impl HostOrDeviceSlice<ProjectiveG2> + ?Sized)
) {
    debug_println!("commit_g2_batched: section_idx={}, d_scalars len={}, points len={}, batch_size={}", 
             section_idx, d_scalars.len(), commit_config.batch_size, commit_config.batch_size);
    
    let points_raw = from_u8(&zkey.file.read_section(&zkey.sections, section_idx).unwrap());
    let points = HostSlice::from_slice(&points_raw);
    let mut d_points = DeviceVec::device_malloc_async(points.len(), stream).unwrap();
    d_points.copy_from_host_async(points, stream).unwrap();
    G2::from_mont(&mut d_points, stream);
    stream.synchronize().unwrap();

    // Use Icicle's native batch MSM
    icicle_core::msm::msm(d_scalars, &d_points, commit_config, results).unwrap();
    debug_println!("commit_g2_batched: MSM completed for section {}", section_idx);
}

fn prove_cpu(scalars: &[F], zkey: &ZKey, header: &Groth16ZKeyHeader) -> (ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1) {
    let coset_gen = F::from_hex(W[header.power + 1]);
    let mut stream = IcicleStream::create().unwrap();
    let (pi_a, pi_b1, pi_b, pi_c) = commitments(scalars, zkey, header.n_public, &stream);
 
    let mut d_vec = construct_r1cs(scalars, zkey, header, &stream);
    let d_h = compute_h(&mut d_vec, Some(coset_gen), header.domain_size, None, &stream);
    let mut msm_config = MSMConfig::default();
    msm_config.is_async = false;
    msm_config.c = 14;
    let pi_h = commit_g1(&d_h, zkey, 9, "h", &msm_config, &stream);
    stream.synchronize().unwrap();
    stream.destroy().unwrap();

    (pi_a, pi_b1, pi_b, pi_c, pi_h)
}

fn prove_metal_cpu(scalars: &[F], zkey: &ZKey, header: &Groth16ZKeyHeader) -> (ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1) {
    std::thread::scope(|s| {
        let cpu_thread = s.spawn(|| {
            let device = Device::new("CPU", 0);
            icicle_runtime::set_device(&device).unwrap();
            let mut cpu_stream = IcicleStream::create().unwrap();
            let result = commitments(scalars, zkey, header.n_public, &cpu_stream);
            cpu_stream.synchronize().unwrap();
            cpu_stream.destroy().unwrap();
            result
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

fn prove_cpu_batched(
    scalars_array: &[Vec<F>], 
    zkey: &ZKey, 
    header: &Groth16ZKeyHeader,
    batch_size: usize
) -> Vec<(ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1)> {
    let coset_gen = F::from_hex(W[header.power + 1]);
    let mut stream = IcicleStream::create().unwrap();
    
    // Batch commitments
    let commitments_results = commitments_batched(scalars_array, zkey, header.n_public, &stream, batch_size);
    
    // Batch R1CS construction and compute_h
    let mut d_vecs: Vec<DeviceVec<ScalarField>> = Vec::with_capacity(scalars_array.len());
    for scalars in scalars_array {
        let d_vec = construct_r1cs(scalars, zkey, header, &stream);
        d_vecs.push(d_vec);
    }
    
    let d_h_results = compute_h_batched(&mut d_vecs, Some(coset_gen), header.domain_size, None, &stream, batch_size);
    
    // Batch pi_h computation
    let mut msm_config = MSMConfig::default();
    msm_config.is_async = false;
    msm_config.c = 14;
    msm_config.batch_size = scalars_array.len() as i32;
    msm_config.are_points_shared_in_batch = true;
    
    let mut pi_h_results = vec![ProjectiveG1::zero(); scalars_array.len()];
    let pi_h_slice = HostSlice::from_mut_slice(&mut pi_h_results);
    
    // Concatenate all d_h results for batch MSM
    let total_d_h_size: usize = d_h_results.iter().map(|d_h| d_h.len()).sum();
    let mut batched_d_h = DeviceVec::device_malloc_async(total_d_h_size, &stream).unwrap();
    
    let mut offset = 0;
    for d_h in &d_h_results {
        let len = d_h.len();
        batched_d_h[offset..offset + len].copy(&d_h[..]).unwrap();
        offset += len;
    }
    
    commit_g1_batched(&batched_d_h, zkey, 9, "h", &msm_config, &stream, pi_h_slice);
    
    stream.synchronize().unwrap();
    stream.destroy().unwrap();

    // Combine all results
    commitments_results.into_iter()
        .zip(pi_h_results.into_iter())
        .map(|((pi_a, pi_b1, pi_b, pi_c), pi_h)| (pi_a, pi_b1, pi_b, pi_c, pi_h))
        .collect()
}

fn prove_metal_cpu_batched(
    scalars_array: &[Vec<F>], 
    zkey: &ZKey, 
    header: &Groth16ZKeyHeader,
    batch_size: usize
) -> Vec<(ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1)> {
    std::thread::scope(|s| {
        let cpu_thread = s.spawn(|| {
            let device = Device::new("CPU", 0);
            icicle_runtime::set_device(&device).unwrap();
            let mut cpu_stream = IcicleStream::create().unwrap();
            let result = commitments_batched(scalars_array, zkey, header.n_public, &cpu_stream, batch_size);
            cpu_stream.synchronize().unwrap();
            cpu_stream.destroy().unwrap();
            result
        });

        let domain_size = header.domain_size;
        let keys = super::compute_keys(F::one(), F::from_hex(W[header.power + 1]), domain_size).unwrap();
        let mut stream = IcicleStream::create().unwrap();
        
        // Batch R1CS construction and compute_h
        debug_println!("Building R1CS");
        let mut d_vecs: Vec<DeviceVec<ScalarField>> = Vec::with_capacity(scalars_array.len());
        debug_println!("scalars_array.len(): {}", scalars_array.len());
        for scalars in scalars_array {
            debug_println!("scalars.len(): {}", scalars.len());
            let d_vec = construct_r1cs(scalars, zkey, header, &stream);
            d_vecs.push(d_vec);
        }
        
        let d_h_results = compute_h_batched(&mut d_vecs, None, domain_size, Some(&keys), &stream, batch_size);

        let mut msm_config = MSMConfig::default();
        msm_config.stream_handle = stream.handle;
        msm_config.is_async = false;
        msm_config.batch_size = scalars_array.len() as i32;
        msm_config.are_points_shared_in_batch = true;
        
        let mut pi_h_results = vec![ProjectiveG1::zero(); scalars_array.len()];
        let pi_h_slice = HostSlice::from_mut_slice(&mut pi_h_results);
        
        // Concatenate all d_h results for batch MSM
        let total_d_h_size: usize = d_h_results.iter().map(|d_h| d_h.len()).sum();
        let mut batched_d_h = DeviceVec::device_malloc_async(total_d_h_size, &stream).unwrap();
        
        let mut offset = 0;
        for d_h in &d_h_results {
            let len = d_h.len();
            batched_d_h[offset..offset + len].copy(&d_h[..]).unwrap();
            offset += len;
        }
        
        commit_g1_batched(&batched_d_h, zkey, 9, "h", &msm_config, &stream, pi_h_slice);

        let commitments_results = cpu_thread.join().unwrap();
        stream.synchronize().unwrap();
        stream.destroy().unwrap();
        
        // Combine all results
        commitments_results.into_iter()
            .zip(pi_h_results.into_iter())
            .map(|((pi_a, pi_b1, pi_b, pi_c), pi_h)| (pi_a, pi_b1, pi_b, pi_c, pi_h))
            .collect()
    })
}

fn prove_metal_batched(
    scalars_array: &[Vec<F>], 
    zkey: &ZKey, 
    header: &Groth16ZKeyHeader,
    batch_size: usize
) -> Vec<(ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1)> {
    // TODO: Implement full metal proof
    prove_metal_cpu_batched(scalars_array, zkey, header, batch_size)
}

pub fn prove(
    witness: &str,
    zkey: &ZKey,
    device_type: DeviceType,
) -> Result<(Value, Value), Box<dyn std::error::Error>> {
    
    let start = Instant::now();
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
    let scalars = from_u8::<F>(buff_witness);
    
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
    debug_println!("proof took: {:?}", start.elapsed());
    Ok((serde_json::json!(proof), serde_json::json!(public_signals)))
}



pub fn parallel_prove(
    witness_paths: &[String],
    zkey_path: &str,
    proof_paths: &[String],
    public_paths: &[String],
    device_type: DeviceType,
    max_batch_size: Option<usize>,
) -> Result<Vec<ProverResult>, Box<dyn std::error::Error>> {
    let max_batch_size = max_batch_size.unwrap_or(10);
    debug_println!("[GROTH16] parallel_prove called with {} witness paths, max_batch_size: {}", witness_paths.len(), max_batch_size);
    
    if witness_paths.len() != proof_paths.len() || 
       proof_paths.len() != public_paths.len() {
                debug_println!("[GROTH16] ERROR: Array length mismatch - witness: {}, proof: {}, public: {}",
                 witness_paths.len(), proof_paths.len(), public_paths.len());
        return Err("All input arrays must have the same length".into());
    }

    debug_println!("[GROTH16] Batched proof generation started");
    debug_println!("[GROTH16] Witness paths: {:?}", witness_paths);
    debug_println!("[GROTH16] Proof paths: {:?}", proof_paths);
    debug_println!("[GROTH16] Public paths: {:?}", public_paths);
    debug_println!("[GROTH16] Zkey path: {:?}", zkey_path);
    debug_println!("[GROTH16] Device type: {:?}", device_type);
    debug_println!("[GROTH16] Total witnesses: {}, Max batch size: {}", witness_paths.len(), max_batch_size);

    // Load zkey once for all proofs
    let zkey = match ZKey::load(zkey_path) {
        Ok(zkey) => zkey,
        Err(_) => return Err("Failed to load zkey file".into()),
    };

    // Set device and initialize domain once for all proofs
    set_device(device_type);
    let ZKeyHeader::Groth16(header) = &zkey.header;
    icicle_initialize_domain(header.domain_size as u64);

    // Process witnesses in batches
    let mut all_results = Vec::with_capacity(witness_paths.len());
    let total_batches = (witness_paths.len() + max_batch_size - 1) / max_batch_size; // Ceiling division
    
        debug_println!("[GROTH16] Processing {} total witnesses in {} batches of max size {}",
             witness_paths.len(), total_batches, max_batch_size);

    for batch_idx in 0..total_batches {
        let start_idx = batch_idx * max_batch_size;
        let end_idx = std::cmp::min(start_idx + max_batch_size, witness_paths.len());
        let current_batch_size = end_idx - start_idx;
        
        debug_println!("[GROTH16] Processing batch {}/{} (witnesses {} to {})", 
                 batch_idx + 1, total_batches, start_idx + 1, end_idx);

        // Load witnesses for current batch
        let mut scalars_array: Vec<Vec<F>> = Vec::with_capacity(current_batch_size);
        
        for i in start_idx..end_idx {
            let witness_path = &witness_paths[i];
            debug_println!("[GROTH16] Loading witness {}: {}", i + 1, witness_path);
            
            let (mut wtns_file, sections_wtns) = match FileWrapper::read_bin_file(witness_path, "wtns", 2) {
                Ok(result) => result,
                Err(e) => {
                    debug_println!("[GROTH16] ERROR: Failed to read witness file {}: {:?}", i + 1, e);
                    // Fill remaining results with failures
                    all_results.extend(vec![ProverResult::FAILURE; witness_paths.len() - all_results.len()]);
                    return Ok(all_results);
                },
            };

            let wtns = match wtns_file.read_wtns_header(&sections_wtns[..]) {
                Ok(wtns) => wtns,
                Err(e) => {
                    debug_println!("[GROTH16] ERROR: Failed to read witness header {}: {:?}", i + 1, e);
                    // Fill remaining results with failures
                    all_results.extend(vec![ProverResult::FAILURE; witness_paths.len() - all_results.len()]);
                    return Ok(all_results);
                },
            };

            if !F::eq(&header.r, &wtns.q) || wtns.n_witness != header.n_vars {
                debug_println!("[GROTH16] ERROR: Witness {} validation failed - r: {:?} vs {:?}, n_witness: {} vs {}", 
                         i + 1, header.r, wtns.q, header.n_vars, wtns.n_witness);
                // Fill remaining results with failures
                all_results.extend(vec![ProverResult::FAILURE; witness_paths.len() - all_results.len()]);
                return Ok(all_results);
            }

            let buff_witness = match wtns_file.read_section(&sections_wtns[..], 2) {
                Ok(buff) => buff,
                Err(e) => {
                    debug_println!("[GROTH16] ERROR: Failed to read witness section {}: {:?}", i + 1, e);
                    // Fill remaining results with failures
                    all_results.extend(vec![ProverResult::FAILURE; witness_paths.len() - all_results.len()]);
                    return Ok(all_results);
                },
            };

            let scalars = from_u8::<F>(buff_witness).to_vec();
            debug_println!("[GROTH16] Witness {} loaded with {} scalars", i + 1, scalars.len());
            scalars_array.push(scalars);
        }
        
        debug_println!("[GROTH16] Successfully loaded {} witnesses for batch {}", scalars_array.len(), batch_idx + 1);

        // Generate proofs for current batch
        let prove_results = match device_type {
            DeviceType::Cpu => {
                prove_cpu_batched(&scalars_array, &zkey, &header, current_batch_size)
            }
            DeviceType::CpuMetal => {
                prove_metal_cpu_batched(&scalars_array, &zkey, &header, current_batch_size)
            }
            DeviceType::Metal => {
                prove_metal_batched(&scalars_array, &zkey, &header, current_batch_size)
            }
        };

        debug_println!("[GROTH16] Processing {} proof results for batch {}...", prove_results.len(), batch_idx + 1);
        
        // Process results and save files for current batch
        for (batch_result_idx, (pi_a, pi_b1, pi_b, pi_c, pi_h)) in prove_results.into_iter().enumerate() {
            let global_idx = start_idx + batch_result_idx;
            debug_println!("[GROTH16] Processing proof result {} (global index {})...", batch_result_idx + 1, global_idx + 1);
            
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

            // Extract public signals from the scalars array
            let mut public_signals = Vec::with_capacity(header.n_public);
            for j in 1..=header.n_public {
                let scalar_bytes: BigUint = BigUint::from_bytes_le(&scalars_array[batch_result_idx][j].to_bytes_le());
                public_signals.push(scalar_bytes.to_str_radix(10));
            }

            let proof = Proof {
                pi_a: serialize_g1_affine(pi_a.into()),
                pi_b: serialize_g2_affine(pi_b.into()),
                pi_c: serialize_g1_affine(pi_c.into()),
                protocol: "groth16".to_string(),
                curve: "bn128".to_string(),
            };

            let proof_save_result = FileWrapper::save_json_file(&proof_paths[global_idx], &proof);
            let public_save_result = FileWrapper::save_json_file(&public_paths[global_idx], &public_signals);
            
            let result = if proof_save_result.is_err() || public_save_result.is_err() {
                debug_println!("[GROTH16] ERROR: Failed to save files for proof {} (global index {})", batch_result_idx + 1, global_idx + 1);
                if proof_save_result.is_err() {
                    debug_println!("[GROTH16] Proof save error: {:?}", proof_save_result.err());
                }
                if public_save_result.is_err() {
                    debug_println!("[GROTH16] Public save error: {:?}", public_save_result.err());
                }
                ProverResult::FAILURE
            } else {
                debug_println!("[GROTH16] Successfully saved files for proof {} (global index {})", batch_result_idx + 1, global_idx + 1);
                ProverResult::SUCCESS
            };
            
            all_results.push(result);
            debug_println!("[GROTH16] Proof {} (global index {}) result: {:?}", batch_result_idx + 1, global_idx + 1, result);
        }
        
        debug_println!("[GROTH16] Completed batch {}/{} with {} results", batch_idx + 1, total_batches, current_batch_size);
    }

    debug_println!("[GROTH16] All batches completed with {} total results: {:?}", all_results.len(), all_results);
    
    // Clean up NTT domain after all batches are complete
    let _ = release_domain::<ScalarField>();
    
    Ok(all_results)
}



