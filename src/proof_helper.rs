use std::thread;
use std::time::Instant;
use icicle_bn254::curve:: G2Affine;
use crate::{
    cache::{VerificationKey, ZKeyCache}, conversions::{deserialize_g1_affine, deserialize_g2_affine, from_u8, serialize_g1_affine, serialize_g2_affine}, file_wrapper::FileWrapper, icicle_helper::{msm_helper, ntt_helper}, ProjectiveG1, ProjectiveG2,
    F, G1, G2
};
use icicle_bn254::curve::{G1Projective, ScalarField};
use icicle_core::{
    msm::{MSMConfig}, 
    field::Field, pairing::pairing, traits::{FieldImpl, MontgomeryConvertible}, vec_ops::{mul_scalars, sub_scalars, VecOpsConfig}, ntt::{NTTConfig, release_domain}
};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice}, stream::IcicleStream
};
use num_bigint::BigUint;
use serde::{Serialize, Deserialize};
use serde_json::Value;

use rayon::prelude::*;

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

fn construct_r1cs(witness: &[ScalarField], zkey_cache: &ZKeyCache) -> DeviceVec<ScalarField> {
    let mut stream = IcicleStream::create().unwrap();
    let mut cfg = VecOpsConfig::default();
    cfg.is_async = true;
    cfg.stream_handle = *stream;

    // ------------------------------------------------------------
    let buff_coeffs = zkey_cache.file.read_section(&zkey_cache.sections[..], 4).unwrap();
    let s_coef = 4 * 3 + zkey_cache.header.n8r;
    let n_coef = (buff_coeffs.len() - 4) / s_coef;

    let mut first_slice = Vec::with_capacity(n_coef);
    let mut s_values = Vec::with_capacity(n_coef);
    let mut c_values = Vec::with_capacity(n_coef);
    let mut m_values = Vec::with_capacity(n_coef);

    unsafe {
        first_slice.set_len(n_coef);
        s_values.set_len(n_coef);
        c_values.set_len(n_coef);
        m_values.set_len(n_coef);
    }
    let n8 = 32;

    s_values
        .par_iter_mut()
        .zip(c_values.par_iter_mut())
        .zip(m_values.par_iter_mut())
        .zip(first_slice.par_iter_mut())
        .enumerate()
        .for_each(|(i, (((s_val, c_val), m_val), coef_val))| {
            let start = 4 + i * s_coef;
            let buff_coef = &buff_coeffs[start..start + s_coef];

            let s =
                u32::from_le_bytes([buff_coef[8], buff_coef[9], buff_coef[10], buff_coef[11]])
                    as usize;
            let c = u32::from_le_bytes([buff_coef[4], buff_coef[5], buff_coef[6], buff_coef[7]])
                as usize;
            let m = buff_coef[0];
            let coef = ScalarField::from_bytes_le(&buff_coef[12..12 + n8]);

            *s_val = s;
            *c_val = c;
            *m_val = m as usize;
            *coef_val = coef;
        });

    let mut d_first_slice = DeviceVec::device_malloc_async(first_slice.len(), &stream).unwrap();
    let first_slice = HostSlice::from_slice(&first_slice);
    d_first_slice
        .copy_from_host_async(first_slice, &stream)
        .unwrap();

    ScalarField::from_mont(&mut d_first_slice, &stream);

    stream.synchronize().unwrap();
    stream.destroy().unwrap();
    // ------------------------------------------------------------

    let mut second_slice = Vec::with_capacity(n_coef);
    unsafe {
        second_slice.set_len(n_coef);
    }

    second_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, slice_elem)| {
            let s = s_values[i];
            *slice_elem = witness[s];
        });

    let second_slice = HostSlice::from_mut_slice(&mut second_slice);

    let mut res = Vec::with_capacity(n_coef);
    unsafe {
        res.set_len(n_coef);
    }
    let res = HostSlice::from_mut_slice(&mut res);

    let mut d_second_slice = DeviceVec::device_malloc_async(n_coef, &stream).unwrap();
    d_second_slice
        .copy_from_host_async(second_slice, &stream)
        .unwrap();
    ScalarField::from_mont(&mut d_second_slice, &stream);
    mul_scalars(&d_first_slice[..], &d_second_slice, res, &cfg).unwrap();

    stream.synchronize().unwrap();

    let nof_coef = zkey_cache.header.domain_size;
    let zero_scalar = ScalarField::zero();
    let mut out_buff_b_a = vec![ScalarField::zero(); nof_coef * 2];

    for i in 0..n_coef {
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

fn compute_h(d_vec: &mut DeviceVec<ScalarField>, coset_gen: ScalarField, nof_coef: usize) -> DeviceVec<ScalarField> {
    
    println!("SP: INTT input size: {}", d_vec.len());
    let mut ntt_cfg = NTTConfig::default();
    ntt_cfg.is_async = true;
    ntt_cfg.batch_size = 3;
    ntt_helper(d_vec, true, &ntt_cfg);

    ntt_cfg.coset_gen = coset_gen;
    println!("SP: NTT input size: {}", d_vec.len());
    ntt_helper(d_vec, false, &ntt_cfg);

    // L * R - O
    let cfg: VecOpsConfig = VecOpsConfig::default();
    let d_vec_copy = unsafe {
        DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            d_vec.as_mut_ptr(),
            d_vec.len(),
        ))
    };
    mul_scalars(&d_vec[0..nof_coef], &d_vec[nof_coef..nof_coef * 2], &mut d_vec_copy[0..nof_coef], &cfg).unwrap();
    let mut d_h = DeviceVec::device_malloc(nof_coef).unwrap();
    sub_scalars(&d_vec[0..nof_coef], &d_vec[2 * nof_coef..], &mut d_h, &cfg).unwrap();
    release_domain::<ScalarField>();

    d_h
}

fn groth16_commitments(
    scalars: &[F],
    zkey_cache: &ZKeyCache,
) -> (
    ProjectiveG1,
    ProjectiveG1,
    ProjectiveG2,
    ProjectiveG1,
) {
    let host_scalars = HostSlice::from_slice(scalars);
    let mut msm_config = MSMConfig::default();
    msm_config.is_async = false;
    msm_config.c = 14;
    // msm_config.are_bases_montgomery_form = true;
    
    let a: ProjectiveG1;
    let b1: ProjectiveG1;
    let b: ProjectiveG2;
    let c: ProjectiveG1;
    
    {
        let mut stream = IcicleStream::create().unwrap();
        let points_a_raw = from_u8(&zkey_cache.file.read_section(&zkey_cache.sections, 5).unwrap());
        let points_a = HostSlice::from_slice(&points_a_raw);
        let mut d_points_a = DeviceVec::device_malloc_async(points_a.len(), &stream).unwrap();
        d_points_a.copy_from_host_async(points_a, &stream).unwrap();
        G1::from_mont(&mut d_points_a, &stream);
        stream.synchronize().unwrap();
        stream.destroy().unwrap();
        
        // let points_a = &zkey_cache.points_a;
        println!("MSM a input sizes - scalars: {}, points: {}", host_scalars.len(), points_a.len());
        a = msm_helper(&host_scalars[..], &d_points_a, &msm_config, "a");
    }
    
    {
        let mut stream = IcicleStream::create().unwrap();
        let points_b1_raw = from_u8(&zkey_cache.file.read_section(&zkey_cache.sections, 6).unwrap());
        let points_b1 = HostSlice::from_slice(&points_b1_raw);
        let mut d_points_b1 = DeviceVec::device_malloc_async(points_b1.len(), &stream).unwrap();
        d_points_b1.copy_from_host_async(points_b1, &stream).unwrap();
        G1::from_mont(&mut d_points_b1, &stream);
        stream.synchronize().unwrap();
        stream.destroy().unwrap();
        
        // let points_b1 = &zkey_cache.points_b1;
        println!("MSM b1 input sizes - scalars: {}, points: {}", host_scalars.len(), points_b1.len());
        b1 = msm_helper(&host_scalars[..], &d_points_b1, &msm_config, "b1");
    }
    
    {
        let mut stream = IcicleStream::create().unwrap();
        let points_b_raw = from_u8(&zkey_cache.file.read_section(&zkey_cache.sections, 7).unwrap());
        let points_b = HostSlice::from_slice(&points_b_raw);
        let mut d_points_b = DeviceVec::device_malloc_async(points_b.len(), &stream).unwrap();
        d_points_b.copy_from_host_async(points_b, &stream).unwrap();
        G2::from_mont(&mut d_points_b, &stream);
        stream.synchronize().unwrap();
        stream.destroy().unwrap();
        
        // let points_b = &zkey_cache.points_b;
        println!("MSM b input sizes - scalars: {}, points: {}", host_scalars.len(), points_b.len());
        b = msm_helper(&host_scalars[..], &d_points_b, &msm_config, "b");
    }
    
    {
        let mut stream = IcicleStream::create().unwrap();
        let points_c_raw = from_u8(&zkey_cache.file.read_section(&zkey_cache.sections, 8).unwrap());
        let points_c = HostSlice::from_slice(&points_c_raw);
        let mut d_points_c = DeviceVec::device_malloc_async(points_c.len(), &stream).unwrap();
        d_points_c.copy_from_host_async(points_c, &stream).unwrap();
        G1::from_mont(&mut d_points_c, &stream);
        stream.synchronize().unwrap();
        stream.destroy().unwrap();
        
        // let points_c = &zkey_cache.points_c;
        println!("MSM c input sizes - scalars: {}, points: {}", zkey_cache.header.n_public + 1, points_c.len());
        c = msm_helper(&host_scalars[zkey_cache.header.n_public + 1..], &d_points_c, &msm_config, "c");
    }

    (a, b1, b, c)
}

pub fn groth16_prove_helper(
    witness: &str,
    zkey_cache: &ZKeyCache,
) -> Result<(Value, Value), Box<dyn std::error::Error>> {   
    let (fd_wtns, sections_wtns) = FileWrapper::read_bin_file(witness, "wtns", 2).unwrap();
    let mut wtns_file = FileWrapper::new(fd_wtns).unwrap();
    let wtns = wtns_file.read_wtns_header(&sections_wtns[..]).unwrap();
    let header = &zkey_cache.header;

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

    // TODO: move this to a separate thread operating on CPU
    let (pi_a, pi_b1, pi_b, pi_c) = groth16_commitments(scalars, zkey_cache);
    // END CPU
    
    // TODO: move this to a separate thread operating on METAL
    let mut d_vec = construct_r1cs(scalars, zkey_cache);
    let num_coef = header.domain_size;
    let d_h = compute_h(&mut d_vec, zkey_cache.coset_gen, num_coef);
    
    let mut msm_config = MSMConfig::default();
    msm_config.is_async = false;
    // TODO: @jeremy try letting this be dynamic
    msm_config.c = 14;

    let mut stream = IcicleStream::create().unwrap();
    let points_h_raw = from_u8(&zkey_cache.file.read_section(&zkey_cache.sections, 9).unwrap());
    let points_h = HostSlice::from_slice(&points_h_raw);
    let mut d_points_h = DeviceVec::device_malloc_async(points_h.len(), &stream).unwrap();
    d_points_h.copy_from_host_async(points_h, &stream).unwrap();
    G1::from_mont(&mut d_points_h, &stream);
    stream.synchronize().unwrap();
    stream.destroy().unwrap();
    
    // let points_h = &zkey_cache.points_h;
    println!("MSM h input sizes - scalars: {}, points: {}", num_coef, points_h.len());
    let pi_h = msm_helper(&d_h[..], &d_points_h, &msm_config, "h");

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

    Ok((serde_json::json!(proof), serde_json::json!(public_signals)))
}

pub fn groth16_verify_helper(
    proof: &Proof,
    public: &[String],
    verification_key: &VerificationKey,
) -> Result<bool, Box<dyn std::error::Error>> {
    let pi_a = deserialize_g1_affine(&proof.pi_a);
    let pi_b = deserialize_g2_affine(&proof.pi_b);
    let pi_c = deserialize_g1_affine(&proof.pi_c);
    
    let n_public = verification_key.n_public;
    println!("SP: n_public: {}", n_public);
    
    let ic = verification_key.ic.clone();

    let mut public_scalars = Vec::with_capacity(n_public);
    for s in public.iter().take(n_public) {
        let hex = BigUint::parse_bytes(s.as_bytes(), 10).unwrap();
        let scalar = ScalarField::from_bytes_le(&hex.to_bytes_le());
        public_scalars.push(scalar);
    }

    let mut cpub = ic[0].to_projective();
    for i in 0..public_scalars.len() {
        cpub = cpub + ic[i + 1].to_projective() * public_scalars[i];
    }

    let neg_pi_a = ProjectiveG1::zero() - pi_a.to_projective();

    // e(-A, B) * e(cpub, gamma_2) * e(C, delta_2) * e(alpha_1, beta_2) = 1
    let vk_gamma_2 = verification_key.vk_gamma_2.clone();
    let vk_delta_2 = verification_key.vk_delta_2.clone();
    let vk_alpha_1 = verification_key.vk_alpha_1.clone();
    let vk_beta_2 = verification_key.vk_beta_2.clone();

    // let first_thread = std::thread::spawn(move || {
    let first0 = pairing(&neg_pi_a.into(), &pi_b); //.unwrap();
    let first = first0.unwrap();
    // });
    //let second_thread = std::thread::spawn(move || {
    let second =    pairing(&cpub.into(), &vk_gamma_2).unwrap();
    //});
    //let third_thread = std::thread::spawn(move || {
    let third =     pairing(&pi_c, &vk_delta_2).unwrap();
    //});
    //let fourth_thread = std::thread::spawn(move || {
    let fourth =    pairing(&vk_alpha_1, &vk_beta_2).unwrap();
    //});

    //let first = first_thread.join().unwrap();
    // let second = second_thread.join().unwrap();
    // let third = third_thread.join().unwrap();
    // let fourth = fourth_thread.join().unwrap();

    let result = Field::one() == first * second * third * fourth;

    Ok(result)
}