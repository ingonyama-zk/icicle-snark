use std::thread;
use std::time::Instant;
use icicle_bn254::curve:: G2Affine;
use crate::{
    cache::{VerificationKey, ZKeyCache}, conversions::{deserialize_g1_affine, deserialize_g2_affine, from_u8, serialize_g1_affine, serialize_g2_affine}, file_wrapper::FileWrapper, icicle_helper::{msm_helper, ntt_helper}, ProjectiveG1, ProjectiveG2, F
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

pub fn construct_r1cs(witness: &[ScalarField], zkey_cache: &ZKeyCache) -> DeviceVec<ScalarField> {
    let mut stream = IcicleStream::create().unwrap();
    let mut cfg = VecOpsConfig::default();
    cfg.is_async = true;
    cfg.stream_handle = *stream;

    let n_coef = zkey_cache.c_values.len();
    let nof_coef = zkey_cache.zkey.domain_size;

    let mut d_second_slice = DeviceVec::device_malloc_async(n_coef, &stream).unwrap();
    let mut d_vec = DeviceVec::device_malloc_async(nof_coef * 3, &stream).unwrap();

    let mut out_buff_b_a = vec![ScalarField::zero(); nof_coef * 2];

    let first_slice = &zkey_cache.first_slice;
    let s_values = &zkey_cache.s_values;
    let c_values = &zkey_cache.c_values;
    let m_values = &zkey_cache.m_values;

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

    d_second_slice
        .copy_from_host_async(second_slice, &stream)
        .unwrap();
    ScalarField::from_mont(&mut d_second_slice, &stream);
    mul_scalars(&first_slice[..], &d_second_slice, res, &cfg).unwrap();

    stream.synchronize().unwrap();

    let zero_scalar = ScalarField::zero();

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
    
    println!("SP: INTT input size: {}", d_vec.len());
    let mut ntt_cfg = NTTConfig::default();
    ntt_cfg.stream_handle = (&stream).into();
    ntt_cfg.is_async = true;
    ntt_cfg.batch_size = 3;
    ntt_helper(&mut d_vec, true, &ntt_cfg);

    ntt_cfg.coset_gen = zkey_cache.coset_gen;
    println!("SP: NTT input size: {}", d_vec.len());
    ntt_helper(&mut d_vec, false, &ntt_cfg);

    stream.synchronize().unwrap();
    stream.destroy().unwrap();

    // L * R - O
    let cfg: VecOpsConfig = VecOpsConfig::default();
    mul_scalars(&d_vec[0..nof_coef], &d_vec[nof_coef..nof_coef * 2], &mut d_vec_copy[0..nof_coef], &cfg).unwrap();
    sub_scalars(&d_vec[0..nof_coef], &d_vec[2 * nof_coef..], &mut d_vec_copy[nof_coef..nof_coef * 2], &cfg).unwrap();
    //r1csrelease_domain::<ScalarField>();    
    d_vec
}

pub fn groth16_commitments(
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
    msm_config.are_bases_montgomery_form = true;
    
    let a: ProjectiveG1;
    let b1: ProjectiveG1;
    let b: ProjectiveG2;
    let c: ProjectiveG1;
    
    {
        // let points_a = from_u8(zkey_file.read_section(sections_zkey, 5).unwrap());
        let points_a = &zkey_cache.points_a;
        println!("MSM a input sizes - scalars: {}, points: {}", host_scalars.len(), points_a.len());
        a = msm_helper(&host_scalars[..], points_a, &msm_config, "a");
    }
    
    {
        // let points_b1 = from_u8(zkey_file.read_section(sections_zkey, 6).unwrap());
        let points_b1 = &zkey_cache.points_b1;
        println!("MSM b1 input sizes - scalars: {}, points: {}", host_scalars.len(), points_b1.len());
        b1 = msm_helper(&host_scalars[..], points_b1, &msm_config, "b1");
    }
    
    {
        // let points_b = from_u8(zkey_file.read_section(sections_zkey, 7).unwrap());
        let points_b = &zkey_cache.points_b;
        println!("MSM b input sizes - scalars: {}, points: {}", host_scalars.len(), points_b.len());
        b = msm_helper(&host_scalars[..], points_b, &msm_config, "b");
    }
    
    {
        // let points_c = from_u8(zkey_file.read_section(sections_zkey, 8).unwrap());
        let points_c = &zkey_cache.points_c;
        println!("MSM c input sizes - scalars: {}, points: {}", zkey_cache.zkey.n_public + 1, points_c.len());
        c = msm_helper(&host_scalars[zkey_cache.zkey.n_public + 1..], points_c, &msm_config, "c");
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

    let zkey = &zkey_cache.zkey;

    if !F::eq(&zkey.r, &wtns.q) {
        panic!("Curve of the witness does not match the curve of the proving key");
    }

    if wtns.n_witness != zkey.n_vars {
        panic!(
            "Invalid witness length. Circuit: {}, witness: {}",
            zkey.n_vars, wtns.n_witness
        );
    }

    let buff_witness = wtns_file.read_section(&sections_wtns[..], 2).unwrap();

    let scalars = from_u8(buff_witness);

    let d_vec = construct_r1cs(scalars, zkey_cache);

    // TODO: move this to a separate thread operating on CPU
    let (pi_a, pi_b1, pi_b, pi_c) = groth16_commitments(scalars, zkey_cache);
    // END CPU
    
    // TODO: move this to a separate thread operating on METAL
    let num_coef = zkey_cache.zkey.domain_size;
    let mut msm_config = MSMConfig::default();
    msm_config.is_async = false;
    msm_config.c = 14;
    msm_config.are_bases_montgomery_form = true;
    let points_h = &zkey_cache.points_h;
    println!("MSM h input sizes - scalars: {}, points: {}", num_coef, points_h.len());
    let pi_h = msm_helper(&d_vec[num_coef..num_coef * 2], points_h, &msm_config, "h");

    #[cfg(not(feature = "no-randomness"))]
    let (pi_a, pi_b, pi_c) = {
        let rs = ScalarCfg::generate_random(2);
        let r = rs[0];
        let s = rs[1];

        let pi_a = pi_a + zkey.vk_alpha_1 + zkey.vk_delta_1 * r;
        let pi_b = pi_b + zkey.vk_beta_2 + zkey.vk_delta_2 * s;
        let pi_b1 = pi_b1 + zkey.vk_beta_1 + zkey.vk_delta_1 * s;
        let pi_c = pi_c + pi_h + pi_a * s + pi_b1 * r - zkey.vk_delta_1 * r * s;

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

    let mut public_signals = Vec::with_capacity(zkey.n_public);
    let field_size = ScalarField::zero().to_bytes_le().len();

    for i in 1..=zkey.n_public {
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