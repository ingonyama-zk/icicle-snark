use crate::{
    cache::{ZKeyCache}, conversions::from_u8, file_wrapper::FileWrapper, icicle_helper::{msm_helper, ntt_helper}, ProjectiveG1, ProjectiveG2, F
};
use icicle_bn254::curve::{ScalarField};
use icicle_core::{
    traits::{FieldImpl, MontgomeryConvertible}, vec_ops::{mul_scalars, sub_scalars, VecOpsConfig}
};
use icicle_runtime::{
    memory::{HostOrDeviceSlice, HostSlice}, stream::IcicleStream
};
use serde::{Serialize, Deserialize};

use rayon::prelude::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct Proof {
    pub pi_a: Vec<String>,
    pub pi_b: Vec<Vec<String>>,
    pub pi_c: Vec<String>,
    pub protocol: String,
    pub curve: String,
}

pub fn construct_r1cs(witness: &[ScalarField], zkey_cache: &ZKeyCache) -> Vec<ScalarField> {
    let mut stream = IcicleStream::create().unwrap();
    let mut cfg = VecOpsConfig::default();
    cfg.is_async = true;
    cfg.stream_handle = *stream;

    let n_coef = zkey_cache.c_values.len();
    let nof_coef = zkey_cache.zkey.domain_size;

    let mut vec = vec![ScalarField::zero(); nof_coef * 3];
    let mut h_vec = HostSlice::from_mut_slice(&mut vec);

    let mut out_buff_b_a = vec![ScalarField::zero(); nof_coef * 2];

    let first_slice = &zkey_cache.first_slice;
    let h_first_slice = HostSlice::from_slice(first_slice);
    let s_values = &zkey_cache.s_values;
    let c_values = &zkey_cache.c_values;
    let m_values = &zkey_cache.m_values;
    let keys = &zkey_cache.keys;

    let h_keys = HostSlice::from_slice(keys);

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

    ScalarField::from_mont(&mut second_slice[..], &stream);
    mul_scalars(&h_first_slice[..], &second_slice[..], res, &cfg).unwrap();

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

    h_vec[0..nof_coef]
        .copy(HostSlice::from_slice(&out_buff_b_a[nof_coef..]))
        .unwrap();
    h_vec[nof_coef..nof_coef * 2]
        .copy(HostSlice::from_slice(&out_buff_b_a[..nof_coef]))
        .unwrap();

    let h_vec_copy = unsafe {
        HostSlice::from_mut_slice(std::slice::from_raw_parts_mut(
            h_vec.as_mut_ptr(),
            h_vec.len(),
        ))
    };

    mul_scalars(
        &h_vec[0..nof_coef],
        &h_vec[nof_coef..nof_coef * 2],
        &mut h_vec_copy[2 * nof_coef..],
        &cfg,
    )
    .unwrap();

    ntt_helper(&mut h_vec, true, &stream);

    mul_scalars(
        &h_vec[..nof_coef],
        &h_keys[..],
        &mut h_vec_copy[..nof_coef],
        &cfg,
    )
    .unwrap();
    mul_scalars(
        &h_vec[nof_coef..nof_coef * 2],
        &h_keys[..],
        &mut h_vec_copy[nof_coef..2 * nof_coef],
        &cfg,
    )
    .unwrap();
    mul_scalars(
        &h_vec[nof_coef * 2..],
        &h_keys[..],
        &mut h_vec_copy[2 * nof_coef..],
        &cfg,
    )
    .unwrap();

    ntt_helper(&mut h_vec, false, &stream);

    stream.synchronize().unwrap();
    stream.destroy().unwrap();

    // L * R - O
    let cfg: VecOpsConfig = VecOpsConfig::default();
    mul_scalars(
        &h_vec[0..nof_coef],
        &h_vec[nof_coef..nof_coef * 2],
        &mut h_vec_copy[0..nof_coef],
        &cfg,
    )
    .unwrap();
    sub_scalars(
        &h_vec[0..nof_coef],
        &h_vec[2 * nof_coef..],
        &mut h_vec_copy[nof_coef..nof_coef * 2],
        &cfg,
    )
    .unwrap();

    h_vec[..].as_slice().to_vec()
}

pub fn groth16_commitments(
    h_vec: Vec<F>,
    scalars: &[F],
    zkey_cache: &ZKeyCache,
) -> (
    ProjectiveG1,
    ProjectiveG1,
    ProjectiveG2,
    ProjectiveG1,
    ProjectiveG1,
) {
    let nof_coef = zkey_cache.zkey.domain_size;
    // A, B, C
    let points_a = &zkey_cache.points_a;
    let points_b1 = &zkey_cache.points_b1;
    let points_b = &zkey_cache.points_b;
    let points_c = &zkey_cache.points_c;
    let points_h = &zkey_cache.points_h;

    let mut stream_g1 = IcicleStream::create().unwrap();
    let mut stream_g2 = IcicleStream::create().unwrap();

    let pi_a = msm_helper(&scalars[..], points_a, &stream_g1);
    let pi_b1 = msm_helper(&scalars[..], points_b1, &stream_g1);
    let pi_c = msm_helper(
        &scalars[zkey_cache.zkey.n_public + 1..],
        points_c,
        &stream_g1,
    );
    let pi_h = msm_helper(&h_vec[nof_coef..nof_coef * 2], points_h, &stream_g1);
    let pi_b = msm_helper(&scalars[..], points_b, &stream_g2);

    stream_g1.synchronize().unwrap();
    stream_g2.synchronize().unwrap();

    stream_g1.destroy().unwrap();
    stream_g2.destroy().unwrap();

    (pi_a, pi_b1, pi_b, pi_c, pi_h)
}

pub fn groth16_prove_helper(
    witness: &str,
    zkey_cache: &ZKeyCache,
) {
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

    groth16_commitments(d_vec, scalars, zkey_cache);
}
