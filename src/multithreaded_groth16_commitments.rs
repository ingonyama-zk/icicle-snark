use std::thread;
use std::time::Instant;
use crate::{
    cache::ZKeyCache, conversions::{from_u8, serialize_g1_affine, serialize_g2_affine}, file_wrapper::FileWrapper, icicle_helper::{msm_helper, ntt_helper}, ProjectiveG1, ProjectiveG2, F
};
use icicle_bn254::curve::ScalarField;
use icicle_core::{
    traits::{FieldImpl, MontgomeryConvertible}, vec_ops::{mul_scalars, sub_scalars, VecOpsConfig}
};
use icicle_runtime::{memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice}, stream::IcicleStream};

pub fn multithreaded_groth16_commitments(
    d_vec: DeviceVec<F>,
    scalars: &[F],
    zkey_cache: &ZKeyCache
) -> (ProjectiveG1, ProjectiveG1, ProjectiveG2, ProjectiveG1, ProjectiveG1) {
    let nof_coef = zkey_cache.zkey.domain_size;
    let points_a = &zkey_cache.points_a;
    let points_b1 = &zkey_cache.points_b1;
    let points_b = &zkey_cache.points_b;
    let points_c = &zkey_cache.points_c;
    let points_h = &zkey_cache.points_h;

    // Create a copy of scalars for the G2 thread
    let scalars_g2 = scalars.to_vec();
    let d_vec_g2 = d_vec.clone();

    // Spawn G2 thread
    let g2_handle = thread::spawn(move || {
        let mut stream_g2 = IcicleStream::create().unwrap();
        let scalars = HostSlice::from_slice(&scalars_g2);
        let mut d_scalars = DeviceVec::device_malloc_async(scalars.len(), &stream_g2).unwrap();
        d_scalars.copy_from_host_async(scalars, &stream_g2).unwrap();

        println!("Starting G2 MSM");
        let g2_start = Instant::now();
        let commitment_b = msm_helper(&d_scalars[..], points_b, &stream_g2);
        println!("G2 MSM took: {:?}", g2_start.elapsed());

        let mut pi_b = [ProjectiveG2::zero(); 1];
        commitment_b.copy_to_host_async(HostSlice::from_mut_slice(&mut pi_b[..]), &stream_g2).unwrap();
        stream_g2.synchronize().unwrap();
        stream_g2.destroy().unwrap();
        
        pi_b[0]
    });

    // Main thread handles G1 operations
    let mut stream_g1 = IcicleStream::create().unwrap();
    let scalars = HostSlice::from_slice(scalars);
    let mut d_scalars = DeviceVec::device_malloc_async(scalars.len(), &stream_g1).unwrap();
    d_scalars.copy_from_host_async(scalars, &stream_g1).unwrap();

    println!("Starting G1 MSMs");
    let g1_start = Instant::now();
    let commitment_a = msm_helper(&d_scalars[..], points_a, &stream_g1);
    let commitment_b1 = msm_helper(&d_scalars[..], points_b1, &stream_g1);
    let commitment_c = msm_helper(&d_scalars[zkey_cache.zkey.n_public + 1..], points_c, &stream_g1);
    let commitment_h = msm_helper(&d_vec[nof_coef..nof_coef * 2], points_h, &stream_g1);
    println!("4 G1 MSMs took: {:?}", g1_start.elapsed());

    let mut pi_a = [ProjectiveG1::zero(); 1];
    let mut pi_b1 = [ProjectiveG1::zero(); 1];
    let mut pi_c = [ProjectiveG1::zero(); 1];
    let mut pi_h = [ProjectiveG1::zero(); 1];

    commitment_a.copy_to_host_async(HostSlice::from_mut_slice(&mut pi_a[..]), &stream_g1).unwrap();
    commitment_b1.copy_to_host_async(HostSlice::from_mut_slice(&mut pi_b1[..]), &stream_g1).unwrap();
    commitment_c.copy_to_host_async(HostSlice::from_mut_slice(&mut pi_c[..]), &stream_g1).unwrap();
    commitment_h.copy_to_host_async(HostSlice::from_mut_slice(&mut pi_h[..]), &stream_g1).unwrap();

    stream_g1.synchronize().unwrap();
    stream_g1.destroy().unwrap();

    // Wait for G2 thread to complete
    let pi_b = g2_handle.join().unwrap();

    (pi_a[0], pi_b1[0], pi_b, pi_c[0], pi_h[0])
} 