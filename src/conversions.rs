use icicle_core::{
    curve::{Affine, Curve},
    traits::{FieldImpl, MontgomeryConvertible},
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};
use num_bigint::BigUint;

use crate::{G1, G2};

pub fn from_affine_mont<C: Curve>(points: &mut [Affine<C>]) {
    let mut stream = IcicleStream::create().unwrap();
    let mut d_affine = DeviceVec::device_malloc_async(points.len(), &stream).unwrap();
    d_affine
        .copy_from_host_async(HostSlice::from_slice(points), &stream)
        .unwrap();

    Affine::from_mont(&mut d_affine, &stream).wrap().unwrap();

    d_affine
        .copy_to_host_async(HostSlice::from_mut_slice(points), &stream)
        .unwrap();

    stream.synchronize().unwrap();
    stream.destroy().unwrap();
}

pub fn serialize_g1_affine(point: G1) -> Vec<String> {
    let x_bytes = BigUint::from_bytes_le(&point.x.to_bytes_le()[..]);
    let y_bytes = BigUint::from_bytes_le(&point.y.to_bytes_le()[..]);

    vec![
        x_bytes.to_str_radix(10),
        y_bytes.to_str_radix(10),
        "1".to_string(),
    ]
}

pub fn serialize_g2_affine(point: G2) -> Vec<Vec<String>> {
    let x_bytes = point.x.to_bytes_le();
    let size = x_bytes.len() / 2;
    let x_bytes_1 = BigUint::from_bytes_le(&x_bytes[..size]);
    let x_bytes_2 = BigUint::from_bytes_le(&x_bytes[size..]);

    let y_bytes = point.y.to_bytes_le();
    let y_bytes_1 = BigUint::from_bytes_le(&y_bytes[..size]);
    let y_bytes_2 = BigUint::from_bytes_le(&y_bytes[size..]);

    vec![
        vec![x_bytes_1.to_str_radix(10), x_bytes_2.to_str_radix(10)],
        vec![y_bytes_1.to_str_radix(10), y_bytes_2.to_str_radix(10)],
        vec!["1".to_string(), "0".to_string()],
    ]
}

pub fn deserialize_g1_affine(data: &[String]) -> G1 {
    let to_limbs = |s: &String| {
        let mut bytes = BigUint::parse_bytes(s.as_bytes(), 10).unwrap().to_bytes_le();
        bytes.resize(32, 0);
        let mut limbs = [0u32; 8];
        for (i, chunk) in bytes.chunks(4).enumerate() {
            limbs[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        limbs
    };

    G1::from_limbs(to_limbs(&data[0]), to_limbs(&data[1]))
}

pub fn deserialize_g2_affine(data: &[Vec<String>]) -> G2 {
    let to_limbs = |s: &String| {
        let mut bytes = BigUint::parse_bytes(s.as_bytes(), 10).unwrap().to_bytes_le();
        bytes.resize(32, 0);
        let mut limbs = [0u32; 8];
        for (i, chunk) in bytes.chunks(4).enumerate() {
            limbs[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        limbs
    };

    let mut x_limbs = [0u32; 16];
    let x_c0 = to_limbs(&data[0][0]);
    let x_c1 = to_limbs(&data[0][1]);
    x_limbs[..8].copy_from_slice(&x_c0);
    x_limbs[8..].copy_from_slice(&x_c1);

    let mut y_limbs = [0u32; 16];
    let y_c0 = to_limbs(&data[1][0]);
    let y_c1 = to_limbs(&data[1][1]);
    y_limbs[..8].copy_from_slice(&y_c0);
    y_limbs[8..].copy_from_slice(&y_c1);

    G2::from_limbs(x_limbs, y_limbs)
}


pub fn from_u8<T>(data: &[u8]) -> &[T] {
    let num_data = data.len() / size_of::<T>();

    let ptr = data.as_ptr() as *mut T;
    let target_data = unsafe { std::slice::from_raw_parts(ptr, num_data) };

    target_data
}
