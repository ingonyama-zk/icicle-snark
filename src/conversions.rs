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

pub fn from_u8<T>(data: &[u8]) -> &[T] {
    let num_data = data.len() / size_of::<T>();

    let ptr = data.as_ptr() as *mut T;
    let target_data = unsafe { std::slice::from_raw_parts(ptr, num_data) };

    target_data
}
