use icicle_core::{
    curve::{Affine, Curve},
    traits::MontgomeryConvertible,
};
use icicle_runtime::{
    memory::HostSlice,
    stream::IcicleStream,
};

pub fn from_affine_mont<C: Curve>(points: &mut [Affine<C>]) {
    let mut stream = IcicleStream::create().unwrap();
    let h_points = HostSlice::from_mut_slice(points);

    Affine::from_mont(&mut h_points[..], &stream).wrap().unwrap();

    stream.synchronize().unwrap();
    stream.destroy().unwrap();
}

pub fn from_u8<T>(data: &[u8]) -> &[T] {
    let num_data = data.len() / size_of::<T>();

    let ptr = data.as_ptr() as *mut T;
    let target_data = unsafe { std::slice::from_raw_parts(ptr, num_data) };

    target_data
}
