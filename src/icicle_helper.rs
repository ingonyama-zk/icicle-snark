use crate::F;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    msm::{msm, MSMConfig, MSM},
    ntt::{ntt_inplace, NTTConfig, NTTDir, NTT},
    traits::FieldImpl,
};

use icicle_runtime::{
    memory::{DeviceSlice, HostSlice, HostOrDeviceSlice}
};

use std::time::Instant;

pub fn ntt_helper(vec: &mut DeviceSlice<F>, inverse: bool, cfg: &NTTConfig<F>)
where
    <F as FieldImpl>::Config: NTT<F, F>,
{
    let dir = if inverse {
        NTTDir::kInverse
    } else {
        NTTDir::kForward
    };

    let timer = Instant::now();
    ntt_inplace(vec, dir, cfg).unwrap();
    println!("NTT took:\t\t{:?}", timer.elapsed());
}

pub fn msm_helper<C: Curve + MSM<C>>(
    scalars: &(impl HostOrDeviceSlice<C::ScalarField> + ?Sized),
    points: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
    msm_config: &MSMConfig,
    msm_name: &str,
) -> Projective<C>
{
    let mut msm_result = vec![Projective::zero(); 1];
    let timer = Instant::now();
    msm(scalars, points, &msm_config, HostSlice::from_mut_slice(&mut msm_result[..])).unwrap();
    println!("{:?} MSM took:\t\t{:?}", msm_name, timer.elapsed());

    msm_result[0]
}
