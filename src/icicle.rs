use crate::{F, DeviceType};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    msm::{msm, MSMConfig, MSM},
    ntt::{ntt_inplace, NTTConfig, NTTDir, NTT, get_root_of_unity, initialize_domain, NTTInitDomainConfig},
    traits::FieldImpl,
};

use icicle_runtime::{
    memory::{DeviceSlice, HostSlice, HostOrDeviceSlice}
};

use std::time::Instant;

pub fn icicle_initialize_domain(size: u64) {
    let root_of_unity: F = get_root_of_unity(size);
    let cfg = NTTInitDomainConfig::default();
    initialize_domain(root_of_unity, &cfg).unwrap();
}

pub fn icicle_ntt(vec: &mut DeviceSlice<F>, inverse: bool, cfg: &NTTConfig<F>)
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

pub fn icicle_msm<C: Curve + MSM<C>>(
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

pub fn set_device(device_type: DeviceType) {
    match device_type {
        DeviceType::Cpu => {/* noop as this is the default in icicle */},
        DeviceType::Metal | DeviceType::CpuMetal => {
            let device = icicle_runtime::Device::new("METAL", 0 /* =device_id*/);
            icicle_runtime::set_device(&device).unwrap();
        }
    }
}
