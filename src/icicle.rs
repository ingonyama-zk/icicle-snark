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

#[cfg(feature = "android")]
use std::time::Instant;
#[cfg(feature = "android")]
use log::debug;

pub fn icicle_initialize_domain(size: u64) {
    let root_of_unity: F = get_root_of_unity(size);
    let cfg = NTTInitDomainConfig::default();
    #[cfg(feature = "android")]
    let start = Instant::now();
    initialize_domain(root_of_unity, &cfg).unwrap();
    #[cfg(feature = "android")]
    let duration = start.elapsed();
    #[cfg(feature = "android")]
    debug!("initialize_domain took {:?}", duration);
}

pub fn icicle_ntt(vec: &mut DeviceSlice<F>, inverse: bool, cfg: &NTTConfig<F>, ntt_name: &str)
where
    <F as FieldImpl>::Config: NTT<F, F>,
{
    let dir = if inverse {
        NTTDir::kInverse
    } else {
        NTTDir::kForward
    };

    #[cfg(feature = "android")]
    let start = Instant::now();
    ntt_inplace(vec, dir, cfg).unwrap();
    #[cfg(feature = "android")]
    let duration = start.elapsed();
    #[cfg(feature = "android")]
    debug!("{} ntt_inplace {} of size {} took {:?}", if inverse { "inv" } else { "fwd" }, ntt_name, vec.len(), duration);
}

pub fn icicle_msm<C: Curve + MSM<C>>(
    scalars: &(impl HostOrDeviceSlice<C::ScalarField> + ?Sized),
    points: &(impl HostOrDeviceSlice<Affine<C>> + ?Sized),
    msm_config: &MSMConfig,
    msm_name: &str,
) -> Projective<C>
{
    let mut msm_result = vec![Projective::zero(); 1];
    #[cfg(feature = "android")]
    let start = Instant::now();
    msm(scalars, points, &msm_config, HostSlice::from_mut_slice(&mut msm_result[..])).unwrap();
    #[cfg(feature = "android")]
    let duration = start.elapsed();
    #[cfg(feature = "android")]
    debug!("msm {} of size {} took {:?}", msm_name, scalars.len(), duration);

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
